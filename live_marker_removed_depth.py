import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


def load_calibration(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    return {
        "camera_matrix": np.asarray(cfg["camera_matrix"], dtype=np.float64),
        "dist_coeffs": np.asarray(cfg["dist_coeffs"], dtype=np.float64).reshape(-1, 1),
        "image_size": tuple(cfg["image_size"]),
        "rvec": np.asarray(cfg["rvecs"], dtype=np.float64).reshape(3, 1),
        "tvec": np.asarray(cfg["tvecs"], dtype=np.float64).reshape(3, 1),
    }


def load_reference_markers(path):
    markers = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row.get("valid", 1)) == 0:
                continue
            markers.append(
                {
                    "id": int(row["id"]),
                    "pixel_x": float(row["pixel_x"]),
                    "pixel_y": float(row["pixel_y"]),
                    "point": np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64),
                }
            )
    markers.sort(key=lambda item: item["id"])
    if not markers:
        raise ValueError(f"No valid reference markers found in {path}")
    return markers


def reset_reference_pixels(reference_markers, matches):
    updated = []
    for idx, marker in enumerate(reference_markers):
        current = matches.get(idx)
        if current is None:
            updated.append(marker)
            continue
        copied = dict(marker)
        copied["pixel_x"] = float(current["pixel_x"])
        copied["pixel_y"] = float(current["pixel_y"])
        updated.append(copied)
    return updated


def component_markers(mask, min_area, max_area):
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    markers = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if min_area <= area <= max_area and 2 <= w <= 50 and 2 <= h <= 50:
            markers.append(
                {
                    "pixel_x": float(centroids[label][0]),
                    "pixel_y": float(centroids[label][1]),
                    "area": area,
                    "bbox_x": x,
                    "bbox_y": y,
                    "bbox_w": w,
                    "bbox_h": h,
                }
            )
    markers.sort(key=lambda item: (item["pixel_y"], item["pixel_x"]))
    return markers


def keep_expected_marker_count(markers, expected_count):
    if expected_count <= 0 or len(markers) <= expected_count:
        return markers
    ranked = sorted(markers, key=lambda item: item["area"], reverse=True)[:expected_count]
    ranked.sort(key=lambda item: (item["pixel_y"], item["pixel_x"]))
    return ranked


def build_removal_mask(image_shape, markers, mask_radius, mask_padding, mask_dilate_iterations):
    removal_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for marker in markers:
        radius = max(mask_radius, int(max(marker["bbox_w"], marker["bbox_h"]) * 0.5) + mask_padding)
        center = (int(round(marker["pixel_x"])), int(round(marker["pixel_y"])))
        cv2.ellipse(removal_mask, center, (radius, radius), 0, 0, 360, 255, -1)
    if mask_dilate_iterations > 0 and cv2.countNonZero(removal_mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        removal_mask = cv2.dilate(removal_mask, kernel, iterations=mask_dilate_iterations)
    return removal_mask


def detect_marker_mask(
    image,
    diff_threshold,
    diff_scale,
    large_kernel,
    small_kernel,
    min_area,
    max_area,
    mask_radius,
    mask_padding,
    mask_dilate_iterations,
):
    small = cv2.pyrDown(image)
    broad = cv2.GaussianBlur(small, (large_kernel, large_kernel), 0)
    local = cv2.GaussianBlur(small, (small_kernel, small_kernel), 0)
    diff = cv2.subtract(local, broad)
    diff = np.clip(diff.astype(np.float32) * diff_scale, 0, 255).astype(np.uint8)

    b, g, r = cv2.split(diff)
    detect_mask = (
        ((b > diff_threshold) & (g > diff_threshold))
        | ((b > diff_threshold) & (r > diff_threshold))
        | ((g > diff_threshold) & (r > diff_threshold))
    ).astype(np.uint8) * 255
    detect_mask = cv2.resize(detect_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    detect_mask = cv2.morphologyEx(detect_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    markers = component_markers(detect_mask, min_area, max_area)

    removal_mask = build_removal_mask(
        image.shape,
        markers,
        mask_radius,
        mask_padding,
        mask_dilate_iterations,
    )
    return markers, detect_mask, removal_mask


def precompute_hemisphere_depth(width, height, calibration, sphere_center, sphere_radius, depth_mode):
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    pixels = np.stack([xs.ravel(), ys.ravel()], axis=1).reshape(-1, 1, 2)

    normalized = cv2.undistortPoints(
        pixels,
        calibration["camera_matrix"],
        calibration["dist_coeffs"],
    ).reshape(-1, 2)
    dirs_camera = np.c_[normalized, np.ones(len(normalized), dtype=np.float64)]
    dirs_camera /= np.linalg.norm(dirs_camera, axis=1, keepdims=True)

    rotation, _ = cv2.Rodrigues(calibration["rvec"])
    ray_origin = (-rotation.T @ calibration["tvec"]).reshape(3)
    dirs_object = (rotation.T @ dirs_camera.T).T
    dirs_object /= np.linalg.norm(dirs_object, axis=1, keepdims=True)

    center = np.asarray(sphere_center, dtype=np.float64)
    oc = ray_origin - center
    b = 2.0 * (dirs_object @ oc)
    c = float(oc @ oc - sphere_radius * sphere_radius)
    disc = b * b - 4.0 * c

    depth = np.full(len(dirs_object), np.nan, dtype=np.float32)
    hit_points = np.full((len(dirs_object), 3), np.nan, dtype=np.float64)

    valid_disc = disc >= 0.0
    sqrt_disc = np.zeros_like(disc)
    sqrt_disc[valid_disc] = np.sqrt(disc[valid_disc])
    t_near = (-b - sqrt_disc) / 2.0
    t_far = (-b + sqrt_disc) / 2.0

    for t_candidate in (t_near, t_far):
        candidate_valid = valid_disc & (t_candidate > 0.0) & np.isnan(depth)
        if not np.any(candidate_valid):
            continue
        points = ray_origin + t_candidate[candidate_valid, None] * dirs_object[candidate_valid]
        upper = points[:, 2] >= center[2] - 1e-6
        idx = np.flatnonzero(candidate_valid)[upper]
        hit_points[idx] = points[upper]

        if depth_mode == "height":
            values = points[upper, 2] - center[2]
        elif depth_mode == "object-z":
            values = points[upper, 2]
        elif depth_mode == "camera-z":
            points_camera = (rotation @ points[upper].T + calibration["tvec"]).T
            values = points_camera[:, 2]
        elif depth_mode == "camera-range":
            values = t_candidate[idx]
        else:
            raise ValueError(f"Unsupported depth mode: {depth_mode}")
        depth[idx] = values.astype(np.float32)

    depth = depth.reshape(height, width)
    valid_mask = np.isfinite(depth).astype(np.uint8) * 255
    depth_filled = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
    return depth_filled, valid_mask, ray_origin, hit_points.reshape(height, width, 3)


def match_markers_to_reference(reference_markers, current_markers, max_distance):
    ref_pixels = np.array([[m["pixel_x"], m["pixel_y"]] for m in reference_markers], dtype=np.float64)
    cur_pixels = np.array([[m["pixel_x"], m["pixel_y"]] for m in current_markers], dtype=np.float64)
    matches = {}
    if len(ref_pixels) == 0 or len(cur_pixels) == 0:
        return matches

    costs = np.linalg.norm(ref_pixels[:, None, :] - cur_pixels[None, :, :], axis=2)
    if linear_sum_assignment is not None:
        ref_ids, cur_ids = linear_sum_assignment(costs)
        pairs = zip(ref_ids, cur_ids)
    else:
        pairs = []
        used_ref = set()
        used_cur = set()
        for ref_id, cur_id in sorted(
            ((r, c) for r in range(costs.shape[0]) for c in range(costs.shape[1])),
            key=lambda pair: costs[pair[0], pair[1]],
        ):
            if ref_id in used_ref or cur_id in used_cur:
                continue
            used_ref.add(ref_id)
            used_cur.add(cur_id)
            pairs.append((ref_id, cur_id))

    for ref_id, cur_id in pairs:
        distance = float(costs[ref_id, cur_id])
        if distance <= max_distance:
            matches[int(ref_id)] = {**current_markers[int(cur_id)], "match_distance": distance}
    return matches


def project_points(points, calibration):
    projected, _ = cv2.projectPoints(
        np.asarray(points, dtype=np.float64).reshape(-1, 3),
        calibration["rvec"],
        calibration["tvec"],
        calibration["camera_matrix"],
        calibration["dist_coeffs"],
    )
    return projected.reshape(-1, 2)


def estimate_marker_displacements(
    reference_markers,
    matches,
    calibration,
    sphere_center,
    sphere_radius,
    max_normal_displacement,
    deadzone,
):
    center = np.asarray(sphere_center, dtype=np.float64)
    displacements = np.zeros(len(reference_markers), dtype=np.float32)
    valid = np.zeros(len(reference_markers), dtype=bool)
    eps = max(sphere_radius * 1e-3, 1e-3)

    for idx, marker in enumerate(reference_markers):
        current = matches.get(idx)
        if current is None:
            continue

        point = marker["point"]
        normal = point - center
        normal /= np.linalg.norm(normal) + 1e-12

        # The reference detector pixel is more reliable than a synthetic
        # reprojected pixel for the offset; the camera model is used for scale.
        ref_pixel = np.array([marker["pixel_x"], marker["pixel_y"]], dtype=np.float64)
        cur_pixel = np.array([current["pixel_x"], current["pixel_y"]], dtype=np.float64)
        duv = cur_pixel - ref_pixel

        proj0 = project_points([point], calibration)[0]
        proj1 = project_points([point + eps * normal], calibration)[0]
        jacobian = (proj1 - proj0) / eps
        denom = float(jacobian @ jacobian)
        if denom < 1e-12:
            continue

        normal_delta = float((jacobian @ duv) / denom)
        normal_delta = float(np.clip(normal_delta, -max_normal_displacement, max_normal_displacement))
        if abs(normal_delta) < deadzone:
            normal_delta = 0.0

        displacements[idx] = normal_delta * normal[2]
        valid[idx] = True

    return displacements, valid


def precompute_marker_interpolation(surface_points, valid_mask, reference_markers, power):
    valid_flat = (valid_mask.reshape(-1) > 0)
    surface_xy = surface_points.reshape(-1, 3)[valid_flat, :2].astype(np.float32)
    marker_xy = np.array([m["point"][:2] for m in reference_markers], dtype=np.float32)
    diff = surface_xy[:, None, :] - marker_xy[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    weights = 1.0 / np.maximum(dist2, 1e-6) ** (power * 0.5)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return np.flatnonzero(valid_flat), weights.astype(np.float32)


def interpolate_marker_depth_delta(image_shape, valid_indices, weights, marker_delta, marker_valid):
    delta = np.zeros(image_shape[0] * image_shape[1], dtype=np.float32)
    if np.count_nonzero(marker_valid) >= 3:
        selected_weights = weights[:, marker_valid]
        selected_weights /= np.maximum(selected_weights.sum(axis=1, keepdims=True), 1e-12)
        delta[valid_indices] = selected_weights @ marker_delta[marker_valid]
    return delta.reshape(image_shape).astype(np.float32)


def odd_kernel_size(value):
    value = max(3, int(value))
    return value if value % 2 else value + 1


def remove_markers_from_depth(
    depth,
    valid_mask,
    marker_mask,
    repair_mode,
    inpaint_radius,
    smooth_kernel,
    feather_sigma,
):
    marker_on_depth = cv2.bitwise_and(marker_mask, valid_mask)
    if cv2.countNonZero(marker_on_depth) == 0:
        return depth.copy(), marker_on_depth

    if repair_mode == "model":
        removed = depth.copy()
    elif repair_mode == "smooth":
        kernel_size = odd_kernel_size(smooth_kernel)
        smooth = cv2.GaussianBlur(depth.astype(np.float32), (kernel_size, kernel_size), 0)
        alpha = (marker_on_depth > 0).astype(np.float32)
        if feather_sigma > 0:
            alpha = cv2.GaussianBlur(alpha, (0, 0), feather_sigma)
            alpha = np.clip(alpha * 1.35, 0.0, 1.0)
        removed = depth.astype(np.float32) * (1.0 - alpha) + smooth * alpha
    elif repair_mode == "inpaint":
        removed = cv2.inpaint(depth.astype(np.float32), marker_on_depth, inpaint_radius, cv2.INPAINT_TELEA)
    else:
        raise ValueError(f"Unsupported repair mode: {repair_mode}")

    valid_values = depth[valid_mask > 0]
    if len(valid_values):
        removed[valid_mask > 0] = np.clip(removed[valid_mask > 0], float(valid_values.min()), float(valid_values.max()))
    removed[valid_mask == 0] = 0.0
    return removed.astype(np.float32), marker_on_depth


def colorize_depth(depth, valid_mask, depth_min=None, depth_max=None):
    valid = valid_mask > 0
    if depth_min is None:
        depth_min = float(np.min(depth[valid])) if np.any(valid) else 0.0
    if depth_max is None:
        depth_max = float(np.max(depth[valid])) if np.any(valid) else 1.0
    if abs(depth_max - depth_min) < 1e-9:
        depth_max = depth_min + 1.0
    scaled = np.clip((depth - depth_min) / (depth_max - depth_min), 0.0, 1.0)
    gray = (scaled * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[valid_mask == 0] = 0
    return color


def colorize_signed(values, valid_mask, limit):
    limit = max(float(limit), 1e-6)
    scaled = np.clip((values / limit) * 127.0 + 128.0, 0.0, 255.0).astype(np.uint8)
    color = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    color[valid_mask == 0] = 0
    return color


def draw_markers(image, markers):
    output = image.copy()
    for idx, marker in enumerate(markers):
        center = (int(round(marker["pixel_x"])), int(round(marker["pixel_y"])))
        cv2.circle(output, center, 5, (0, 220, 0), 1)
        cv2.putText(
            output,
            str(idx),
            (center[0] + 7, center[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )
    return output


def save_outputs(prefix, frame, marker_mask, depth, depth_view, deformation=None, deformation_view=None):
    cv2.imwrite(f"{prefix}_frame.jpg", frame)
    cv2.imwrite(f"{prefix}_marker_mask.png", marker_mask)
    cv2.imwrite(f"{prefix}_depth_marker_removed.png", depth_view)
    np.save(f"{prefix}_depth_marker_removed.npy", depth)
    if deformation is not None:
        np.save(f"{prefix}_deformation.npy", deformation)
    if deformation_view is not None:
        cv2.imwrite(f"{prefix}_deformation.png", deformation_view)


def main():
    parser = argparse.ArgumentParser(description="Live marker-removed depth map for the calibrated hemisphere.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--calibration", default="calibration_new.cfg")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--sphere-center", nargs=3, type=float, default=[0.0, 0.0, 6.0])
    parser.add_argument("--sphere-radius", type=float, default=15.1)
    parser.add_argument("--reference-markers", default="ref_marker_points_3d.csv")
    parser.add_argument("--expected-markers", type=int, default=25)
    parser.add_argument("--max-match-distance", type=float, default=80.0)
    parser.add_argument("--live-depth-source", choices=["markers", "model"], default="markers")
    parser.add_argument("--max-normal-displacement", type=float, default=5.0)
    parser.add_argument("--deadzone", type=float, default=0.02)
    parser.add_argument("--idw-power", type=float, default=2.0)
    parser.add_argument("--deformation-smooth-sigma", type=float, default=1.0)
    parser.add_argument("--temporal-alpha", type=float, default=0.45)
    parser.add_argument("--deformation-display-limit", type=float, default=2.0)
    parser.add_argument("--depth-mode", choices=["height", "object-z", "camera-z", "camera-range"], default="height")
    parser.add_argument("--diff-threshold", type=int, default=120)
    parser.add_argument("--diff-scale", type=float, default=15.0)
    parser.add_argument("--diff-large-kernel", type=int, default=15)
    parser.add_argument("--diff-small-kernel", type=int, default=3)
    parser.add_argument("--min-area", type=int, default=3)
    parser.add_argument("--max-area", type=int, default=500)
    parser.add_argument("--mask-radius", type=int, default=9)
    parser.add_argument("--mask-padding", type=int, default=5)
    parser.add_argument("--mask-dilate-iterations", type=int, default=0)
    parser.add_argument(
        "--depth-repair-mode",
        choices=["model", "smooth", "inpaint"],
        default="model",
        help="model keeps the clean analytical hemisphere depth; smooth/inpaint repair only marker regions.",
    )
    parser.add_argument("--inpaint-radius", type=float, default=5.0)
    parser.add_argument("--smooth-kernel", type=int, default=41)
    parser.add_argument("--feather-sigma", type=float, default=2.5)
    parser.add_argument("--output-prefix", default="live_depth")
    parser.add_argument("--once", action="store_true", help="Capture one frame, save outputs, and exit.")
    parser.add_argument("--no-window", action="store_true", help="Do not open realtime display windows.")
    args = parser.parse_args()

    calibration = load_calibration(args.calibration)
    if calibration["image_size"] != (args.width, args.height):
        raise ValueError(
            f"Calibration image_size is {calibration['image_size']}, "
            f"but requested capture is {(args.width, args.height)}"
        )

    reference_markers = load_reference_markers(args.reference_markers)
    if len(reference_markers) != args.expected_markers:
        raise ValueError(
            f"Expected {args.expected_markers} reference markers, "
            f"but found {len(reference_markers)} in {args.reference_markers}"
        )

    base_depth, valid_mask, camera_origin, surface_points = precompute_hemisphere_depth(
        args.width,
        args.height,
        calibration,
        args.sphere_center,
        args.sphere_radius,
        args.depth_mode,
    )
    valid_indices, marker_weights = precompute_marker_interpolation(
        surface_points,
        valid_mask,
        reference_markers,
        args.idw_power,
    )
    depth_valid = base_depth[valid_mask > 0]
    depth_min = float(depth_valid.min())
    depth_max = float(depth_valid.max())
    deformation_state = np.zeros_like(base_depth, dtype=np.float32)

    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    last_save = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("Camera opened but returned no frame")
            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)

            markers, detect_mask, marker_mask = detect_marker_mask(
                frame,
                args.diff_threshold,
                args.diff_scale,
                args.diff_large_kernel,
                args.diff_small_kernel,
                args.min_area,
                args.max_area,
                args.mask_radius,
                args.mask_padding,
                args.mask_dilate_iterations,
            )
            markers = keep_expected_marker_count(markers, args.expected_markers)
            marker_mask = build_removal_mask(
                frame.shape,
                markers,
                args.mask_radius,
                args.mask_padding,
                args.mask_dilate_iterations,
            )
            matches = match_markers_to_reference(reference_markers, markers, args.max_match_distance)
            marker_delta, marker_valid = estimate_marker_displacements(
                reference_markers,
                matches,
                calibration,
                args.sphere_center,
                args.sphere_radius,
                args.max_normal_displacement,
                args.deadzone,
            )

            if args.live_depth_source == "markers":
                deformation = interpolate_marker_depth_delta(
                    base_depth.shape,
                    valid_indices,
                    marker_weights,
                    marker_delta,
                    marker_valid,
                )
                if args.deformation_smooth_sigma > 0:
                    deformation = cv2.GaussianBlur(deformation, (0, 0), args.deformation_smooth_sigma)
                if 0.0 < args.temporal_alpha < 1.0:
                    deformation_state = (
                        args.temporal_alpha * deformation + (1.0 - args.temporal_alpha) * deformation_state
                    ).astype(np.float32)
                else:
                    deformation_state = deformation.astype(np.float32)
                current_depth = base_depth + deformation_state
                current_depth[valid_mask == 0] = 0.0
            else:
                deformation_state.fill(0.0)
                current_depth = base_depth

            depth_removed, depth_marker_mask = remove_markers_from_depth(
                current_depth,
                valid_mask,
                marker_mask,
                args.depth_repair_mode,
                args.inpaint_radius,
                args.smooth_kernel,
                args.feather_sigma,
            )
            depth_view = colorize_depth(depth_removed, valid_mask, depth_min, depth_max)
            deformation_view = colorize_signed(
                deformation_state,
                valid_mask,
                args.deformation_display_limit,
            )
            marker_view = draw_markers(frame, markers)

            cv2.putText(
                marker_view,
                f"markers: {len(markers)}",
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                marker_view,
                f"matched: {len(matches)}",
                (12, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if len(matches) == args.expected_markers else (0, 180, 255),
                2,
                cv2.LINE_AA,
            )

            if args.once:
                save_outputs(
                    args.output_prefix,
                    marker_view,
                    depth_marker_mask,
                    depth_removed,
                    depth_view,
                    deformation_state,
                    deformation_view,
                )
                stats = {
                    "markers": len(markers),
                    "matched_markers": len(matches),
                    "valid_marker_displacements": int(np.count_nonzero(marker_valid)),
                    "live_depth_source": args.live_depth_source,
                    "depth_mode": args.depth_mode,
                    "depth_repair_mode": args.depth_repair_mode,
                    "marker_delta_z_min": float(marker_delta[marker_valid].min()) if np.any(marker_valid) else 0.0,
                    "marker_delta_z_max": float(marker_delta[marker_valid].max()) if np.any(marker_valid) else 0.0,
                    "depth_min": depth_min,
                    "depth_max": depth_max,
                    "mask_radius": args.mask_radius,
                    "mask_padding": args.mask_padding,
                    "mask_dilate_iterations": args.mask_dilate_iterations,
                    "camera_origin_object_coordinates": camera_origin.tolist(),
                    "saved_prefix": args.output_prefix,
                }
                Path(f"{args.output_prefix}_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
                print(json.dumps(stats, indent=2))
                break

            if not args.no_window:
                cv2.imshow("raw + markers", marker_view)
                cv2.imshow("marker mask on depth", depth_marker_mask)
                cv2.imshow("marker removed depth", depth_view)
                cv2.imshow("live deformation", deformation_view)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    if len(matches) == args.expected_markers:
                        reference_markers = reset_reference_pixels(reference_markers, matches)
                        deformation_state.fill(0.0)
                        print("reference marker pixels reset from current frame")
                    else:
                        print(f"reference reset skipped: matched {len(matches)}/{args.expected_markers}")
                if key == ord("s"):
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    last_save = f"{args.output_prefix}_{stamp}"
                    save_outputs(
                        last_save,
                        marker_view,
                        depth_marker_mask,
                        depth_removed,
                        depth_view,
                        deformation_state,
                        deformation_view,
                    )
                    print(f"saved {last_save}_*.png/jpg/npy")
            else:
                time.sleep(0.01)
    finally:
        cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()
        if last_save:
            print(f"last_save={last_save}")


if __name__ == "__main__":
    main()
