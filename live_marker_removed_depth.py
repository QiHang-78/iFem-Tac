import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


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


def detect_marker_mask(
    image,
    diff_threshold,
    diff_scale,
    large_kernel,
    small_kernel,
    min_area,
    max_area,
    mask_radius,
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

    removal_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for marker in markers:
        radius = max(mask_radius, int(max(marker["bbox_w"], marker["bbox_h"]) * 0.5) + 3)
        center = (int(round(marker["pixel_x"])), int(round(marker["pixel_y"])))
        cv2.ellipse(removal_mask, center, (radius, radius), 0, 0, 360, 255, -1)

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
    return depth_filled, valid_mask, ray_origin


def remove_markers_from_depth(depth, valid_mask, marker_mask, inpaint_radius):
    marker_on_depth = cv2.bitwise_and(marker_mask, valid_mask)
    if cv2.countNonZero(marker_on_depth) == 0:
        return depth.copy(), marker_on_depth
    removed = cv2.inpaint(depth.astype(np.float32), marker_on_depth, inpaint_radius, cv2.INPAINT_TELEA)
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


def draw_markers(image, markers):
    output = image.copy()
    for idx, marker in enumerate(markers):
        center = (int(round(marker["pixel_x"])), int(round(marker["pixel_y"])))
        cv2.circle(output, center, 8, (0, 220, 0), 2)
        cv2.putText(
            output,
            str(idx),
            (center[0] + 9, center[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )
    return output


def save_outputs(prefix, frame, marker_mask, depth, depth_view):
    cv2.imwrite(f"{prefix}_frame.jpg", frame)
    cv2.imwrite(f"{prefix}_marker_mask.png", marker_mask)
    cv2.imwrite(f"{prefix}_depth_marker_removed.png", depth_view)
    np.save(f"{prefix}_depth_marker_removed.npy", depth)


def main():
    parser = argparse.ArgumentParser(description="Live marker-removed depth map for the calibrated hemisphere.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--calibration", default="calibration_new.cfg")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--sphere-center", nargs=3, type=float, default=[0.0, 0.0, 6.0])
    parser.add_argument("--sphere-radius", type=float, default=15.1)
    parser.add_argument("--depth-mode", choices=["height", "object-z", "camera-z", "camera-range"], default="height")
    parser.add_argument("--diff-threshold", type=int, default=120)
    parser.add_argument("--diff-scale", type=float, default=15.0)
    parser.add_argument("--diff-large-kernel", type=int, default=15)
    parser.add_argument("--diff-small-kernel", type=int, default=3)
    parser.add_argument("--min-area", type=int, default=3)
    parser.add_argument("--max-area", type=int, default=500)
    parser.add_argument("--mask-radius", type=int, default=8)
    parser.add_argument("--inpaint-radius", type=float, default=5.0)
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

    base_depth, valid_mask, camera_origin = precompute_hemisphere_depth(
        args.width,
        args.height,
        calibration,
        args.sphere_center,
        args.sphere_radius,
        args.depth_mode,
    )
    depth_valid = base_depth[valid_mask > 0]
    depth_min = float(depth_valid.min())
    depth_max = float(depth_valid.max())

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
            )
            depth_removed, depth_marker_mask = remove_markers_from_depth(
                base_depth,
                valid_mask,
                marker_mask,
                args.inpaint_radius,
            )
            depth_view = colorize_depth(depth_removed, valid_mask, depth_min, depth_max)
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

            if args.once:
                save_outputs(args.output_prefix, marker_view, depth_marker_mask, depth_removed, depth_view)
                stats = {
                    "markers": len(markers),
                    "depth_mode": args.depth_mode,
                    "depth_min": depth_min,
                    "depth_max": depth_max,
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
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("s"):
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    last_save = f"{args.output_prefix}_{stamp}"
                    save_outputs(last_save, marker_view, depth_marker_mask, depth_removed, depth_view)
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
