import argparse
import csv
import json
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


def scale_camera_matrix(camera_matrix, src_size, dst_size):
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    scaled = camera_matrix.copy()
    sx = dst_w / src_w
    sy = dst_h / src_h
    scaled[0, 0] *= sx
    scaled[0, 2] *= sx
    scaled[1, 1] *= sy
    scaled[1, 2] *= sy
    return scaled, sx, sy


def component_markers(mask, min_area, max_area):
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    markers = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if min_area <= area <= max_area and 1 <= w <= 50 and 1 <= h <= 50:
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


def detect_white_markers_hsv(image, value_threshold, saturation_threshold, min_area, max_area):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] <= saturation_threshold) & (hsv[:, :, 2] >= value_threshold)).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return component_markers(mask, min_area, max_area), mask


def detect_white_markers_diff(
    image,
    diff_threshold,
    diff_scale,
    large_kernel,
    small_kernel,
    min_area,
    max_area,
):
    # This follows the marker_processing.md pattern: downsample, subtract two
    # Gaussian blur scales to isolate local bright dots, amplify, threshold at
    # channel level, then clean the binary mask before center extraction.
    small = cv2.pyrDown(image)
    broad = cv2.GaussianBlur(small, (large_kernel, large_kernel), 0)
    local = cv2.GaussianBlur(small, (small_kernel, small_kernel), 0)
    diff = cv2.subtract(local, broad)
    diff = np.clip(diff.astype(np.float32) * diff_scale, 0, 255).astype(np.uint8)
    b, g, r = cv2.split(diff)
    mask = (
        ((b > diff_threshold) & (g > diff_threshold))
        | ((b > diff_threshold) & (r > diff_threshold))
        | ((g > diff_threshold) & (r > diff_threshold))
    ).astype(np.uint8) * 255
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return component_markers(mask, min_area, max_area), mask


def rays_in_object_coordinates(pixel_points, camera_matrix, dist_coeffs, rvec, tvec):
    pixels = np.asarray(pixel_points, dtype=np.float64).reshape(-1, 1, 2)
    normalized = cv2.undistortPoints(pixels, camera_matrix, dist_coeffs).reshape(-1, 2)
    directions_camera = np.c_[normalized, np.ones(len(normalized))]
    directions_camera /= np.linalg.norm(directions_camera, axis=1, keepdims=True)

    rotation, _ = cv2.Rodrigues(rvec)
    camera_center_object = (-rotation.T @ tvec).reshape(3)
    directions_object = (rotation.T @ directions_camera.T).T
    directions_object /= np.linalg.norm(directions_object, axis=1, keepdims=True)
    return camera_center_object, directions_object


def intersect_sphere(ray_origin, ray_dirs, sphere_center, sphere_radius, min_z):
    oc = ray_origin - sphere_center
    b = 2.0 * (ray_dirs @ oc)
    c = float(oc @ oc - sphere_radius * sphere_radius)
    disc = b * b - 4.0 * c
    points = []
    valid = []

    for i, d in enumerate(ray_dirs):
        if disc[i] < 0:
            points.append([np.nan, np.nan, np.nan])
            valid.append(False)
            continue
        sqrt_disc = float(np.sqrt(disc[i]))
        candidates = [(-b[i] - sqrt_disc) / 2.0, (-b[i] + sqrt_disc) / 2.0]
        candidates = [t for t in candidates if t > 0]
        hit = None
        for t in sorted(candidates):
            p = ray_origin + t * d
            if p[2] >= min_z - 1e-6:
                hit = p
                break
        if hit is None:
            points.append([np.nan, np.nan, np.nan])
            valid.append(False)
        else:
            points.append(hit.tolist())
            valid.append(True)

    return np.asarray(points, dtype=np.float64), np.asarray(valid, dtype=bool)


def draw_debug(image, markers, points3d, valid, output):
    canvas = image.copy()
    for idx, marker in enumerate(markers):
        x = int(round(marker["pixel_x"]))
        y = int(round(marker["pixel_y"]))
        color = (0, 220, 0) if valid[idx] else (0, 0, 255)
        cv2.circle(canvas, (x, y), 8, color, 2)
        cv2.putText(canvas, str(idx), (x + 9, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(output), canvas)


def write_csv(path, markers, points3d, valid):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "pixel_x", "pixel_y", "area", "valid", "x", "y", "z"])
        for idx, marker in enumerate(markers):
            row = [
                idx,
                f"{marker['pixel_x']:.6f}",
                f"{marker['pixel_y']:.6f}",
                marker["area"],
                int(valid[idx]),
            ]
            if valid[idx]:
                row.extend(f"{value:.6f}" for value in points3d[idx])
            else:
                row.extend(["", "", ""])
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Detect white markers and project them onto the recovered hemisphere.")
    parser.add_argument("--image", default="data/reference/ref.jpg")
    parser.add_argument("--calibration", default="config/calibration_new.cfg")
    parser.add_argument("--csv", default="data/reference/ref_marker_points_3d.csv")
    parser.add_argument("--debug-image", default="data/reference/ref_marker_detection.jpg")
    parser.add_argument("--stats", default="data/reference/ref_marker_projection_stats.json")
    parser.add_argument("--sphere-center", nargs=3, type=float, default=[0.0, 0.0, 6.0])
    parser.add_argument("--sphere-radius", type=float, default=15.1)
    parser.add_argument("--detection-mode", choices=["diff", "hsv"], default="diff")
    parser.add_argument("--value-threshold", type=int, default=160)
    parser.add_argument("--saturation-threshold", type=int, default=80)
    parser.add_argument("--diff-threshold", type=int, default=120)
    parser.add_argument("--diff-scale", type=float, default=15.0)
    parser.add_argument("--diff-large-kernel", type=int, default=15)
    parser.add_argument("--diff-small-kernel", type=int, default=3)
    parser.add_argument("--min-area", type=int, default=3)
    parser.add_argument("--max-area", type=int, default=500)
    parser.add_argument(
        "--intrinsics-mode",
        choices=["scale", "raw"],
        default="scale",
        help="scale adjusts calibration intrinsics from cfg image_size to the actual image size.",
    )
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    image_h, image_w = image.shape[:2]

    calib = load_calibration(args.calibration)
    camera_matrix = calib["camera_matrix"]
    scale = [1.0, 1.0]
    if args.intrinsics_mode == "scale":
        camera_matrix, sx, sy = scale_camera_matrix(camera_matrix, calib["image_size"], (image_w, image_h))
        scale = [sx, sy]

    if args.detection_mode == "diff":
        markers, mask = detect_white_markers_diff(
            image,
            args.diff_threshold,
            args.diff_scale,
            args.diff_large_kernel,
            args.diff_small_kernel,
            args.min_area,
            args.max_area,
        )
    else:
        markers, mask = detect_white_markers_hsv(
            image,
            args.value_threshold,
            args.saturation_threshold,
            args.min_area,
            args.max_area,
        )
    pixel_points = [(marker["pixel_x"], marker["pixel_y"]) for marker in markers]

    if pixel_points:
        camera_origin, ray_dirs = rays_in_object_coordinates(
            pixel_points,
            camera_matrix,
            calib["dist_coeffs"],
            calib["rvec"],
            calib["tvec"],
        )
        points3d, valid = intersect_sphere(
            camera_origin,
            ray_dirs,
            np.asarray(args.sphere_center, dtype=np.float64),
            args.sphere_radius,
            args.sphere_center[2],
        )
    else:
        camera_origin = np.full(3, np.nan)
        points3d = np.empty((0, 3), dtype=np.float64)
        valid = np.empty(0, dtype=bool)

    write_csv(args.csv, markers, points3d, valid)
    draw_debug(image, markers, points3d, valid, args.debug_image)

    stats = {
        "image": args.image,
        "image_size_actual": [image_w, image_h],
        "calibration": args.calibration,
        "image_size_in_calibration": list(calib["image_size"]),
        "intrinsics_mode": args.intrinsics_mode,
        "intrinsics_scale": scale,
        "camera_matrix_used": camera_matrix.tolist(),
        "sphere_center": args.sphere_center,
        "sphere_radius": args.sphere_radius,
        "marker_detection": {
            "mode": args.detection_mode,
            "value_threshold": args.value_threshold,
            "saturation_threshold": args.saturation_threshold,
            "diff_threshold": args.diff_threshold,
            "diff_scale": args.diff_scale,
            "diff_large_kernel": args.diff_large_kernel,
            "diff_small_kernel": args.diff_small_kernel,
            "min_area": args.min_area,
            "max_area": args.max_area,
            "detected_markers": int(len(markers)),
            "valid_sphere_intersections": int(valid.sum()),
        },
        "camera_origin_object_coordinates": camera_origin.tolist(),
        "outputs": {
            "csv": args.csv,
            "debug_image": args.debug_image,
        },
    }
    Path(args.stats).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
