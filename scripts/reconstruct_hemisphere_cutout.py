import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def load_obj(path):
    vertices = []
    triangles = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.split()
            if not parts or line.startswith("#"):
                continue
            if parts[0] == "v":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                face = []
                for token in parts[1:]:
                    raw = token.split("/")[0]
                    if raw:
                        index = int(raw)
                        face.append(index - 1 if index > 0 else len(vertices) + index)
                for i in range(1, len(face) - 1):
                    triangles.append([face[0], face[i], face[i + 1]])
    if not vertices or not triangles:
        raise ValueError(f"{path} does not contain usable mesh data")
    return np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int64)


def face_geometry(vertices, triangles):
    a = vertices[triangles[:, 0]]
    b = vertices[triangles[:, 1]]
    c = vertices[triangles[:, 2]]
    normals = np.cross(b - a, c - a)
    lengths = np.linalg.norm(normals, axis=1)
    centers = (a + b + c) / 3.0
    return centers, normals / (lengths[:, None] + 1e-12), lengths * 0.5


def estimate_hemisphere(vertices):
    z_min = float(vertices[:, 2].min())
    z_max = float(vertices[:, 2].max())
    z_tol = max((z_max - z_min) * 1e-6, 1e-5)

    top = vertices[np.abs(vertices[:, 2] - z_max) <= z_tol]
    center_xy = top[:, :2].mean(axis=0) if len(top) else np.zeros(2)

    radial = np.linalg.norm(vertices[:, :2] - center_xy, axis=1)
    bottom = vertices[np.abs(vertices[:, 2] - z_min) <= z_tol]
    bottom_radial = np.linalg.norm(bottom[:, :2] - center_xy, axis=1)
    radius_modes = Counter(np.round(bottom_radial, 3))
    if not radius_modes:
        raise ValueError("Could not infer hemisphere radius from the source mesh")

    # The cut-out radius is the smallest strong radius mode at the bottom.
    max_count = max(radius_modes.values())
    strong_modes = sorted(radius for radius, count in radius_modes.items() if count >= max(8, max_count * 0.2))
    radius_guess = float(strong_modes[0])
    radius_tol = max(0.05, radius_guess * 0.01)

    # The mesh contains a straight neck below the hemisphere. Pick the highest
    # full ring at this radius as the hemisphere equator.
    ring_mask = np.abs(radial - radius_guess) <= radius_tol
    z_modes = Counter(np.round(vertices[ring_mask, 2], 3))
    strong_z = [z for z, count in z_modes.items() if count >= max(8, max(z_modes.values()) * 0.2)]
    equator_z = float(max(strong_z))

    fit_mask = (vertices[:, 2] >= equator_z - radius_tol) & (radial <= radius_guess + radius_tol)
    fit_points = vertices[fit_mask]
    a = np.c_[fit_points[:, 0], fit_points[:, 1], fit_points[:, 2], np.ones(len(fit_points))]
    b = -(fit_points[:, 0] ** 2 + fit_points[:, 1] ** 2 + fit_points[:, 2] ** 2)
    sx, sy, sz, d = np.linalg.lstsq(a, b, rcond=None)[0]
    center = np.array([-sx / 2.0, -sy / 2.0, -sz / 2.0])
    radius = float(np.sqrt(max(0.0, np.dot(center, center) - d)))
    residual = np.linalg.norm(fit_points - center, axis=1) - radius

    return {
        "center": center,
        "radius": radius,
        "equator_z": equator_z,
        "radius_guess": radius_guess,
        "radius_tolerance": radius_tol,
        "fit_points": int(len(fit_points)),
        "fit_residual_max_abs": float(np.abs(residual).max()),
        "fit_residual_std": float(residual.std()),
        "bottom_radius_modes": [
            {"radius": float(radius), "count": int(count)}
            for radius, count in sorted(radius_modes.items(), key=lambda item: (item[0], -item[1]))[:12]
        ],
    }


def select_spherical_surface(vertices, triangles, hemi):
    centers, normals, areas = face_geometry(vertices, triangles)
    center = hemi["center"]
    radius = hemi["radius"]
    dist = np.linalg.norm(centers - center, axis=1)
    from_center = centers - center
    normal_dot = (normals * from_center).sum(axis=1) / (dist + 1e-12)

    surface_tol = max(0.05, radius * 0.004)
    mask = (
        (centers[:, 2] >= hemi["equator_z"] - 1e-5)
        & (np.abs(dist - radius) <= surface_tol)
        & (normal_dot < -0.5)
    )
    selected = np.flatnonzero(mask)
    if len(selected) == 0:
        raise ValueError("Could not select the source hemisphere surface")

    return selected, {
        "surface_faces": int(len(selected)),
        "surface_area": float(areas[selected].sum()),
        "source_normal_dot_min": float(normal_dot[selected].min()),
        "source_normal_dot_max": float(normal_dot[selected].max()),
        "surface_z_min": float(centers[selected, 2].min()),
        "surface_z_max": float(centers[selected, 2].max()),
    }


def boundary_edges(triangles):
    counts = Counter()
    for tri in triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge = (int(a), int(b)) if a < b else (int(b), int(a))
            counts[edge] += 1
    return [edge for edge, count in counts.items() if count == 1]


def signed_volume(vertices, faces):
    volume = 0.0
    for face in faces:
        a = vertices[face[0]]
        for i in range(1, len(face) - 1):
            b = vertices[face[i]]
            c = vertices[face[i + 1]]
            volume += float(np.dot(a, np.cross(b, c)) / 6.0)
    return volume


def edge_stats(faces):
    counts = Counter()
    for face in faces:
        for a, b in zip(face, face[1:] + face[:1]):
            edge = (int(a), int(b)) if a < b else (int(b), int(a))
            counts[edge] += 1
    return {
        "edges": int(len(counts)),
        "boundary_edges": int(sum(1 for count in counts.values() if count == 1)),
        "nonmanifold_edges": int(sum(1 for count in counts.values() if count > 2)),
    }


def build_closed_hemisphere(vertices, triangles, selected, hemi):
    surface_triangles = triangles[selected]
    boundary = boundary_edges(surface_triangles)
    boundary_ids = sorted({vertex_id for edge in boundary for vertex_id in edge})
    boundary_points = vertices[boundary_ids]

    z_span = float(boundary_points[:, 2].max() - boundary_points[:, 2].min())
    if z_span > 1e-3:
        raise ValueError(f"Hemisphere boundary is not planar enough to cap, z span={z_span}")

    used = sorted(set(int(vertex_id) for tri in surface_triangles for vertex_id in tri))
    remap = {old: new for new, old in enumerate(used)}
    out_vertices = [vertices[old] for old in used]

    # Source normals point into the removed hemisphere. Reverse them so the
    # recovered solid has outward normals.
    out_faces = [[remap[int(c)], remap[int(b)], remap[int(a)]] for a, b, c in surface_triangles]

    center_id = len(out_vertices)
    cap_center = np.array([hemi["center"][0], hemi["center"][1], hemi["equator_z"]], dtype=np.float64)
    out_vertices.append(cap_center)

    rel = boundary_points[:, :2] - hemi["center"][:2]
    order = np.argsort(np.arctan2(rel[:, 1], rel[:, 0]))
    ring = [remap[boundary_ids[i]] for i in order]
    for i, current in enumerate(ring):
        nxt = ring[(i + 1) % len(ring)]
        out_faces.append([center_id, nxt, current])

    out_vertices = np.asarray(out_vertices, dtype=np.float64)
    volume = signed_volume(out_vertices, out_faces)
    if volume < 0:
        out_faces = [list(reversed(face)) for face in out_faces]
        volume = -volume

    return out_vertices, out_faces, {
        "open_boundary_edges_before_cap": int(len(boundary)),
        "open_boundary_vertices_before_cap": int(len(boundary_ids)),
        "cap_triangles": int(len(ring)),
        "signed_volume": float(volume),
    }


def write_obj(path, vertices, faces, stats):
    output_path = Path(path)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Recovered hemispherical cut-out from data/source/ref.obj\n")
        handle.write(f"# sphere_center {' '.join(f'{v:.6f}' for v in stats['hemisphere']['center'])}\n")
        handle.write(f"# sphere_radius {stats['hemisphere']['radius']:.6f}\n")
        handle.write("o recovered_hemisphere_cutout\n")
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        handle.write("g hemisphere_surface\n")
        for face in faces:
            handle.write("f " + " ".join(str(index + 1) for index in face) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Recover the hemispherical cut-out from data/source/ref.obj.")
    parser.add_argument("input", nargs="?", default="data/source/ref.obj")
    parser.add_argument("-o", "--output", default="data/reference/ref_cutout_part.obj")
    parser.add_argument("--stats", default="data/reference/ref_cutout_part_stats.json")
    args = parser.parse_args()

    vertices, triangles = load_obj(args.input)
    hemi = estimate_hemisphere(vertices)
    selected, selection = select_spherical_surface(vertices, triangles, hemi)
    out_vertices, out_faces, mesh = build_closed_hemisphere(vertices, triangles, selected, hemi)

    theoretical_volume = 2.0 * np.pi * hemi["radius"] ** 3 / 3.0
    theoretical_area = 2.0 * np.pi * hemi["radius"] ** 2
    stats = {
        "source": str(Path(args.input)),
        "output": str(Path(args.output)),
        "hemisphere": {
            "center": hemi["center"].tolist(),
            "radius": hemi["radius"],
            "equator_z": hemi["equator_z"],
            "fit_points": hemi["fit_points"],
            "fit_residual_max_abs": hemi["fit_residual_max_abs"],
            "fit_residual_std": hemi["fit_residual_std"],
            "bottom_radius_modes": hemi["bottom_radius_modes"],
            "theoretical_hemisphere_volume": float(theoretical_volume),
            "theoretical_curved_surface_area": float(theoretical_area),
        },
        "selection": selection,
        "mesh": {
            **mesh,
            **edge_stats(out_faces),
            "vertices": int(len(out_vertices)),
            "faces": int(len(out_faces)),
            "bbox_min": out_vertices.min(axis=0).tolist(),
            "bbox_max": out_vertices.max(axis=0).tolist(),
        },
    }

    write_obj(args.output, out_vertices, out_faces, stats)
    stats_path = Path(args.stats)
    if stats_path.parent != Path("."):
        stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
