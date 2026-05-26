#!/usr/bin/env python3
"""
Convert exported SMPL pose npz files into LiDAR-like point clouds.

The input convention matches smpl_random_pc_viz.py:
  root_pos, root_rot_xyzw, dof_pos, smpl_betas

Default usage from this directory:
  python smpl_lidar_scan.py

Process one file:
  python smpl_lidar_scan.py smpl_ep000001_h025_000000.npz -o lidar_pc

The output is an npz point-cloud file. Original SMPL fields are copied through,
and the LiDAR result is stored in points/xyz/lidar_points with per-point range,
ring, azimuth, and intensity arrays.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


_THIS_DIR = Path(__file__).resolve().parent


def _candidate_roots(cli_root: str | None) -> list[Path]:
    candidates: list[Path] = []
    if cli_root:
        candidates.append(Path(cli_root).expanduser().resolve())
    if os.environ.get("PULSE_ROOT"):
        candidates.append(Path(os.environ["PULSE_ROOT"]).expanduser().resolve())

    # Preserve the assumption used by smpl_random_pc_viz.py, then try parents.
    candidates.append((_THIS_DIR / "human2humanoid").resolve())
    candidates.append((_THIS_DIR / ".." / "..").resolve())
    candidates.extend(parent.resolve() for parent in [_THIS_DIR, *_THIS_DIR.parents])

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _setup_import_path(cli_root: str | None) -> Path | None:
    for root in _candidate_roots(cli_root):
        if (root / "phc" / "phc").exists():
            sys.path.insert(0, str(root / "phc"))
            return root
        if (root / "phc").exists():
            sys.path.insert(0, str(root))
            return root
    return None


def _default_smpl_path(pulse_root: Path | None, cli_path: str | None) -> str:
    if cli_path:
        return str(Path(cli_path).expanduser().resolve())
    if os.environ.get("SMPL_MODEL_PATH"):
        return os.environ["SMPL_MODEL_PATH"]
    if pulse_root is not None:
        return str((pulse_root / "data" / "smpl").resolve())
    return str((_THIS_DIR / ".." / ".." / "data" / "smpl").resolve())


def _as_pose_tensor(data: np.lib.npyio.NpzFile, key: str, width: int | None = None) -> np.ndarray:
    if key not in data:
        raise KeyError(f"missing required field {key!r}")

    arr = np.asarray(data[key], dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim == 0:
        arr = arr.reshape(1, 1)

    if width is not None:
        if arr.shape[-1] < width:
            raise ValueError(f"{key!r} has width {arr.shape[-1]}, expected at least {width}")
        arr = arr[:, :width]
    return arr


def _discover_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(path)
    files = sorted(path.glob("*.npz"))
    return [
        f
        for f in files
        if not f.name.endswith("_lidar.npz")
        and not f.name.endswith("_lidar_pc.npz")
        and not f.name.startswith(".")
    ]


def _hdl64_vertical_angles_deg() -> np.ndarray:
    """Approximate Velodyne HDL-64E vertical firing angles.

    The exact calibration differs by unit. These 64 channels span roughly
    +2 to -24.9 degrees, which is enough for a realistic sparse human scan.
    """

    upper = np.linspace(2.0, -8.83, 32, dtype=np.float32)
    lower = np.linspace(-8.87, -24.9, 32, dtype=np.float32)
    return np.concatenate([upper, lower])


def _uniform_vertical_angles_deg(num_lines: int, fov_up: float, fov_down: float) -> np.ndarray:
    return np.linspace(float(fov_up), float(fov_down), int(num_lines), dtype=np.float32)


def _points_to_sensor_frame(points: np.ndarray, distance: float, sensor_height: float) -> np.ndarray:
    sensor_origin = np.array([-float(distance), 0.0, float(sensor_height)], dtype=np.float32)
    return points.astype(np.float32, copy=False) - sensor_origin


def _angle_diff(a: np.ndarray, b: float) -> np.ndarray:
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def _ray_directions(azimuth_rad: np.ndarray, elevation_rad: np.ndarray) -> np.ndarray:
    az_grid, elev_grid = np.meshgrid(azimuth_rad, elevation_rad)
    cos_elev = np.cos(elev_grid)
    dirs = np.stack(
        [
            cos_elev * np.cos(az_grid),
            cos_elev * np.sin(az_grid),
            np.sin(elev_grid),
        ],
        axis=-1,
    )
    return dirs.reshape(-1, 3).astype(np.float32)


def _candidate_beams_for_mesh(
    vertices_sensor: np.ndarray,
    vertical_angles_rad: np.ndarray,
    azimuth_rad: np.ndarray,
    *,
    padding_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Select beams that can hit the mesh bounding sphere.

    A 128x1800 full scan has 230k rays. The human occupies only a few degrees
    at typical distances, so this conservative angular crop keeps exact
    ray-triangle intersection practical while preserving true ray geometry.
    """

    center = vertices_sensor.mean(axis=0)
    radius = float(np.linalg.norm(vertices_sensor - center, axis=1).max())
    center_range = float(np.linalg.norm(center))
    center_horizontal = float(np.hypot(center[0], center[1]))
    padding = math.radians(float(padding_deg))

    if center_range <= 1e-6 or center[0] <= 0.0:
        return np.arange(vertical_angles_rad.size), np.arange(azimuth_rad.size)

    center_az = float(math.atan2(center[1], center[0]))
    center_elev = float(math.atan2(center[2], center_horizontal))
    az_span = math.asin(min(1.0, radius / max(center_horizontal, 1e-6))) + padding
    elev_span = math.asin(min(1.0, radius / max(center_range, 1e-6))) + padding

    az_idx = np.nonzero(np.abs(_angle_diff(azimuth_rad, center_az)) <= az_span)[0]
    ring_idx = np.nonzero(np.abs(vertical_angles_rad - center_elev) <= elev_span)[0]

    if az_idx.size == 0:
        az_idx = np.arange(azimuth_rad.size)
    if ring_idx.size == 0:
        ring_idx = np.arange(vertical_angles_rad.size)
    return ring_idx.astype(np.int32), az_idx.astype(np.int32)


def _ray_mesh_intersections(
    ray_dirs: np.ndarray,
    vertices_sensor: np.ndarray,
    faces: np.ndarray,
    *,
    min_range: float,
    max_range: float,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest hit distance and face index for each ray.

    Uses the Moller-Trumbore triangle intersection test. Rays originate at the
    LiDAR origin in sensor coordinates; ray_dirs must be unit vectors.
    """

    tri = vertices_sensor[faces.astype(np.int64)]
    v0 = tri[:, 0].astype(np.float32, copy=False)
    edge1 = (tri[:, 1] - tri[:, 0]).astype(np.float32, copy=False)
    edge2 = (tri[:, 2] - tri[:, 0]).astype(np.float32, copy=False)
    tvec = (-v0).astype(np.float32, copy=False)
    qvec = np.cross(tvec, edge1).astype(np.float32, copy=False)
    edge2_dot_qvec = np.einsum("fj,fj->f", edge2, qvec).astype(np.float32, copy=False)

    hit_t = np.full(ray_dirs.shape[0], np.inf, dtype=np.float32)
    hit_face = np.full(ray_dirs.shape[0], -1, dtype=np.int32)
    eps = np.float32(1e-8)

    for start in range(0, ray_dirs.shape[0], int(chunk_size)):
        dirs = ray_dirs[start : start + int(chunk_size)].astype(np.float32, copy=False)
        pvec = np.cross(dirs[:, None, :], edge2[None, :, :]).astype(np.float32, copy=False)
        det = np.einsum("fj,rfj->rf", edge1, pvec)
        valid = np.abs(det) > eps
        inv_det = np.zeros_like(det, dtype=np.float32)
        inv_det[valid] = 1.0 / det[valid]

        u = np.einsum("fj,rfj->rf", tvec, pvec) * inv_det
        valid &= (u >= 0.0) & (u <= 1.0)

        v = np.einsum("rj,fj->rf", dirs, qvec) * inv_det
        valid &= (v >= 0.0) & ((u + v) <= 1.0)

        t = edge2_dot_qvec[None, :] * inv_det
        valid &= (t >= min_range) & (t <= max_range)
        t = np.where(valid, t, np.inf)

        local_face = np.argmin(t, axis=1)
        local_t = t[np.arange(t.shape[0]), local_face]
        hit = np.isfinite(local_t)
        hit_t[start : start + dirs.shape[0]][hit] = local_t[hit]
        hit_face[start : start + dirs.shape[0]][hit] = local_face[hit].astype(np.int32)

    return hit_t, hit_face


def simulate_lidar_scan(
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    *,
    distance: float = 10.0,
    sensor_height: float = 1.3,
    vertical_angles_deg: np.ndarray | None = None,
    horizontal_res_deg: float = 0.2,
    min_range: float = 0.2,
    max_range: float = 80.0,
    range_noise_std: float = 0.0,
    dropout: float = 0.0,
    scan_padding_deg: float = 1.0,
    ray_chunk_size: int = 256,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Simulate a multi-line LiDAR by exact ray-triangle mesh intersection."""

    rng = np.random.default_rng(seed)
    if vertical_angles_deg is None:
        vertical_angles_deg = _uniform_vertical_angles_deg(128, 2.0, -24.9)

    vertical_angles_deg = np.asarray(vertical_angles_deg, dtype=np.float32).reshape(-1)
    vertical_angles_rad = np.deg2rad(vertical_angles_deg)
    num_rings = vertical_angles_deg.size
    num_cols = int(round(360.0 / float(horizontal_res_deg)))
    if num_rings <= 0:
        raise ValueError("at least one LiDAR line is required")
    if num_cols <= 0:
        raise ValueError("horizontal_res_deg must be positive")

    vertices_sensor = _points_to_sensor_frame(mesh_vertices, distance=distance, sensor_height=sensor_height)
    if vertices_sensor.size == 0 or mesh_faces.size == 0:
        return _empty_lidar_result(vertical_angles_deg)

    all_azimuth_rad = -math.pi + (np.arange(num_cols, dtype=np.float32) + 0.5) * math.radians(horizontal_res_deg)
    ring_idx, col_idx = _candidate_beams_for_mesh(
        vertices_sensor,
        vertical_angles_rad,
        all_azimuth_rad,
        padding_deg=scan_padding_deg,
    )
    if ring_idx.size == 0 or col_idx.size == 0:
        return _empty_lidar_result(vertical_angles_deg)

    ray_dirs = _ray_directions(all_azimuth_rad[col_idx], vertical_angles_rad[ring_idx])
    ring_grid, col_grid = np.meshgrid(ring_idx, col_idx, indexing="ij")
    ray_rings = ring_grid.reshape(-1).astype(np.int32)
    ray_cols = col_grid.reshape(-1).astype(np.int32)

    ranges, face_idx = _ray_mesh_intersections(
        ray_dirs,
        vertices_sensor,
        mesh_faces,
        min_range=min_range,
        max_range=max_range,
        chunk_size=ray_chunk_size,
    )
    hit = np.isfinite(ranges)
    if not np.any(hit):
        return _empty_lidar_result(vertical_angles_deg)

    ray_dirs = ray_dirs[hit]
    ranges = ranges[hit]
    nearest_ring = ray_rings[hit]
    col = ray_cols[hit]
    face_idx = face_idx[hit]

    if dropout > 0.0:
        keep = rng.random(ray_dirs.shape[0]) >= float(dropout)
        ray_dirs = ray_dirs[keep]
        ranges = ranges[keep]
        nearest_ring = nearest_ring[keep]
        col = col[keep]
        face_idx = face_idx[keep]

    if range_noise_std > 0.0 and ranges.size:
        noisy_ranges = np.maximum(
            min_range,
            ranges + rng.normal(0.0, float(range_noise_std), size=ranges.shape).astype(np.float32),
        )
        ranges = noisy_ranges

    pts = (ray_dirs * ranges[:, None]).astype(np.float32, copy=False)

    azimuth_deg = -180.0 + (col.astype(np.float32) + 0.5) * float(horizontal_res_deg)
    intensity = _range_intensity(ranges)
    sort_order = np.lexsort((azimuth_deg, nearest_ring))

    return {
        "points": pts[sort_order].astype(np.float32, copy=False),
        "range": ranges[sort_order].astype(np.float32, copy=False),
        "ring": nearest_ring[sort_order].astype(np.int32, copy=False),
        "azimuth": azimuth_deg[sort_order].astype(np.float32, copy=False),
        "intensity": intensity[sort_order].astype(np.float32, copy=False),
        "vertical_angles_deg": vertical_angles_deg.astype(np.float32, copy=False),
        "face_index": face_idx[sort_order].astype(np.int32, copy=False),
    }


def _range_intensity(ranges: np.ndarray) -> np.ndarray:
    # A simple distance falloff, normalized for common visualization tools.
    return np.clip(1.0 / np.maximum(ranges, 1e-6) ** 2 * 100.0, 0.0, 1.0)


def _empty_lidar_result(vertical_angles_deg: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "points": np.empty((0, 3), dtype=np.float32),
        "range": np.empty((0,), dtype=np.float32),
        "ring": np.empty((0,), dtype=np.int32),
        "azimuth": np.empty((0,), dtype=np.float32),
        "intensity": np.empty((0,), dtype=np.float32),
        "vertical_angles_deg": vertical_angles_deg.astype(np.float32, copy=False),
        "face_index": np.empty((0,), dtype=np.int32),
    }


def _write_ascii_ply(path: Path, points: np.ndarray, intensity: np.ndarray, ring: np.ndarray) -> None:
    colors = np.clip(intensity.reshape(-1, 1) * np.array([[255, 180, 80]], dtype=np.float32), 0, 255)
    colors = colors.astype(np.uint8)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property int ring\n")
        f.write("end_header\n")
        for p, c, r in zip(points, colors, ring):
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])} {int(r)}\n"
            )


def _convert_one(
    input_path: Path,
    output_dir: Path,
    smpl_to_pc,
    torch_module,
    args: argparse.Namespace,
    vertical_angles_deg: np.ndarray,
) -> Path:
    data = np.load(input_path, allow_pickle=False)
    root_pos = _as_pose_tensor(data, "root_pos", 3)
    root_rot = _as_pose_tensor(data, "root_rot_xyzw", 4)
    dof_pos = _as_pose_tensor(data, "dof_pos", 69)

    beta_key = "smpl_betas" if "smpl_betas" in data else "betas"
    betas = _as_pose_tensor(data, beta_key, args.num_betas)

    device = torch_module.device(args.device)
    root_pos_t = torch_module.from_numpy(root_pos).to(device)
    root_rot_t = torch_module.from_numpy(root_rot).to(device)
    dof_pos_t = torch_module.from_numpy(dof_pos).to(device)
    betas_t = torch_module.from_numpy(betas).to(device)

    with torch_module.no_grad():
        vertices = smpl_to_pc.vertices_from_pose(
            root_pos=root_pos_t,
            root_rot_xyzw=root_rot_t,
            dof_pos=dof_pos_t,
            betas=betas_t,
        )
    mesh_vertices = vertices[0].detach().cpu().numpy().astype(np.float32, copy=False)
    mesh_faces = smpl_to_pc.faces().detach().cpu().numpy().astype(np.int32, copy=False)

    lidar = simulate_lidar_scan(
        mesh_vertices,
        mesh_faces,
        distance=args.distance,
        sensor_height=args.sensor_height,
        vertical_angles_deg=vertical_angles_deg,
        horizontal_res_deg=args.horizontal_res_deg,
        min_range=args.min_range,
        max_range=args.max_range,
        range_noise_std=args.range_noise_std,
        dropout=args.dropout,
        scan_padding_deg=args.scan_padding_deg,
        ray_chunk_size=args.ray_chunk_size,
        seed=args.seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem + "_lidar"
    out_npz = output_dir / f"{stem}.npz"

    payload = {k: data[k] for k in data.files}
    payload.update(
        {
            "points": lidar["points"],
            "xyz": lidar["points"],
            "lidar_points": lidar["points"],
            "lidar_range": lidar["range"],
            "lidar_ring": lidar["ring"],
            "lidar_azimuth_deg": lidar["azimuth"],
            "lidar_intensity": lidar["intensity"],
            "lidar_face_index": lidar["face_index"],
            "lidar_vertical_angles_deg": lidar["vertical_angles_deg"],
            "lidar_distance_m": np.asarray(args.distance, dtype=np.float32),
            "lidar_sensor_height_m": np.asarray(args.sensor_height, dtype=np.float32),
            "lidar_horizontal_res_deg": np.asarray(args.horizontal_res_deg, dtype=np.float32),
            "lidar_num_lines": np.asarray(lidar["vertical_angles_deg"].shape[0], dtype=np.int32),
            "lidar_method": np.asarray("ray_mesh_intersection"),
            "source_file": np.asarray(input_path.name),
        }
    )
    np.savez_compressed(out_npz, **payload)

    if args.output_format in {"ply", "both"}:
        _write_ascii_ply(
            output_dir / f"{stem}.ply",
            lidar["points"],
            lidar["intensity"],
            lidar["ring"],
        )

    return out_npz


def _build_smpl_converter(args: argparse.Namespace):
    pulse_root = _setup_import_path(args.pulse_root)
    try:
        import torch
        from phc.utils.pc_anomaly import SmplToPointCloud
    except ImportError as exc:
        message = (
            "Failed to import torch/phc. Run this script in the same environment as "
            "smpl_random_pc_viz.py, or pass --pulse-root/--smpl-model-path explicitly."
        )
        raise SystemExit(f"{message}\nOriginal error: {exc}") from exc

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device(args.device)
    smpl_model_path = _default_smpl_path(pulse_root, args.smpl_model_path)
    smpl_to_pc = SmplToPointCloud(
        smpl_model_path=smpl_model_path,
        num_betas=args.num_betas,
        num_points=args.surface_points,
        local_coord=not args.world_coord,
        device=device,
    )
    return smpl_to_pc, torch


def _vertical_angles_from_args(args: argparse.Namespace) -> np.ndarray:
    if args.vertical_angles_file:
        arr = np.loadtxt(args.vertical_angles_file, dtype=np.float32)
        return np.asarray(arr, dtype=np.float32).reshape(-1)
    if args.vertical_model == "uniform":
        return _uniform_vertical_angles_deg(args.lines, args.fov_up_deg, args.fov_down_deg)
    if args.lines != 64:
        return _uniform_vertical_angles_deg(args.lines, args.fov_up_deg, args.fov_down_deg)
    return _hdl64_vertical_angles_deg()


def _positive_float(value: str) -> float:
    result = float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SMPL npz pose files into ray-mesh simulated LiDAR point clouds."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=str(_THIS_DIR),
        help="Input .npz file or directory. Defaults to this script directory.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(_THIS_DIR / "lidar_pc"),
        help="Directory for converted point-cloud files.",
    )
    parser.add_argument("--pulse-root", default=None, help="Project root containing phc/ and data/smpl.")
    parser.add_argument("--smpl-model-path", default=None, help="Path to SMPL model directory.")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda:0.")
    parser.add_argument("--num-betas", type=int, default=10, help="Number of SMPL beta coefficients to use.")
    parser.add_argument(
        "--surface-points",
        type=int,
        default=1024,
        help="Legacy compatibility option; ray-mesh mode does not use surface sampling.",
    )
    parser.add_argument("--world-coord", action="store_true", help="Ask SmplToPointCloud for world coordinates.")

    parser.add_argument("--distance", type=_positive_float, default=10.0, help="LiDAR-to-human distance in meters.")
    parser.add_argument("--sensor-height", type=float, default=1.3, help="LiDAR sensor height in SMPL coordinates.")
    parser.add_argument("--lines", type=int, default=128, help="Number of vertical LiDAR lines.")
    parser.add_argument(
        "--vertical-model",
        choices=("hdl64", "uniform"),
        default="uniform",
        help="Vertical angle layout. hdl64 is available only when --lines 64.",
    )
    parser.add_argument("--fov-up-deg", type=float, default=2.0, help="Uniform model upper vertical FOV.")
    parser.add_argument("--fov-down-deg", type=float, default=-24.9, help="Uniform model lower vertical FOV.")
    parser.add_argument("--vertical-angles-file", default=None, help="Text file with one vertical angle per line.")
    parser.add_argument(
        "--horizontal-res-deg",
        type=_positive_float,
        default=0.2,
        help="Horizontal angular resolution in degrees.",
    )
    parser.add_argument("--min-range", type=float, default=0.2, help="Minimum valid LiDAR range.")
    parser.add_argument("--max-range", type=float, default=80.0, help="Maximum valid LiDAR range.")
    parser.add_argument("--range-noise-std", type=float, default=0.0, help="Gaussian range noise std in meters.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Random point dropout probability.")
    parser.add_argument(
        "--scan-padding-deg",
        type=float,
        default=1.0,
        help="Extra angular padding around the mesh bounding sphere before exact ray-mesh intersection.",
    )
    parser.add_argument(
        "--ray-chunk-size",
        type=int,
        default=256,
        help="Number of rays processed per vectorized intersection chunk.",
    )
    parser.add_argument(
        "--no-snap-to-beams",
        action="store_true",
        help="Deprecated compatibility option; ray-mesh hits already lie on beam centers.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for noise/dropout.")
    parser.add_argument(
        "--output-format",
        choices=("npz", "ply", "both"),
        default="npz",
        help="Write npz only, ply only in addition to npz metadata, or both.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N input files.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not (0.0 <= args.dropout < 1.0):
        raise SystemExit("--dropout must be in [0, 1)")
    if args.max_range <= args.min_range:
        raise SystemExit("--max-range must be larger than --min-range")
    if args.lines <= 0:
        raise SystemExit("--lines must be positive")
    if args.ray_chunk_size <= 0:
        raise SystemExit("--ray-chunk-size must be positive")

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    input_files = _discover_inputs(input_path)
    if args.limit is not None:
        input_files = input_files[: args.limit]
    if not input_files:
        raise SystemExit(f"No .npz files found in {input_path}")

    vertical_angles_deg = _vertical_angles_from_args(args)
    smpl_to_pc, torch_module = _build_smpl_converter(args)

    print(f"Found {len(input_files)} input file(s). Writing to {output_dir}")
    for i, path in enumerate(input_files, 1):
        out_path = _convert_one(path, output_dir, smpl_to_pc, torch_module, args, vertical_angles_deg)
        points = np.load(out_path, allow_pickle=False)["lidar_points"]
        print(f"[{i:04d}/{len(input_files):04d}] {path.name} -> {out_path.name} ({points.shape[0]} points)")


if __name__ == "__main__":
    main()
