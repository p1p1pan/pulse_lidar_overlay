"""
多线 LiDAR 风格点云：SMPL 三角网格 + 射线–面相交（与 scripts/smpl_lidar_raymesh_internal/smpl_lidar_scan.py 同源逻辑）。
仅依赖 numpy / math，供训练侧可选落盘，不参与可微分支。
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np


def _hdl64_vertical_angles_deg() -> np.ndarray:
    upper = np.linspace(2.0, -8.83, 32, dtype=np.float32)
    lower = np.linspace(-8.87, -24.9, 32, dtype=np.float32)
    return np.concatenate([upper, lower])


def _uniform_vertical_angles_deg(num_lines: int, fov_up: float, fov_down: float) -> np.ndarray:
    return np.linspace(float(fov_up), float(fov_down), int(num_lines), dtype=np.float32)


def vertical_angles_deg_for_lidar(num_lines: int, fov_up: float, fov_down: float) -> np.ndarray:
    if int(num_lines) == 64:
        return _hdl64_vertical_angles_deg()
    return _uniform_vertical_angles_deg(num_lines, fov_up, fov_down)


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
) -> Tuple[np.ndarray, np.ndarray]:
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
) -> Tuple[np.ndarray, np.ndarray]:
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


def _range_intensity(ranges: np.ndarray) -> np.ndarray:
    return np.clip(1.0 / np.maximum(ranges, 1e-6) ** 2 * 100.0, 0.0, 1.0)


def _empty_lidar_result(vertical_angles_deg: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "points": np.empty((0, 3), dtype=np.float32),
        "range": np.empty((0,), dtype=np.float32),
        "ring": np.empty((0,), dtype=np.int32),
        "azimuth": np.empty((0,), dtype=np.float32),
        "intensity": np.empty((0,), dtype=np.float32),
        "vertical_angles_deg": vertical_angles_deg.astype(np.float32, copy=False),
        "face_index": np.empty((0,), dtype=np.int32),
    }


def simulate_lidar_scan(
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    *,
    distance: float = 10.0,
    sensor_height: float = 1.3,
    vertical_angles_deg: Optional[np.ndarray] = None,
    horizontal_res_deg: float = 0.2,
    min_range: float = 0.2,
    max_range: float = 80.0,
    range_noise_std: float = 0.0,
    dropout: float = 0.0,
    scan_padding_deg: float = 1.0,
    ray_chunk_size: int = 256,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
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


__all__ = ["simulate_lidar_scan", "vertical_angles_deg_for_lidar"]
