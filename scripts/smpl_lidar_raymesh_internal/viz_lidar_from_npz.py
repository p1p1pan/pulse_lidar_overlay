#!/usr/bin/env python3
"""
从本包内 pc_smpl_dataset 随机选一个 SMPL npz，在内存中做射线网格 LiDAR 仿真，
仅 Open3D 弹窗，不写文件。逻辑与 smpl_lidar_scan.py 一致。

用法（在 smpl_lidar_raymesh_internal 目录下）:
  python viz_lidar_from_npz.py
"""
from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
_DATASET_DIR = _THIS / "pc_smpl_dataset"


def _load_sls():
    path = _THIS / "smpl_lidar_scan.py"
    spec = importlib.util.spec_from_file_location("smpl_lidar_scan_viz_only", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pick_random_npz(sls) -> Path:
    if not _DATASET_DIR.is_dir():
        raise FileNotFoundError(
            f"缺少数据集目录: {_DATASET_DIR}\n"
            "请创建该目录并放入与 smpl_lidar_scan 相同字段的 smpl_*.npz（不要只恢复脚本而不放数据）。"
        )
    files = sls._discover_inputs(_DATASET_DIR)
    if not files:
        raise FileNotFoundError(
            f"{_DATASET_DIR} 下没有可用 SMPL npz（已排除 *_lidar.npz）。\n"
            "请把训练导出的 smpl_*.npz 复制到该目录后再运行。"
        )
    return random.choice(files)


def _lidar_from_npz(npz_path: Path, sls, conv_args) -> tuple[np.ndarray, np.ndarray, str]:
    with np.load(npz_path, allow_pickle=False) as data:
        return _lidar_from_npz_loaded(data, npz_path.name, sls, conv_args)


def _lidar_from_npz_loaded(data, stem: str, sls, conv_args) -> tuple[np.ndarray, np.ndarray, str]:
    root_pos = sls._as_pose_tensor(data, "root_pos", 3)
    root_rot = sls._as_pose_tensor(data, "root_rot_xyzw", 4)
    dof_pos = sls._as_pose_tensor(data, "dof_pos", 69)
    beta_key = "smpl_betas" if "smpl_betas" in data else "betas"
    betas = sls._as_pose_tensor(data, beta_key, conv_args.num_betas)

    import torch

    device = torch.device(conv_args.device)
    root_pos_t = torch.from_numpy(root_pos).to(device)
    root_rot_t = torch.from_numpy(root_rot).to(device)
    dof_pos_t = torch.from_numpy(dof_pos).to(device)
    betas_t = torch.from_numpy(betas).to(device)

    smpl_to_pc, torch_module = sls._build_smpl_converter(conv_args)
    vertical_angles_deg = sls._vertical_angles_from_args(conv_args)

    with torch_module.no_grad():
        vertices = smpl_to_pc.vertices_from_pose(
            root_pos=root_pos_t,
            root_rot_xyzw=root_rot_t,
            dof_pos=dof_pos_t,
            betas=betas_t,
        )
    mesh_vertices = vertices[0].detach().cpu().numpy().astype(np.float32, copy=False)
    mesh_faces = smpl_to_pc.faces().detach().cpu().numpy().astype(np.int32, copy=False)

    lidar = sls.simulate_lidar_scan(
        mesh_vertices,
        mesh_faces,
        distance=conv_args.distance,
        sensor_height=conv_args.sensor_height,
        vertical_angles_deg=vertical_angles_deg,
        horizontal_res_deg=conv_args.horizontal_res_deg,
        min_range=conv_args.min_range,
        max_range=conv_args.max_range,
        range_noise_std=conv_args.range_noise_std,
        dropout=conv_args.dropout,
        scan_padding_deg=conv_args.scan_padding_deg,
        ray_chunk_size=conv_args.ray_chunk_size,
        seed=conv_args.seed,
    )
    pts = np.asarray(lidar["points"], dtype=np.float64)
    intensity = np.asarray(lidar["intensity"], dtype=np.float64)
    return pts, intensity, stem


def main() -> None:
    sls = _load_sls()
    npz_path = _pick_random_npz(sls)

    # parse_args(args) 会解析整个列表，不含 sys.argv[0]；不要伪造脚本名，否则第一个串会占掉 positional input
    noop_dir = _THIS / ".viz_lidar_noop_out"
    argv = [str(npz_path), "-o", str(noop_dir)]
    conv_args = sls.parse_args(argv)
    if not (0.0 <= conv_args.dropout < 1.0):
        raise SystemExit("--dropout must be in [0, 1)")
    if conv_args.max_range <= conv_args.min_range:
        raise SystemExit("--max-range must be larger than --min-range")

    print(f"[viz] 随机选中: {npz_path}")
    pts, intensity, name = _lidar_from_npz(npz_path, sls, conv_args)
    print(f"[viz] LiDAR 点数: {pts.shape[0]}")

    try:
        import open3d as o3d
    except ImportError:
        print("请安装: pip install open3d")
        sys.exit(1)

    if pts.shape[0] == 0:
        print("[viz] 无命中点，无法绘制。")
        sys.exit(1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    c = np.clip(
        np.stack([intensity, 0.4 + 0.45 * intensity, 0.15 + 0.25 * intensity], axis=1),
        0.0,
        1.0,
    )
    pcd.colors = o3d.utility.Vector3dVector(c)
    o3d.visualization.draw_geometries([pcd], window_name=f"LiDAR (no save) | {name}")


if __name__ == "__main__":
    main()
