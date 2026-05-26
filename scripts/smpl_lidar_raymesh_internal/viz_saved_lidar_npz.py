#!/usr/bin/env python3
"""
用 Open3D 查看已保存的 LiDAR npz（lidar_points），不需要分类器权重。

默认打开本包 lidar_pc/smpl_standing_default_lidar.npz（gen_standing_lidar.py 的输出）。

用法（在 smpl_lidar_raymesh_internal 下）:
  python viz_saved_lidar_npz.py
  python viz_saved_lidar_npz.py lidar_pc/smpl_standing_default_lidar.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
_DEFAULT = _THIS / "lidar_pc" / "smpl_standing_default_lidar.npz"


def _load_xyz(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as d:
        if "lidar_points" in d.files:
            xyz = np.asarray(d["lidar_points"], dtype=np.float64)
        elif "points" in d.files:
            xyz = np.asarray(d["points"], dtype=np.float64)
        else:
            raise KeyError(f"{path} 无 lidar_points/points，键: {d.files}")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"期望 (N,3)，得到 {xyz.shape}")
    return xyz


def main() -> None:
    ap = argparse.ArgumentParser(description="Open3D 查看 LiDAR npz")
    ap.add_argument(
        "npz",
        nargs="?",
        default=str(_DEFAULT),
        help=f"npz 路径，默认 {_DEFAULT.name}",
    )
    args = ap.parse_args()
    path = Path(args.npz).expanduser()
    if not path.is_absolute():
        path = (_THIS / path).resolve()
    else:
        path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    try:
        import open3d as o3d
    except ImportError:
        print("请安装: pip install open3d")
        sys.exit(1)

    xyz = _load_xyz(path)
    if xyz.shape[0] == 0:
        print("点数为 0，无法绘制")
        sys.exit(1)

    z = xyz[:, 2]
    span = max(float(z.max() - z.min()), 1e-6)
    t = (z - float(z.min())) / span
    colors = np.stack([0.2 + 0.6 * t, 0.4 + 0.3 * (1 - t), 0.9 - 0.4 * t], axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    print(f"[viz] N={xyz.shape[0]}  {path}")
    o3d.visualization.draw_geometries([pcd], window_name=path.name)


if __name__ == "__main__":
    main()
