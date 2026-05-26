#!/usr/bin/env python3
"""
生成「规范站立 / 零姿态」SMPL 对应的 LiDAR 点云，输出在本包目录下。

与 scripts/viz_pointcloud_one_frame.py 默认 SMPL 一致：
  root_pos=0, root_rot_xyzw=(0,0,0,1), dof_pos=0^69, smpl_betas=0^10

步骤：
  1) 写入 pc_smpl_dataset/smpl_standing_default.npz
  2) 调用同目录 smpl_lidar_scan.py，输出到 lidar_pc/smpl_standing_default_lidar.npz

用法（在 smpl_lidar_raymesh_internal 下）:
  python gen_standing_lidar.py
  python gen_standing_lidar.py --output-format both
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
_DATASET = _THIS / "pc_smpl_dataset"
_STEM = "smpl_standing_default"


def _write_standing_smpl_npz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root_pos = np.zeros((1, 3), dtype=np.float32)
    root_rot_xyzw = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    dof_pos = np.zeros((1, 69), dtype=np.float32)
    smpl_betas = np.zeros((1, 10), dtype=np.float32)
    np.savez_compressed(
        path,
        root_pos=root_pos,
        root_rot_xyzw=root_rot_xyzw,
        dof_pos=dof_pos,
        smpl_betas=smpl_betas,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="零姿态 SMPL → LiDAR，输出在 lidar_pc/")
    ap.add_argument(
        "--output-format",
        choices=("npz", "ply", "both"),
        default="npz",
        help="与 smpl_lidar_scan.py 相同",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(_THIS / "lidar_pc"),
        help="LiDAR 输出目录（默认本包 lidar_pc）",
    )
    args = ap.parse_args()

    smpl_npz = _DATASET / f"{_STEM}.npz"
    _write_standing_smpl_npz(smpl_npz)
    print(f"[gen_standing_lidar] 已写 SMPL: {smpl_npz}")

    scan = _THIS / "smpl_lidar_scan.py"
    out_dir = Path(args.out_dir).expanduser().resolve()
    cmd = [
        sys.executable,
        str(scan),
        str(smpl_npz),
        "-o",
        str(out_dir),
        "--output-format",
        args.output_format,
    ]
    print(f"[gen_standing_lidar] 运行: {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(_THIS))
    if r.returncode != 0:
        sys.exit(r.returncode)

    lidar_npz = out_dir / f"{_STEM}_lidar.npz"
    print(f"[gen_standing_lidar] 完成，LiDAR 预期路径: {lidar_npz}")


if __name__ == "__main__":
    main()
