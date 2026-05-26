#!/usr/bin/env python3
"""
在本目录随机选一个 SMPL 姿态 *.npz，做「训练用 1024 点云」与「射线网格 LiDAR」对比可视化。

- 1024：主工程 phc.utils.pc_anomaly.SmplToPointCloud.forward（与奖励/顶点数据集一致）。
- LiDAR：同姿态下 mesh_vertices_world → 根平移扣除后的网格 + 同目录上级包内
  smpl_lidar_scan.simulate_lidar_scan（与 smpl_lidar_scan.py 射线模型一致）。

两种点不在同一传感器坐标系内，Open3D 里将 LiDAR 沿 +X 平移以便并排看清；
若需严格对齐坐标系，需自行做传感器/根坐标变换。

用法（在本目录执行）:
  cd scripts/smpl_lidar_raymesh_internal/pc_smpl_dataset
  python viz_smpl_npz_in_this_folder.py
"""
from __future__ import annotations

import glob
import importlib.util
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_INTERNAL_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
PULSE_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))


def _load_simulate_lidar_scan():
    scan_path = Path(_INTERNAL_ROOT) / "smpl_lidar_scan.py"
    spec = importlib.util.spec_from_file_location("smpl_lidar_scan_viz", scan_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {scan_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.simulate_lidar_scan


def _pick_npz() -> str:
    paths = sorted(glob.glob(os.path.join(_THIS_DIR, "*.npz")))
    paths = [
        p
        for p in paths
        if not os.path.basename(p).endswith("_lidar.npz")
        and not os.path.basename(p).endswith("_lidar_pc.npz")
        and not os.path.basename(p).startswith("lidar_ep")
    ]
    if not paths:
        raise FileNotFoundError(
            f"本目录下没有可用 SMPL npz: {_THIS_DIR}\n"
            "请把训练导出的 smpl_*.npz（含 root_pos / root_rot_xyzw / dof_pos / smpl_betas）放到此目录。"
        )
    return random.choice(paths)


def _load_row(path: str, env_idx: int = 0):
    d = np.load(path, allow_pickle=False)
    for k in ("root_pos", "root_rot_xyzw", "dof_pos"):
        if k not in d.files:
            raise KeyError(f"{path} 缺少 {k}，现有键: {d.files}")
    rp = np.asarray(d["root_pos"], dtype=np.float32)
    rr = np.asarray(d["root_rot_xyzw"], dtype=np.float32)
    dp = np.asarray(d["dof_pos"], dtype=np.float32)
    if "smpl_betas" in d.files:
        bt = np.asarray(d["smpl_betas"], dtype=np.float32)
    elif "betas" in d.files:
        bt = np.asarray(d["betas"], dtype=np.float32)
    else:
        raise KeyError(f"{path} 缺少 smpl_betas 或 betas")

    if rp.ndim == 1:
        rp = rp.reshape(1, -1)
    if rr.ndim == 1:
        rr = rr.reshape(1, -1)
    if dp.ndim == 1:
        dp = dp.reshape(1, -1)
    if bt.ndim == 1:
        bt = bt.reshape(1, -1)

    b = int(np.clip(env_idx, 0, rp.shape[0] - 1))
    rp, rr = rp[b : b + 1], rr[b : b + 1]
    dp = dp[b : b + 1]
    bt = bt[b : b + 1]
    if dp.shape[1] < 69:
        dp = np.pad(dp, ((0, 0), (0, 69 - dp.shape[1])), mode="constant")
    elif dp.shape[1] > 69:
        dp = dp[:, :69]
    if bt.shape[1] < 10:
        bt = np.pad(bt, ((0, 0), (0, 10 - bt.shape[1])), mode="constant")
    elif bt.shape[1] > 10:
        bt = bt[:, :10]

    root_pos = torch.from_numpy(rp).float()
    root_rot = torch.from_numpy(rr).float()
    dof_pos = torch.from_numpy(dp).float()
    betas = torch.from_numpy(bt).float()
    return root_pos, root_rot, dof_pos, betas


def main() -> None:
    npz_path = _pick_npz()
    print(f"[viz] 随机选中: {npz_path}")
    print(f"[viz] PULSE_ROOT={PULSE_ROOT}（主工程 SmplToPointCloud + 1024）")
    print(f"[viz] LiDAR 射线模型来自: {_INTERNAL_ROOT}/smpl_lidar_scan.py")

    sys.path.insert(0, PULSE_ROOT)
    simulate_lidar_scan = _load_simulate_lidar_scan()

    device = torch.device("cpu")
    root_pos, root_rot, dof_pos, betas = _load_row(npz_path, 0)
    root_pos = root_pos.to(device)
    root_rot = root_rot.to(device)
    dof_pos = dof_pos.to(device)
    betas = betas.to(device)

    smpl_path = os.path.join(PULSE_ROOT, "data", "smpl")
    if not os.path.isdir(smpl_path):
        raise FileNotFoundError(f"未找到 SMPL 模型目录: {smpl_path}")

    from phc.utils.pc_anomaly import SmplToPointCloud

    smpl_to_pc = SmplToPointCloud(
        smpl_model_path=smpl_path,
        num_betas=10,
        num_points=1024,
        local_coord=True,
        device=device,
    )
    with torch.no_grad():
        pc1024 = smpl_to_pc(
            root_pos=root_pos,
            root_rot_xyzw=root_rot,
            dof_pos=dof_pos,
            betas=betas,
        )
        mesh_w = smpl_to_pc.mesh_vertices_world(
            root_pos=root_pos,
            root_rot_xyzw=root_rot,
            dof_pos=dof_pos,
            betas=betas,
        )
    xyz1024 = pc1024[0].detach().cpu().numpy()
    # 与 forward 中 local_coord 一致：世界顶点减根平移，供射线求交与 smpl_lidar_scan 默认局部网格同构
    mesh_local = (mesh_w - root_pos.unsqueeze(1))[0].detach().cpu().numpy().astype(np.float32)
    faces = smpl_to_pc.mesh_faces_numpy().astype(np.int32, copy=False)

    lidar = simulate_lidar_scan(
        mesh_local,
        faces,
        distance=10.0,
        sensor_height=1.3,
        horizontal_res_deg=0.2,
        seed=7,
    )
    lidar_xyz = np.asarray(lidar["points"], dtype=np.float64)
    intensity = np.asarray(lidar["intensity"], dtype=np.float64)

    print(f"[viz] 1024 点数: {xyz1024.shape[0]}，LiDAR 命中点数: {lidar_xyz.shape[0]}")

    try:
        import open3d as o3d
    except ImportError:
        print("请先安装: pip install open3d")
        sys.exit(1)

    pcd1024 = o3d.geometry.PointCloud()
    pcd1024.points = o3d.utility.Vector3dVector(xyz1024.astype(np.float64))
    pcd1024.colors = o3d.utility.Vector3dVector(np.tile([0.25, 0.55, 0.95], (xyz1024.shape[0], 1)))

    pcd_lidar = o3d.geometry.PointCloud()
    if lidar_xyz.shape[0] == 0:
        print("[viz] 警告: LiDAR 无命中点（可检查姿态/距离参数）；仅显示 1024。")
        geoms = [pcd1024]
    else:
        colors = np.clip(np.stack([intensity, 0.35 + 0.4 * intensity, 0.1 + 0.2 * intensity], axis=1), 0.0, 1.0)
        pcd_lidar.points = o3d.utility.Vector3dVector(lidar_xyz)
        pcd_lidar.colors = o3d.utility.Vector3dVector(colors)
        # 并排：LiDAR 在传感器系，1024 在根局部系，仅作视觉对比
        pcd_lidar.translate(np.array([2.8, 0.0, 0.0], dtype=np.float64))
        geoms = [pcd1024, pcd_lidar]

    o3d.visualization.draw_geometries(
        geoms,
        window_name="蓝=训练1024(原点)  橙=LiDAR(+X平移) | " + os.path.basename(npz_path),
    )


if __name__ == "__main__":
    main()
