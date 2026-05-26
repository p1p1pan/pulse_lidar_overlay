#!/usr/bin/env python3
"""
加载训练保存的 LiDAR npz（lidar_points 等），重采样到 1024 点后做与主训练一致的 pc_normalize_torch，
再送入 PointNet++ person_vs_rest 分类器，打印 p_human，并用 Open3D 可视化原始 LiDAR 点云。

用法（仓库根目录）:
  python scripts/viz_lidar_npz_classifier.py
  python scripts/viz_lidar_npz_classifier.py --dir output/pc_lidar_dataset
  python scripts/viz_lidar_npz_classifier.py output/pc_lidar_dataset/lidar_ep000001_h000_000000.npz

默认 --device cpu（避免与 Isaac 等占满 GPU 时在 .to(cuda) 即 OOM）。需要 GPU 时加 --device cuda:0；
若 CUDA 仍失败（含 OOM），会自动改跑 CPU。
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys

import numpy as np
import torch

PULSE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PULSE_ROOT)


def _is_cuda_runtime_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("cuda" in msg) or ("cublas" in msg) or ("cudnn" in msg)


def _is_cuda_oom(exc: Exception) -> bool:
    return "out of memory" in str(exc).lower()


def _should_fallback_to_cpu(exc: Exception) -> bool:
    return _is_cuda_oom(exc) or _is_cuda_runtime_error(exc)


def _list_npz_in_dir(root_dir: str) -> list[str]:
    if not os.path.isdir(root_dir):
        return []
    return sorted(glob.glob(os.path.join(root_dir, "*.npz")))


def _resolve_npz_path(npz_arg: str | None, lidar_dir: str, pick_seed: int | None) -> str:
    """若未给文件、路径不存在或是目录，则在 lidar_dir（或给定目录）下随机选一个 *.npz。"""
    rng = random.Random(pick_seed) if pick_seed is not None else random.SystemRandom()

    def _abs(p: str) -> str:
        if os.path.isabs(p):
            return os.path.abspath(p)
        cand = os.path.join(os.getcwd(), p)
        if os.path.isfile(cand) or os.path.isdir(cand):
            return os.path.abspath(cand)
        cand2 = os.path.join(PULSE_ROOT, p)
        return os.path.abspath(cand2)

    if npz_arg:
        p = _abs(npz_arg)
        if os.path.isfile(p):
            return p
        if os.path.isdir(p):
            files = _list_npz_in_dir(p)
            if files:
                return rng.choice(files)
            raise FileNotFoundError(f"目录内无 npz: {p}")
        print(f"[viz_lidar] 未找到文件 {npz_arg!r}，改为在 LiDAR 目录下随机选择")

    root_dir = _abs(lidar_dir)
    files = _list_npz_in_dir(root_dir)
    if not files:
        raise FileNotFoundError(f"在 {root_dir} 未找到任何 .npz，请先训练保存 lidar 或检查 --dir")
    return rng.choice(files)


def _jet_colors_by_z(xyz: np.ndarray) -> np.ndarray:
    """按 Z 高度映射 jet 色（与参考点云图一致）。"""
    z = xyz[:, 2]
    zmin, zmax = float(z.min()), float(z.max())
    t = (z - zmin) / max(zmax - zmin, 1e-6)
    try:
        import matplotlib.cm as cm

        return cm.get_cmap("jet")(t)[:, :3].astype(np.float64)
    except ImportError:
        # 无 matplotlib 时的简易彩虹近似
        h = t
        r = np.clip(1.5 - np.abs(4.0 * h - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * h - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * h - 1.0), 0.0, 1.0)
        return np.stack([r, g, b], axis=1)


def _show_lidar_open3d(xyz: np.ndarray, title: str) -> None:
    """Open3D 交互窗口：可旋转/缩放；jet 按高度着色 + 深色背景。"""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(_jet_colors_by_z(xyz))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title[:200], width=1024, height=768)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1], dtype=np.float64)
    opt.point_size = 4.0
    vis.reset_view_point(True)
    print("[viz_lidar] Open3D：鼠标拖拽旋转，滚轮缩放，关闭窗口退出")
    vis.run()
    vis.destroy_window()


def resample_points(pts: np.ndarray, n: int, seed: int) -> np.ndarray:
    """(N,3) -> (n,3)，N>=n 无放回，否则有放回。"""
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"lidar_points 期望 (N,3)，得到 {pts.shape}")
    rng = np.random.default_rng(seed)
    n_pts = pts.shape[0]
    if n_pts == 0:
        return np.zeros((n, 3), dtype=np.float32)
    if n_pts >= n:
        idx = rng.choice(n_pts, size=n, replace=False)
    else:
        idx = rng.choice(n_pts, size=n, replace=True)
    return pts[idx].copy()


def main() -> None:
    ap = argparse.ArgumentParser(description="LiDAR npz + PointNet++ 分类 + Open3D 可视化")
    ap.add_argument(
        "npz",
        type=str,
        nargs="?",
        default=None,
        help="可选：具体 npz 路径；省略或文件不存在时从 --dir 随机选一个",
    )
    ap.add_argument(
        "--dir",
        type=str,
        default=os.path.join(PULSE_ROOT, "output", "pc_lidar_dataset"),
        help="随机挑选 *.npz 的目录（默认仓库内 output/pc_lidar_dataset）",
    )
    ap.add_argument(
        "--pick-seed",
        type=int,
        default=None,
        help="随机选文件的种子；不设则每次运行随机不同",
    )
    ap.add_argument(
        "--classifier",
        type=str,
        default=os.path.join(PULSE_ROOT, "output", "pc_classifier", "person_vs_rest.pth"),
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="默认 cpu，避免与 Isaac 等占满 GPU 冲突；需要时用 cuda:0",
    )
    ap.add_argument("--num_points", type=int, default=1024, help="送入分类器的点数（与训练一致）")
    ap.add_argument("--seed", type=int, default=0, help="重采样随机种子")
    ap.add_argument("--human_class_idx", type=int, default=1)
    ap.add_argument("--no_interactive", action="store_true", help="不弹窗，仅打印分类结果")
    args = ap.parse_args()

    npz_path = _resolve_npz_path(args.npz, args.dir, args.pick_seed)
    print(f"[viz_lidar] 使用文件: {npz_path}")

    if not os.path.isfile(args.classifier):
        raise FileNotFoundError(args.classifier)

    d = np.load(npz_path)
    if "lidar_points" not in d.files:
        raise KeyError(f"npz 中无 lidar_points，现有键: {d.files}")
    pts_all = np.asarray(d["lidar_points"], dtype=np.float32)
    print(f"[viz_lidar] 加载 lidar_points shape={pts_all.shape}")
    if "p_human" in d.files:
        print(f"[viz_lidar] npz 内记录的 p_human(顶点分支): {np.asarray(d['p_human'])}")

    cls_np = resample_points(pts_all, args.num_points, args.seed)
    pc_cpu = torch.from_numpy(cls_np).float().unsqueeze(0)

    from phc.utils.pc_anomaly import (
        build_pc_backbone,
        pc_normalize_torch,
        PointCloudMotionClassifier,
    )

    def _run_cls(pc_n: torch.Tensor, dev: torch.device):
        backbone = build_pc_backbone(backbone_type="pointnet2", feat_dim=256, in_channels=3)
        clf = PointCloudMotionClassifier(backbone=backbone, feat_dim=1024, num_classes=2).to(dev)
        state = torch.load(args.classifier, map_location=dev)
        sd = state.get("state_dict") or state
        clf.load_state_dict(sd, strict=False)
        clf.eval()
        with torch.no_grad():
            logits = clf(pc_n.to(dev))
            probs = torch.softmax(logits, dim=-1)
        return logits, probs

    device = torch.device(args.device)
    logits = probs = None
    while True:
        try:
            pc = pc_cpu.to(device)
            pc_n = pc_normalize_torch(pc)
            logits, probs = _run_cls(pc_n, device)
            break
        except RuntimeError as e:
            if device.type == "cuda" and _should_fallback_to_cpu(e):
                print(f"[viz_lidar] CUDA 失败（含显存不足），改用 CPU: {e}")
                torch.cuda.empty_cache()
                device = torch.device("cpu")
                continue
            raise

    p_human = float(probs[0, args.human_class_idx].item())
    p_other = float(probs[0, 1 - args.human_class_idx].item())
    print(f"[viz_lidar] 分类器 p_human (人形, idx={args.human_class_idx}): {p_human:.4f}")
    print(f"[viz_lidar] 分类器 p_non_human: {p_other:.4f}")
    print(f"[viz_lidar] logits: {logits[0].detach().cpu().numpy().tolist()}")

    if args.no_interactive:
        return

    title = f"LiDAR N={pts_all.shape[0]} p_human={p_human:.4f}"
    try:
        _show_lidar_open3d(pts_all.astype(np.float64), title)
    except ImportError:
        print("[viz_lidar] 请安装 open3d: pip install open3d")
        sys.exit(1)


if __name__ == "__main__":
    main()
