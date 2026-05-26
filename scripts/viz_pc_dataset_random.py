#!/usr/bin/env python3
"""
从训练保存的目录中随机选一个 npz，可视化其中的点云（键 `pc`），并输出分类器给出的「人形」概率 p_human。

用法（在仓库根目录）:
  python scripts/viz_pc_dataset_random.py
  python scripts/viz_pc_dataset_random.py --classifier output/pc_classifier/person_vs_rest.pth
  python scripts/viz_pc_dataset_random.py --dir output/pc_anomaly_dataset --seed 42
  python scripts/viz_pc_dataset_random.py --count 3
  python scripts/viz_pc_dataset_random.py --no_interactive --out /tmp/pc_rand.png
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys

import numpy as np

PULSE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PULSE_ROOT)

_DEFAULT_CLASSIFIER = os.path.join(PULSE_ROOT, "output", "pc_classifier", "person_vs_rest.pth")


def _load_pc_arrays(path: str, env_index: int) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """返回 (pc 可视化 Nx3, pc_cls 分类用 Nx3 或 None, meta)。"""
    d = np.load(path)
    pc = np.asarray(d["pc"], dtype=np.float32)
    pc_cls = None
    if "pc_cls" in d.files:
        pc_cls = np.asarray(d["pc_cls"], dtype=np.float32)
    meta: dict = {}
    for k in ("p_human", "epoch", "horizon_step", "per_max", "mean_max"):
        if k in d.files:
            meta[k] = np.asarray(d[k])

    def _take_batch(x: np.ndarray) -> np.ndarray:
        if x.ndim == 3:
            b = int(np.clip(env_index, 0, x.shape[0] - 1))
            return x[b]
        if x.ndim == 2 and x.shape[-1] == 3:
            return x
        raise ValueError(f"点云形状异常: {x.shape}，期望 (N,3) 或 (B,N,3)")

    pc = _take_batch(pc)
    if pc_cls is not None:
        pc_cls = _take_batch(pc_cls)
    return pc, pc_cls, meta


def _npz_recorded_p_human(meta: dict, env_index: int) -> float | None:
    if "p_human" not in meta:
        return None
    ph = meta["p_human"]
    if ph.ndim == 0:
        return float(ph)
    ph = ph.reshape(-1)
    if ph.size == 0:
        return None
    i = int(np.clip(env_index, 0, ph.size - 1))
    return float(ph[i])


def _print_meta(path: str, meta: dict, env_index: int) -> None:
    print(f"[viz_pc_dataset_random] 文件: {path}")
    if "p_human" in meta:
        ph = meta["p_human"]
        if ph.ndim == 0:
            print(f"  npz 内保存的 p_human: {float(ph):.4f}")
        else:
            print(
                f"  npz 内 p_human: shape={ph.shape} "
                f"本 env[{env_index}]={_npz_recorded_p_human(meta, env_index):.4f} "
                f"(全 batch mean={float(ph.mean()):.4f})"
            )
    for k in ("epoch", "horizon_step", "per_max", "mean_max"):
        if k in meta:
            print(f"  {k}: {meta[k]}")


def _infer_p_human_cls(
    pc_cls_xyz: np.ndarray,
    weights_path: str,
    device_str: str,
    human_class_idx: int,
) -> float:
    import torch
    from phc.utils.pc_anomaly import build_pc_backbone, PointCloudMotionClassifier

    device = torch.device(device_str)
    t = torch.from_numpy(pc_cls_xyz).float().unsqueeze(0).to(device)
    backbone = build_pc_backbone(backbone_type="pointnet2", feat_dim=256, in_channels=3)
    classifier = PointCloudMotionClassifier(backbone=backbone, feat_dim=1024, num_classes=2).to(device)
    state = torch.load(weights_path, map_location=device)
    sd = state.get("state_dict") or state
    classifier.load_state_dict(sd, strict=False)
    classifier.eval()
    with torch.no_grad():
        logits = classifier(t)
    probs = torch.softmax(logits, dim=-1)
    return float(probs[0, human_class_idx].item())


def _show_open3d(pc_xyz: np.ndarray, window_title: str) -> bool:
    try:
        import open3d as o3d
    except ImportError:
        return False
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_xyz)
    pcd.colors = o3d.utility.Vector3dVector(np.tile([0.4, 0.6, 0.9], (pc_xyz.shape[0], 1)))
    title = window_title[:200] if len(window_title) > 200 else window_title
    o3d.visualization.draw_geometries([pcd], window_name=title)
    return True


def _save_png(pc_xyz: np.ndarray, out_path: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pc_xyz[:, 0], pc_xyz[:, 1], pc_xyz[:, 2], c=pc_xyz[:, 2], cmap="viridis", s=2, alpha=0.8)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[viz_pc_dataset_random] 已保存: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="随机可视化 pc_anomaly_dataset 目录下的 npz 点云")
    ap.add_argument("--dir", type=str, default=os.path.join(PULSE_ROOT, "output", "pc_anomaly_dataset"))
    ap.add_argument("--glob", type=str, default="*.npz", help="相对 --dir 的 glob，例如 'pc_*.npz'")
    ap.add_argument("--seed", type=int, default=None, help="随机种子（默认可复现则每次不同）")
    ap.add_argument("--count", type=int, default=1, help="连续随机展示个数")
    ap.add_argument("--npz_env", type=int, default=0, help="pc 为 (B,N,3) 时取第几个 batch")
    ap.add_argument("--no_interactive", action="store_true", help="不弹 Open3D，只保存图片")
    ap.add_argument("--out", type=str, default=None, help="非交互时输出 png 路径（仅 count=1 时生效；多帧会加后缀）")
    ap.add_argument(
        "--classifier",
        type=str,
        default=_DEFAULT_CLASSIFIER,
        help="PointNet++ 二分类权重；默认 output/pc_classifier/person_vs_rest.pth，不存在则仅打印 npz 内 p_human（若有）",
    )
    ap.add_argument("--device", type=str, default="cuda:0", help="分类器推理设备，如 cuda:0 或 cpu")
    ap.add_argument("--human_class_idx", type=int, default=1, help="logits 中人形类别下标，与训练一致时为 1")
    args = ap.parse_args()

    pattern = os.path.join(args.dir, args.glob)
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[viz_pc_dataset_random] 未找到 npz: {pattern}")
        sys.exit(1)

    if args.seed is not None:
        random.seed(args.seed)

    for i in range(max(1, args.count)):
        path = random.choice(files)
        pc_xyz, pc_cls, meta = _load_pc_arrays(path, args.npz_env)
        _print_meta(path, meta, args.npz_env)

        p_human_cls: float | None = None
        use_weights = args.classifier and os.path.isfile(args.classifier)
        if use_weights:
            cls_input = pc_cls if pc_cls is not None else pc_xyz
            if pc_cls is None:
                print("  [提示] npz 无 pc_cls，用 pc 直接送入分类器（旧格式，已是归一化输入）。")
            try:
                p_human_cls = _infer_p_human_cls(
                    cls_input, args.classifier, args.device, args.human_class_idx
                )
                print(f"  分类器当前推理 p_human（人形概率）: {p_human_cls:.4f}")
            except Exception as e:
                print(f"  [警告] 分类器推理失败: {e}；仍继续可视化。")
        else:
            print(f"  [提示] 未找到权重文件: {args.classifier}；跳过分类器推理。可传 --classifier 路径。")

        print(
            f"  点数 N={pc_xyz.shape[0]} 范围 X[{pc_xyz[:, 0].min():.3f},{pc_xyz[:, 0].max():.3f}] "
            f"Y[{pc_xyz[:, 1].min():.3f},{pc_xyz[:, 1].max():.3f}] Z[{pc_xyz[:, 2].min():.3f},{pc_xyz[:, 2].max():.3f}]"
        )

        if p_human_cls is not None:
            win_title = f"p_human={p_human_cls:.4f} | {os.path.basename(path)}"
            mpl_title = f"p_human={p_human_cls:.4f} | random npz"
        else:
            rec = _npz_recorded_p_human(meta, args.npz_env)
            if rec is not None:
                win_title = f"npz_p_human={rec:.4f} | {os.path.basename(path)}"
                mpl_title = f"npz p_human={rec:.4f} | random npz"
            else:
                win_title = f"pc | {os.path.basename(path)}"
                mpl_title = "random npz → pc"

        if args.no_interactive:
            out = args.out
            if out is None:
                out = os.path.join(PULSE_ROOT, "output", "pc_classifier", "viz_pc_dataset_random.png")
            if args.count > 1:
                root, ext = os.path.splitext(out)
                out = f"{root}_{i:02d}{ext or '.png'}"
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            _save_png(pc_xyz, out, mpl_title)
        else:
            if not _show_open3d(pc_xyz, win_title):
                print("[viz_pc_dataset_random] 未安装 open3d，请: pip install open3d 或使用 --no_interactive")
                sys.exit(1)


if __name__ == "__main__":
    main()
