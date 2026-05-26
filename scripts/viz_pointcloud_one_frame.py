#!/usr/bin/env python3
"""
可视化「一帧 SMPL → 点云」的结果，默认直接弹出交互式窗口。

用法:
  python scripts/viz_pointcloud_one_frame.py
  python scripts/viz_pointcloud_one_frame.py                 # 默认交互式 3D（需 open3d）
  python scripts/viz_pointcloud_one_frame.py --no_interactive  # 不弹窗，仅保存图片
  python scripts/viz_pointcloud_one_frame.py --motion_file sample_data/amass_isaac_standing_upright_slim.pkl --frame 0
  python scripts/viz_pointcloud_one_frame.py --from_npz output/pc_anomaly_dataset/pc_ep000500_h000_000001.npz  # npz：pc=SmplToPointCloud 与两姿态散点一致；pc_cls=分类器输入
  python scripts/viz_pointcloud_one_frame.py --weird   # 随机怪异姿态
"""
import os
import sys
import argparse
import itertools
import numpy as np
import torch

PULSE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PULSE_ROOT)


def _is_cuda_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("cuda" in msg) and ("out of memory" in msg)

def _is_cuda_runtime_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("cuda" in msg) or ("cublas" in msg) or ("cudnn" in msg)

def _axis_transform_variants():
    perms = list(itertools.permutations([0, 1, 2]))
    signs = list(itertools.product([1.0, -1.0], repeat=3))
    variants = []
    for p in perms:
        for s in signs:
            variants.append((p, s))
    return variants

def _apply_axis_transform(pc_tensor: torch.Tensor, perm, sign):
    out = pc_tensor[:, :, list(perm)]
    sign_t = torch.tensor(sign, dtype=out.dtype, device=out.device).view(1, 1, 3)
    return out * sign_t

def _quat_rotate_xyzw(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_xyz = q[..., 0:3]
    q_w = q[..., 3:4]
    t = 2.0 * torch.cross(q_xyz, v, dim=-1)
    return v + q_w * t + torch.cross(q_xyz, t, dim=-1)


def main():
    parser = argparse.ArgumentParser(description="Visualize one frame SMPL→point cloud")
    parser.add_argument("--motion_file", type=str, default=None, help="AMASS 格式 pkl，取其中一帧")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--weird", action="store_true", help="使用随机怪异姿态（用于对比）")
    parser.add_argument("--out", type=str, default=None, help="输出路径，默认 output/pc_classifier/viz_pointcloud.png")
    parser.add_argument("--classifier", type=str, default=None, help="可选：person_vs_rest.pth 路径，输出该帧的 p_human")
    parser.add_argument("--print_detail", action="store_true", help="打印分类器两类概率、预测类别和logits")
    parser.add_argument("--search_axis_transform", action="store_true", help="搜索轴置换/翻转，查看是否存在更高 p_human 的坐标系")
    parser.add_argument("--use_root_local_frame", action="store_true", help="先将点云变换到 root 局部坐标，再做分类")
    parser.add_argument("--from_sim", action="store_true", help="从仿真抓取的一帧（output/pc_classifier/last_kin_frame.pkl）加载，需先跑训练产生")
    parser.add_argument("--from_npz", type=str, default=None, help="从训练保存的 npz：pc 为 SmplToPointCloud 输出（与脚本两姿态散点一致）；若有 pc_cls 则分类用其且不再 normalize")
    parser.add_argument("--npz_env", type=int, default=0, help="npz 中 pc 为 (B,N,3) 时取第几个 env")
    parser.add_argument("--interactive", action="store_true", help="交互式 3D 窗口，可旋转查看点云（需 open3d: pip install open3d）")
    parser.add_argument("--no_interactive", action="store_true", help="禁用交互窗口，仅保存图片")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    last_kin_path = os.path.join(PULSE_ROOT, "output", "pc_classifier", "last_kin_frame.pkl")

    from_npz_pc = None
    from_npz_pc_cls = None
    if args.from_npz:
        if not os.path.isfile(args.from_npz):
            raise FileNotFoundError(args.from_npz)
        d = np.load(args.from_npz)
        pc_np_raw = np.asarray(d["pc"], dtype=np.float32)
        if pc_np_raw.ndim == 3:
            pc_np_raw = pc_np_raw[int(args.npz_env)]
        from_npz_pc = torch.from_numpy(pc_np_raw).float().unsqueeze(0)
        if "pc_cls" in d.files:
            cls_np = np.asarray(d["pc_cls"], dtype=np.float32)
            if cls_np.ndim == 3:
                cls_np = cls_np[int(args.npz_env)]
            from_npz_pc_cls = torch.from_numpy(cls_np).float().unsqueeze(0)
        print(
            f"[viz] 从 npz 加载: {args.from_npz} pc(viz) shape={tuple(from_npz_pc.shape)}"
            + (
                f", pc_cls shape={tuple(from_npz_pc_cls.shape)}"
                if from_npz_pc_cls is not None
                else "（旧格式：仅 pc，与当时分类器输入一致）"
            )
        )

    # 优先从仿真抓取的帧加载
    if from_npz_pc is not None:
        root_pos = root_rot = dof_pos = betas = None
    elif args.from_sim and os.path.isfile(last_kin_path):
        import joblib
        data = joblib.load(last_kin_path)
        rp, rr, dp, bt = data["root_pos"], data["root_rot"], data["dof_pos"], data["betas"]
        root_pos = torch.from_numpy(np.asarray(rp, dtype=np.float32))
        root_rot = torch.from_numpy(np.asarray(rr, dtype=np.float32))
        dof_pos = torch.from_numpy(np.asarray(dp, dtype=np.float32))
        betas = torch.from_numpy(np.asarray(bt, dtype=np.float32))
        if root_pos.dim() == 1:
            root_pos = root_pos.unsqueeze(0)
        if root_rot.dim() == 1:
            root_rot = root_rot.unsqueeze(0)
        if dof_pos.dim() == 1:
            dof_pos = dof_pos.unsqueeze(0)
        if betas.dim() == 1:
            betas = betas.unsqueeze(0)
        print(f"[viz] 从仿真抓取帧加载: {last_kin_path}")
    elif args.from_sim:
        raise FileNotFoundError(
            f"未找到 {last_kin_path}。请先跑训练（use_pc_anomaly_loss=True）若干步，"
            "仿真会将该帧保存到该路径。"
        )
    elif args.motion_file and os.path.isfile(args.motion_file):
        import joblib
        from scipy.spatial.transform import Rotation
        data = joblib.load(args.motion_file)
        motion = data[list(data.keys())[0]] if isinstance(data, dict) else data
        if "root_trans_offset" in motion and motion.get("root_trans_offset") is not None:
            root_trans = motion.get("root_trans_offset")
        else:
            root_trans = motion.get("root_trans")
        if "pose_aa" in motion and motion.get("pose_aa") is not None:
            pose_aa = motion.get("pose_aa")
        else:
            pose_aa = motion.get("poses")
        if root_trans is None or pose_aa is None:
            raise KeyError(f"pkl 需含 root_trans_offset 和 pose_aa，当前 keys: {list(motion.keys())}")
        root_trans = np.asarray(root_trans)
        pose_aa = np.asarray(pose_aa)
        if root_trans.ndim == 1:
            root_trans = root_trans.reshape(1, -1)
        if pose_aa.ndim == 1:
            pose_aa = pose_aa.reshape(1, -1)
        frame = min(args.frame, root_trans.shape[0] - 1) if root_trans.shape[0] > 0 else 0
        root_pos = torch.from_numpy(root_trans[frame : frame + 1, :3].astype(np.float32))
        go = pose_aa[frame, :3]
        rot = Rotation.from_rotvec(go)
        quat = rot.as_quat()  # (x,y,z,w)
        root_rot = torch.from_numpy(quat.astype(np.float32)).unsqueeze(0)
        body_pose = pose_aa[frame, 3:]
        if body_pose.shape[0] < 69:
            body_pose = np.pad(body_pose, (0, 69 - body_pose.shape[0]), mode="constant")
        dof_pos = torch.from_numpy(body_pose[:69].astype(np.float32)).unsqueeze(0)
        betas = torch.zeros(1, 10)
        if "beta" in motion and motion.get("beta") is not None:
            b = np.asarray(motion.get("beta"))[:10]
            betas = torch.from_numpy(b.astype(np.float32)).unsqueeze(0)
        elif "betas" in motion and motion.get("betas") is not None:
            b = np.asarray(motion.get("betas"))[:10]
            betas = torch.from_numpy(b.astype(np.float32)).unsqueeze(0)
        print(f"[viz] 从 {args.motion_file} 第 {frame} 帧加载")
    elif from_npz_pc is None:
        # 默认：站立姿态。SMPL body_pose 为 69 维（23 关节 × 3）
        root_pos = torch.zeros(1, 3)
        root_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]])  # identity xyzw
        dof_pos = torch.zeros(1, 69)
        if args.weird:
            np.random.seed(42)
            dof_pos = torch.from_numpy(np.random.randn(1, 69).astype(np.float32) * 0.5)
            print("[viz] 使用随机怪异姿态")
        else:
            print("[viz] 使用站立姿态")
        betas = torch.zeros(1, 10)

    if from_npz_pc is None:
        # 确保 betas 与 SmplToPointCloud 的 num_betas=10 一致（仿真可能存 16 维）
        if betas.shape[1] > 10:
            betas = betas[:, :10]
        elif betas.shape[1] < 10:
            pad = torch.zeros(betas.shape[0], 10 - betas.shape[1], dtype=betas.dtype)
            betas = torch.cat([betas, pad], dim=1)

    # SMPL → 点云（npz 已跳过）
    from phc.utils.pc_anomaly import SmplToPointCloud

    if from_npz_pc is not None:
        pc = from_npz_pc.to(device)
    else:
        try:
            smpl_to_pc = SmplToPointCloud(
                smpl_model_path="data/smpl",
                num_betas=10,
                num_points=1024,
                local_coord=True,
                device=device,
            )
            with torch.no_grad():
                pc = smpl_to_pc(
                    root_pos=root_pos.to(device),
                    root_rot_xyzw=root_rot.to(device),
                    dof_pos=dof_pos.to(device),
                    betas=betas.to(device),
                )
        except RuntimeError as e:
            if device.type == "cuda" and (_is_cuda_oom(e) or _is_cuda_runtime_error(e)):
                print(f"[viz] CUDA 阶段失败({type(e).__name__})，自动回退到 CPU 继续。")
                torch.cuda.empty_cache()
                device = torch.device("cpu")
                smpl_to_pc = SmplToPointCloud(
                    smpl_model_path="data/smpl",
                    num_betas=10,
                    num_points=1024,
                    local_coord=True,
                    device=device,
                )
                with torch.no_grad():
                    pc = smpl_to_pc(
                        root_pos=root_pos.to(device),
                        root_rot_xyzw=root_rot.to(device),
                        dof_pos=dof_pos.to(device),
                        betas=betas.to(device),
                    )
            else:
                raise
    pc_np = pc[0].cpu().numpy()  # (N, 3)

    if args.classifier and os.path.isfile(args.classifier):
        from phc.utils.pc_anomaly import build_pc_backbone, PointCloudMotionClassifier, pc_normalize_torch
        if from_npz_pc_cls is not None:
            pc_for_cls = from_npz_pc_cls.to(device)
        elif from_npz_pc is not None:
            pc_for_cls = from_npz_pc.to(device)
        else:
            pc_for_cls = pc.to(device)
        if args.use_root_local_frame and from_npz_pc is None:
            rr = root_rot.to(device)
            rr = torch.nn.functional.normalize(rr, p=2, dim=-1)
            rr_inv = torch.cat([-rr[:, 0:3], rr[:, 3:4]], dim=-1)
            rr_inv = rr_inv[:, None, :].expand(-1, pc_for_cls.shape[1], -1)
            pc_for_cls = _quat_rotate_xyzw(rr_inv.reshape(-1, 4), pc_for_cls.reshape(-1, 3)).view(pc_for_cls.shape[0], pc_for_cls.shape[1], 3)
        try:
            backbone = build_pc_backbone(backbone_type="pointnet2", feat_dim=256, in_channels=3)
            classifier = PointCloudMotionClassifier(backbone=backbone, feat_dim=1024, num_classes=2).to(device)
            state = torch.load(args.classifier, map_location=device)
            sd = state.get("state_dict") or state
            classifier.load_state_dict(sd, strict=False)
            classifier.eval()
            with torch.no_grad():
                if from_npz_pc is not None:
                    logits = classifier(pc_for_cls)
                else:
                    pc_norm = pc_normalize_torch(pc_for_cls)
                    logits = classifier(pc_norm)
                probs = torch.softmax(logits, dim=-1)
        except RuntimeError as e:
            if device.type == "cuda" and (_is_cuda_oom(e) or _is_cuda_runtime_error(e)):
                print(f"[viz] 分类器阶段 CUDA 失败({type(e).__name__})，自动回退到 CPU。")
                torch.cuda.empty_cache()
                device = torch.device("cpu")
                backbone = build_pc_backbone(backbone_type="pointnet2", feat_dim=256, in_channels=3)
                classifier = PointCloudMotionClassifier(backbone=backbone, feat_dim=1024, num_classes=2).to(device)
                state = torch.load(args.classifier, map_location=device)
                sd = state.get("state_dict") or state
                classifier.load_state_dict(sd, strict=False)
                classifier.eval()
                with torch.no_grad():
                    if from_npz_pc is not None:
                        logits = classifier(pc_for_cls.to(device))
                    else:
                        pc_norm = pc_normalize_torch(pc_for_cls.to(device))
                        logits = classifier(pc_norm)
                    probs = torch.softmax(logits, dim=-1)
            else:
                raise
        p_non_human = probs[0, 0].item()
        p_human = probs[0, 1].item()  # 训练时 label 1=person，故 logits[:,1]=人形
        print(f"分类器 p_human (→人形): {p_human:.4f}")
        if args.print_detail:
            pred_idx = int(torch.argmax(probs, dim=-1)[0].item())
            print(
                f"分类器详情: p_non_human={p_non_human:.4f}, p_human={p_human:.4f}, "
                f"pred_class={pred_idx}, logits={logits[0].detach().cpu().numpy().tolist()}"
            )
        if args.search_axis_transform and from_npz_pc is None:
            best = None
            with torch.no_grad():
                for perm, sign in _axis_transform_variants():
                    pc_try = _apply_axis_transform(pc_for_cls, perm, sign)
                    pc_try = pc_normalize_torch(pc_try)
                    logits_try = classifier(pc_try)
                    probs_try = torch.softmax(logits_try, dim=-1)
                    p_h = probs_try[0, 1].item()
                    if best is None or p_h > best["p_human"]:
                        best = {
                            "perm": perm,
                            "sign": sign,
                            "p_human": p_h,
                            "p_non_human": probs_try[0, 0].item(),
                            "logits": logits_try[0].detach().cpu().numpy().tolist(),
                        }
            if best is not None:
                print(
                    "[axis-search] best transform: "
                    f"perm={best['perm']} sign={best['sign']} "
                    f"p_human={best['p_human']:.4f} p_non_human={best['p_non_human']:.4f} "
                    f"logits={best['logits']}"
                )

    os.makedirs(os.path.join(PULSE_ROOT, "output", "pc_classifier"), exist_ok=True)
    out_path = args.out or os.path.join(PULSE_ROOT, "output", "pc_classifier", "viz_pointcloud.png")
    use_interactive = (args.interactive or not args.no_interactive)

    print(f"点云形状: {pc_np.shape}，范围 X[{pc_np[:,0].min():.3f}, {pc_np[:,0].max():.3f}] Y[{pc_np[:,1].min():.3f}, {pc_np[:,1].max():.3f}] Z[{pc_np[:,2].min():.3f}, {pc_np[:,2].max():.3f}]")

    # 交互式 3D 查看（默认启用）
    if use_interactive:
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc_np)
            pcd.colors = o3d.utility.Vector3dVector(np.tile([0.4, 0.6, 0.9], (pc_np.shape[0], 1)))
            print("[viz] 打开交互式 3D 窗口，鼠标拖拽旋转，滚轮缩放，关闭窗口退出")
            o3d.visualization.draw_geometries([pcd], window_name="Point Cloud")
            return
        except ImportError:
            print("[viz] 未安装 open3d，改为保存图片。可执行: pip install open3d")

    # 非交互模式或 open3d 不可用时，保存图片
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pc_np[:, 0], pc_np[:, 1], pc_np[:, 2], c=pc_np[:, 2], cmap="viridis", s=2, alpha=0.8)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("SMPL → Point Cloud (1 frame)")
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"已保存: {out_path}")
    except ImportError:
        np.save(os.path.join(PULSE_ROOT, "output", "pc_classifier", "viz_pointcloud.npy"), pc_np)
        print("未安装 matplotlib，已保存 output/pc_classifier/viz_pointcloud.npy")


if __name__ == "__main__":
    main()
