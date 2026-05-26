import argparse
import joblib
import torch

from phc.utils.pc_anomaly import SmplToPointCloud, PointCloudMotionClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pkl",
        type=str,
        required=True,
        help="路径：HumanoidIm._write_states_to_file 输出的 pkl 文件",
    )
    parser.add_argument(
        "--clip_key",
        type=str,
        default="0_0",
        help="pkl 中的 clip key（例如 0_0）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="运行设备（cuda 或 cpu）",
    )
    args = parser.parse_args()

    data = joblib.load(args.pkl)
    if args.clip_key not in data:
        raise KeyError(f"clip_key {args.clip_key} not in {list(data.keys())}")

    motion = data[args.clip_key]
    dof_pos = motion["dof_pos"]  # (T, J*3)
    root_states = motion["root_states_seg"]  # (T, >=7)

    root_pos = root_states[:, :3]
    root_rot = root_states[:, 3:7]

    betas_np = motion.get("betas", None)
    if betas_np is None:
        betas = torch.zeros(1, 10)
    else:
        betas = torch.from_numpy(betas_np[:10]).float().unsqueeze(0)

    device = torch.device(args.device)
    smpl_to_pc = SmplToPointCloud(
        smpl_model_path="data/smpl",
        num_betas=betas.shape[1],
        num_points=1024,
        local_coord=True,
        device=device,
    )

    classifier = PointCloudMotionClassifier(
        in_channels=3,
        feat_dim=256,
        num_classes=2,
        temporal_pool="mean",
    ).to(device)

    T = dof_pos.shape[0]
    root_pos_t = root_pos.to(device)
    root_rot_t = root_rot.to(device)
    dof_pos_t = dof_pos.to(device)
    betas_t = betas.to(device).expand(T, -1)

    pc = smpl_to_pc(
        root_pos=root_pos_t,
        root_rot_xyzw=root_rot_t,
        dof_pos=dof_pos_t,
        betas=betas_t,
    )  # (T, N, 3)

    logits = classifier(pc)  # (T, 2)
    p = torch.softmax(logits, dim=-1)

    print("per-frame probs (class 0,1) shape:", p.shape)
    print(p[:5])

    loss = p[:, 0].mean()
    loss.backward()
    print("backward success, grad on dof_pos first row norm:", dof_pos_t.grad if dof_pos_t.requires_grad else "no grad")


if __name__ == "__main__":
    main()

