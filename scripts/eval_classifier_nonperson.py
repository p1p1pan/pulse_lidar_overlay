#!/usr/bin/env python3
"""
评估点云分类器在「非 person」物体上的表现：是否会错误地识别成 person。

用法:
  python scripts/eval_classifier_nonperson.py
  python scripts/eval_classifier_nonperson.py --classifier output/pc_classifier/person_vs_rest.pth --data_root Pointnet_Pointnet2_pytorch-master/data/modelnet40_normal_resampled

输出：每类非 person 物体的 p_human 均值、最高值、误判率（p_human>0.5 的比例）
"""
import os
import sys
import argparse
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm

PULSE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PULSE_ROOT)


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    if m > 1e-8:
        pc = pc / m
    return pc.astype(np.float32)


class PersonVsRestDataset:
    """复用 train_person_pointnet 的数据读取逻辑，可筛选仅非 person 样本。"""

    def __init__(self, root, split="test", num_points=1024, person_class_name="person", nonperson_only=True):
        self.root = root
        self.num_points = num_points
        self.person_class_name = person_class_name

        shape_names_path = os.path.join(root, "modelnet40_shape_names.txt")
        self.cat = [line.rstrip() for line in open(shape_names_path)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        list_file = os.path.join(root, f"modelnet40_{split}.txt")
        shape_ids = [line.rstrip() for line in open(list_file)]
        shape_names = ["_".join(x.split("_")[0:-1]) for x in shape_ids]
        self.datapath = []
        for i in range(len(shape_ids)):
            class_name = shape_names[i]
            if nonperson_only and class_name == person_class_name:
                continue
            path = os.path.join(self.root, shape_names[i], shape_ids[i] + ".txt")
            if os.path.isfile(path):
                self.datapath.append((class_name, path))
        print(f"[eval nonperson] {split}: {len(self.datapath)} non-person samples from {len(set(x[0] for x in self.datapath))} classes")

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        class_name, path = self.datapath[index]
        point_set = np.loadtxt(path, delimiter=",").astype(np.float32)
        if point_set.shape[0] >= self.num_points:
            choice = np.random.choice(point_set.shape[0], self.num_points, replace=False)
        else:
            choice = np.random.choice(point_set.shape[0], self.num_points, replace=True)
        point_set = point_set[choice, :]
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        points = point_set[:, 0:3]
        return torch.from_numpy(points), class_name


def main():
    parser = argparse.ArgumentParser(description="Evaluate classifier on non-person objects")
    parser.add_argument("--classifier", type=str, default=None, help="person_vs_rest.pth 路径")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="cuda:0 或 cpu；若遇到 cuDNN 报错可改用 --device cpu")
    args = parser.parse_args()

    if args.classifier is None:
        args.classifier = os.path.join(PULSE_ROOT, "output", "pc_classifier", "person_vs_rest.pth")
    if args.data_root is None:
        args.data_root = os.path.join(PULSE_ROOT, "Pointnet_Pointnet2_pytorch-master", "data", "modelnet40_normal_resampled")

    if not os.path.isfile(args.classifier):
        print(f"未找到分类器: {args.classifier}")
        sys.exit(1)
    if not os.path.isdir(args.data_root):
        print(f"数据目录不存在: {args.data_root}")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False  # 避免 "Unable to find a valid cuDNN algorithm" 错误
    dataset = PersonVsRestDataset(args.data_root, split="test", num_points=args.num_points, nonperson_only=True)
    if len(dataset) == 0:
        print("无非 person 样本")
        sys.exit(0)

    from phc.utils.pc_anomaly import build_pc_backbone, PointCloudMotionClassifier
    backbone = build_pc_backbone(backbone_type="pointnet2", feat_dim=256, in_channels=3)
    classifier = PointCloudMotionClassifier(
        backbone=backbone, in_channels=3, feat_dim=1024, num_classes=2, temporal_pool="mean",
    ).to(device)
    state = torch.load(args.classifier, map_location=device)
    sd = state.get("state_dict") or state
    classifier.load_state_dict(sd, strict=False)
    classifier.eval()

    # 按类收集 p_human
    stats = defaultdict(lambda: {"p_human": [], "count": 0})
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="eval nonperson"):
            points, class_name = dataset[i]
            points = points.unsqueeze(0).to(device)
            logits = classifier(points)
            probs = torch.softmax(logits, dim=-1)
            # 训练时 label=1 为 person、0 为 non-person，故 logits[:,1]=person
            p_human = probs[0, 1].item()
            stats[class_name]["p_human"].append(p_human)
            stats[class_name]["count"] += 1

    # 汇总
    print("\n========== 非 person 物体上的 p_human (人形概率) ==========")
    print("类别           样本数   mean     max     误判率(p>0.5)")
    print("-" * 55)
    all_ph = []
    total_fp = 0
    total_n = 0
    rows = []
    for cls in sorted(stats.keys()):
        ph = stats[cls]["p_human"]
        n = stats[cls]["count"]
        mean_ph = np.mean(ph)
        max_ph = np.max(ph)
        fp = sum(1 for x in ph if x > 0.5)
        fp_rate = fp / n if n else 0
        all_ph.extend(ph)
        total_fp += fp
        total_n += n
        rows.append((cls, n, mean_ph, max_ph, fp_rate, fp))

    for cls, n, mean_ph, max_ph, fp_rate, fp in sorted(rows, key=lambda x: -x[2]):
        print(f"{cls:15s}  {n:5d}   {mean_ph:.4f}   {max_ph:.4f}   {fp_rate*100:.1f}% ({fp}/{n})")

    print("-" * 55)
    print(f"{'总体':15s}  {total_n:5d}   {np.mean(all_ph):.4f}   {np.max(all_ph):.4f}   {total_fp/total_n*100:.1f}% ({total_fp}/{total_n})")
    print("\n若误判率较高，说明分类器易将非人物体识别为人形，训练或阈值需调整。")


if __name__ == "__main__":
    main()
