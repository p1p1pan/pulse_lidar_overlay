#!/usr/bin/env python3
"""
用「本地 ModelNet40 数据（含 person 类）」训练「人 vs 非人」二分类器，
保存的 checkpoint 与 PULSE 里 PointCloudMotionClassifier 结构完全一致，可直接用 pc_classifier_weights 加载。

- 训练数据从哪来：Pointnet 目录下 data/modelnet40_normal_resampled/，
  其中 person/*.txt 为“人”点云，其他类别（airplane, chair, ...）为“非人”。脚本会按 modelnet40_train.txt / test.txt 划分。
- 模型结构：与 phc 里 PointCloudMotionClassifier(pointnet2 backbone + head) 一致，保存 state_dict 后即可在 PULSE 中加载。
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 项目根目录
PULSE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PULSE_ROOT)


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    if m > 1e-8:
        pc = pc / m
    return pc.astype(np.float32)


class PersonVsRestDataset(Dataset):
    """从 ModelNet40 格式数据中读点云，person 类标为 1，其余标为 0。"""

    def __init__(self, root, split="train", num_points=1024, person_class_name="person"):
        self.root = root
        self.num_points = num_points
        self.person_class_name = person_class_name

        shape_names_path = os.path.join(root, "modelnet40_shape_names.txt")
        self.cat = [line.rstrip() for line in open(shape_names_path)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.person_idx = self.classes.get(person_class_name, 24)

        list_file = os.path.join(root, f"modelnet40_{split}.txt")
        shape_ids = [line.rstrip() for line in open(list_file)]
        shape_names = ["_".join(x.split("_")[0:-1]) for x in shape_ids]
        self.datapath = []
        for i in range(len(shape_ids)):
            path = os.path.join(self.root, shape_names[i], shape_ids[i] + ".txt")
            if os.path.isfile(path):
                self.datapath.append((shape_names[i], path))
        print(f"[PersonVsRest] {split}: {len(self.datapath)} samples, person_idx={self.person_idx}")

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
        points = point_set[:, 0:3]  # 只用 XYZ，与 PULSE 一致
        label = 1 if class_name == self.person_class_name else 0
        return torch.from_numpy(points), torch.tensor(label, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser(description="Train person vs non-person point cloud classifier for PULSE")
    parser.add_argument("--data_root", type=str, default=None, help="ModelNet40 数据根目录，默认 Pointnet 下 data/modelnet40_normal_resampled")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8, help="显存不足可改为 4 或 2，或加 --device cpu 用 CPU 训练")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--out", type=str, default=None, help="保存路径，默认 output/pc_classifier/person_vs_rest.pth")
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0 或 cpu（显存不够用 cpu）")
    args = parser.parse_args()

    if args.data_root is None:
        args.data_root = os.path.join(PULSE_ROOT, "Pointnet_Pointnet2_pytorch-master", "data", "modelnet40_normal_resampled")
    if not os.path.isdir(args.data_root):
        print(f"数据目录不存在: {args.data_root}")
        print("请指定 --data_root 或保证 Pointnet 下已有 data/modelnet40_normal_resampled（含 person 文件夹）")
        sys.exit(1)

    if args.out is None:
        args.out = os.path.join(PULSE_ROOT, "output", "pc_classifier", "person_vs_rest.pth")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    from phc.utils.pc_anomaly import build_pc_backbone, PointCloudMotionClassifier

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    backbone = build_pc_backbone(backbone_type="pointnet2", feat_dim=256, in_channels=3)
    classifier = PointCloudMotionClassifier(
        backbone=backbone,
        in_channels=3,
        feat_dim=1024,
        num_classes=2,
        temporal_pool="mean",
    ).to(device)

    train_set = PersonVsRestDataset(args.data_root, split="train", num_points=args.num_points)
    test_set = PersonVsRestDataset(args.data_root, split="test", num_points=args.num_points)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    opt = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    best_test_acc = 0.0
    out_dir = os.path.dirname(args.out)
    latest_path = os.path.join(out_dir, "person_vs_rest_latest.pth")

    for epoch in range(args.epochs):
        classifier.train()
        total, correct = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]", leave=False)
        for points, labels in pbar:
            points, labels = points.to(device), labels.to(device)
            opt.zero_grad()
            logits = classifier(points)
            loss = ce(logits, labels)
            loss.backward()
            opt.step()
            pred = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            pbar.set_postfix(acc=f"{correct/total:.3f}", loss=f"{loss.item():.3f}")
        train_acc = correct / total

        classifier.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for points, labels in tqdm(test_loader, desc=f"Epoch {epoch+1} [test]", leave=False):
                points, labels = points.to(device), labels.to(device)
                logits = classifier(points)
                pred = logits.argmax(dim=1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        test_acc = correct / total
        print(f"Epoch {epoch+1}/{args.epochs}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

        # 每轮都保存「最新」checkpoint，中途关机也不会丢
        state = {"state_dict": classifier.state_dict(), "epoch": epoch + 1, "test_acc": test_acc}
        torch.save(state, latest_path)

        # 若当前 test 准确率最高，则覆盖「最佳」权重（供 PULSE 用）
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save({"state_dict": classifier.state_dict()}, args.out)
            print(f"  -> 更新最佳权重 {args.out} (test_acc={test_acc:.4f})")

    print(f"训练结束。最佳权重: {args.out}，可在 PULSE 配置中设置: pc_classifier_weights: {args.out}")


if __name__ == "__main__":
    main()
