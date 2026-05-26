# 基于生成模型的自动驾驶异常行人场景生成与评估

本项目面向**自动驾驶感知**中的**异常行人**问题：在物理仿真与人体生成模型驱动下，合成非常规行人姿态与运动，并生成**LiDAR 风格点云**观测，再通过分类与可视化工具对「人形置信度」等指标进行评估与数据集构建。

实现上基于 [PULSE](https://github.com/ZhengyiLuo/PULSE)（ICLR 2024 Spotlight，论文主页：<https://www.zhengyiluo.com/PULSE>）的人形仿真与模仿学习框架；本仓库为**独立研究代码**，目录结构与 PULSE 对齐，便于合并到完整 PULSE 工程中使用。

> 本仓库**不是** PULSE 官方仓库。使用前需自行 clone PULSE，并准备 Isaac Gym、SMPL、PointNet++ 等依赖。

## 项目流程（概要）

1. **场景生成**：在仿真中由策略与人体表征产生非常规行人姿态（SMPL 状态）。
2. **传感器建模**：由 SMPL 网格经射线–网格求交得到 LiDAR 稀疏点云（亦可对照稠密表面点云）。
3. **筛选与落盘**：依据人形分类置信度等指标筛选低置信样本，构建异常行人点云数据集。
4. **评估与可视化**：对保存的 LiDAR npz 进行统计、分类器推理与三维可视化。

## 安装步骤

```bash
git clone https://github.com/ZhengyiLuo/PULSE PULSE
cd PULSE

# 将本仓库合并到 PULSE 根目录（覆盖同名路径）
rsync -av --exclude='.git' /path/to/本仓库/ ./

# 其余依赖见 PULSE 官方文档
```

Windows 下可将本仓库的 `scripts/`、`phc/` 复制到 PULSE 对应目录。

## 本仓库主要路径

| 路径 | 说明 |
|------|------|
| `scripts/smpl_lidar_raymesh_internal/` | 离线 SMPL→LiDAR 射线网格仿真 |
| `scripts/viz_lidar_npz_classifier.py` | LiDAR 数据集可视化 + 分类评估 |
| `scripts/viz_pointcloud_one_frame.py` 等 | 稠密点云与分类器相关工具 |
| `phc/utils/smpl_lidar_sim.py` | 训练侧 LiDAR 仿真 |
| `phc/utils/pc_anomaly.py` | SMPL→点云、PointNet++ 人形判别 |
| `phc/learning/amp_agent.py` | 仿真中点云奖励、数据集与 LiDAR 落盘 |
| `phc/data/cfg/learning/im_z_anom_pc.yaml` | 点云异常与 LiDAR 落盘配置 |

完整文件列表见 `MANIFEST.md`。

## 第三方说明

见 **`ATTRIBUTION.md`**。本项目使用 PULSE 及其文档中涉及的 PHC、SMPL、PointNet++ 等开源组件；SMPL 模型须自行按许可下载。

## 不包含（需自行准备）

- 完整 PULSE 仿真环境（`phc/env/`、Isaac 等）
- `data/smpl/` 下的 SMPL 模型文件
- 训练生成的 `output/` 数据与分类器权重

## 快速验证（离线 LiDAR）

```bash
cd scripts/smpl_lidar_raymesh_internal
python gen_standing_lidar.py
python viz_saved_lidar_npz.py
```
