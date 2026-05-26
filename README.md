# PULSE 扩展：LiDAR 仿真点云与低人形置信数据集

本仓库为 **覆盖补丁包**，目录层级与官方 [PULSE](https://github.com/ZhengyiLuo/PULSE) 主仓库一致（ICLR 2024 Spotlight，论文主页：<https://www.zhengyiluo.com/PULSE>）。用法：先 clone PULSE，再将**本仓库根目录下的** `scripts/`、`phc/` 等**合并到 PULSE 根目录**（覆盖同名路径）。

> 本 Git 仓库仅包含扩展部分，**不是**完整 PULSE。上传 GitHub 时只推送本文件夹即可。

## 安装步骤

```bash
git clone <官方 PULSE 仓库> PULSE
cd PULSE

# 将本仓库解压或 clone 到临时目录后，覆盖到 PULSE 根
rsync -av --exclude='.git' /path/to/pulse_lidar_overlay/ ./

# 仍需官方 PULSE 的其余部分：Isaac Gym、data/smpl（自行从 SMPL 获取）、
# Pointnet_Pointnet2_pytorch-master 等，见官方文档。
```

Windows 下可用资源管理器将 `scripts/`、`phc/` 复制到 PULSE 对应文件夹。

## 本包包含的覆盖路径

| 路径 | 说明 |
|------|------|
| `scripts/smpl_lidar_raymesh_internal/` | 离线 SMPL→LiDAR 射线网格仿真包 |
| `scripts/viz_lidar_npz_classifier.py` | 已保存 LiDAR npz 可视化 + 分类 |
| `scripts/viz_pointcloud_one_frame.py` 等 | 稠密点云 / 分类器相关脚本 |
| `phc/utils/smpl_lidar_sim.py` | 训练侧 LiDAR 仿真 |
| `phc/utils/pc_anomaly.py` | SMPL→点云、PointNet++ 分类接口 |
| `phc/learning/amp_agent.py` | 含点云奖励、数据集与 LiDAR 落盘逻辑 |
| `phc/data/cfg/learning/im_z_anom_pc.yaml` | 点云异常与 LiDAR 落盘配置 |

详细文件列表见 `MANIFEST.md`。

## 第三方说明

见 **`ATTRIBUTION.md`**。本补丁依赖官方 PULSE 及其文档中提到的 PHC、SMPL、PointNet++ 等组件；SMPL 模型须自行按许可下载。

## 不包含（需自行准备）

- 完整 PULSE 仿真环境（`phc/env/`、Isaac 等）
- `data/smpl/` 下的 SMPL 模型文件
- 训练生成的 `output/` 数据
- `scripts/smpl_lidar_raymesh_internal` 包内的 `SMPL_NEUTRAL.pkl`（若使用包内独立环境，请按该包 README 放置模型）

## 快速验证（离线 LiDAR）

```bash
cd scripts/smpl_lidar_raymesh_internal
python gen_standing_lidar.py
python viz_saved_lidar_npz.py
```
