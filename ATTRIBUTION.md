# 第三方说明

本仓库为在开源项目 **PULSE** 之上提供的扩展补丁，目录结构与 PULSE 主仓库对齐。下列上游项目版权归原作者所有；使用本补丁前请先取得完整 PULSE 工程，并遵守各项目许可证与引用要求。

## 1. PULSE

- 仓库：<https://github.com/ZhengyiLuo/PULSE>
- 论文：*Universal Humanoid Motion Representations for Physics-Based Control*（ICLR 2024 Spotlight）
- 说明：仿真、模仿学习与 `phc/` 主体代码来自该仓库；本补丁仅覆盖其中列出的部分路径。

## 2. PHC / human2humanoid（包内精简代码）

- 相关仓库：<https://github.com/ZhengyiLuo/PHC>（PULSE 官方 README 中亦有说明）
- 路径：`scripts/smpl_lidar_raymesh_internal/human2humanoid/phc/`
- 说明：离线 SMPL 解析等最小封装，非完整 PHC 或 human2humanoid 仓库。

## 3. SMPL / smplx

- 说明：人体参数化模型；`phc/utils/pc_anomaly.py` 使用 smplx。模型文件（如 `SMPL_NEUTRAL.pkl`）须按 SMPL 官网许可自行下载，本仓库不附带模型权重。

## 4. PointNet++

- 说明：点云特征与分类依赖 PULSE 工程中常用的 `Pointnet_Pointnet2_pytorch-master`，请按 PULSE 官方文档配置环境。

## 本补丁新增内容（技术摘要）

在遵守上述上游许可的前提下，本仓库主要增加或替换：

- LiDAR 射线–网格仿真（`smpl_lidar_scan.py`、`phc/utils/smpl_lidar_sim.py` 等）
- 点云相关训练配置与 `amp_agent.py` 中的数据集落盘逻辑
- 可视化与离线工具脚本（`scripts/viz_*`、`gen_standing_lidar.py` 等）

公开使用或撰写论文时，请按学术规范引用 PULSE、SMPL 等原始文献与仓库，并说明所使用的外部补丁范围。
