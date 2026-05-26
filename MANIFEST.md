# 覆盖文件清单

合并到 PULSE 根目录时，以下路径应被本包同名文件覆盖或新增。

## scripts/

- `smpl_lidar_raymesh_internal/`（整目录，源码与 README；不含示例 npz / SMPL pkl）
- `viz_lidar_npz_classifier.py`
- `viz_pointcloud_one_frame.py`
- `viz_pc_dataset_random.py`
- `train_person_pointnet.py`
- `eval_classifier_nonperson.py`
- `debug_pc_anomaly.py`

## phc/

- `utils/smpl_lidar_sim.py`
- `utils/pc_anomaly.py`
- `learning/amp_agent.py`
- `data/cfg/learning/im_z_anom_pc.yaml`

## examples/（可选，不参与覆盖 PULSE）

- `train_launch_abc.txt` — 训练命令示例
