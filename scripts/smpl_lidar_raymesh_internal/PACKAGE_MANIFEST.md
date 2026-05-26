# Package Manifest

This internal package includes all source/model assets needed to run SMPL to LiDAR conversion after Python dependencies are installed.

Included:

- `smpl_lidar_scan.py`
- `gen_standing_lidar.py`（零姿态 SMPL → `lidar_pc/smpl_standing_default_lidar.npz`）
- `viz_saved_lidar_npz.py`（Open3D 查看已保存的 `*_lidar.npz`，无需分类器）
- `viz_lidar_from_npz.py`（`pc_smpl_dataset` 随机 SMPL npz → LiDAR，Open3D，不写盘）
- `pc_smpl_dataset/viz_smpl_npz_in_this_folder.py`（同目录 npz：主工程 1024 + LiDAR 对比）
- `requirements.txt`
- `setup_wsl_uv.sh`
- `human2humanoid/phc/`
- `human2humanoid/data/smpl/SMPL_NEUTRAL.pkl`

Intentionally excluded:

- Raw SMPL sample files: `smpl_*.npz`
- Generated point-cloud result folders: `lidar_pc/`, `lidar_pc_*/`
- Virtual environments: `.venv/`
- Python caches: `__pycache__/`
- The full upstream `human2humanoid` training repository outside the minimal `phc` package

