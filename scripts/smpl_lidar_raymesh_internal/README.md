# SMPL LiDAR Ray-Mesh Simulator

这个包用于把导出的 SMPL 姿态 `.npz` 转换成类似多线激光雷达扫描人体得到的点云。

当前版本的 LiDAR 模拟方式是 **ray-mesh intersection**：先由 SMPL pose/betas 生成真实人体三角网格，再按可调线数、水平角分辨率和距离发射 LiDAR beam，每条 beam 只保留和 SMPL mesh 的最近交点。

## 包内容

```text
.
├── smpl_lidar_scan.py              # 主转换脚本
├── view_lidar_pointcloud.py        # 点云 HTML/Open3D 可视化脚本
├── smpl_random_pc_viz.py           # 原始随机 SMPL 点云可视化脚本，可选
├── requirements.txt                # Python 运行依赖
├── setup_wsl_uv.sh                 # WSL/Linux 一键建环境脚本
└── human2humanoid/
    ├── phc/                        # 最小 PHC/SMPL parser 源码包
    └── data/smpl/SMPL_NEUTRAL.pkl  # SMPL neutral 模型，内部分享使用
```

包里不包含原始 `smpl_*.npz` 输入数据，也不包含转换后的 `lidar_pc/` 结果数据。使用时把自己的 SMPL `.npz` 文件放到包根目录，或通过命令行传入输入目录。

## 环境部署

在 WSL/Linux 中进入包目录：

```bash
cd smpl_lidar_raymesh_internal
```

推荐直接运行：

```bash
bash setup_wsl_uv.sh
```

这个脚本会：

- 安装 `uv`，如果系统里还没有；
- 创建 `.venv`；
- 安装 CPU 版 PyTorch、`smplx`、`scipy`、`chumpy`；
- 以 editable 模式安装本包内的 `human2humanoid/phc`。

手动部署命令等价于：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install pip setuptools
uv pip install -r requirements.txt --no-build-isolation
uv pip install -e human2humanoid/phc
```

## 转换 SMPL 到 LiDAR 点云

处理当前目录中的全部 `.npz`：

```bash
source .venv/bin/activate
python smpl_lidar_scan.py . -o lidar_pc --output-format both
```

处理单个文件：

```bash
python smpl_lidar_scan.py smpl_ep000001_h025_000000.npz -o lidar_pc --output-format both
```

常用参数：

```bash
python smpl_lidar_scan.py . -o lidar_pc \
  --lines 128 \
  --distance 10 \
  --horizontal-res-deg 0.2 \
  --fov-up-deg 2.0 \
  --fov-down-deg -24.9 \
  --range-noise-std 0.0 \
  --dropout 0.0 \
  --output-format both
```

参数说明：

- `--lines`：LiDAR 垂直线数，默认 `128`，可改成 `64/32/16` 或其他正整数。
- `--distance`：雷达到人体模型原点的距离，单位米，默认 `10`。
- `--horizontal-res-deg`：水平角分辨率，默认 `0.2` 度。
- `--fov-up-deg / --fov-down-deg`：垂直视场角范围，默认近似多线机械 LiDAR 的人体扫描配置。
- `--range-noise-std`：距离高斯噪声标准差，单位米。
- `--dropout`：随机丢点概率。
- `--output-format`：`npz`、`ply` 或 `both`。

输出 `.npz` 会保留输入里的原始 SMPL 字段，并新增：

- `lidar_points` / `points` / `xyz`：点云坐标，shape 为 `[N, 3]`。
- `lidar_range`：每个点的距离。
- `lidar_ring`：对应 LiDAR 线束编号。
- `lidar_azimuth_deg`：水平角。
- `lidar_intensity`：简单距离衰减强度。
- `lidar_face_index`：命中的 SMPL 三角面编号。
- `lidar_num_lines`：本次模拟的线数。
- `lidar_method`：当前为 `ray_mesh_intersection`。

## 可视化

生成 HTML 预览：

```bash
python view_lidar_pointcloud.py lidar_pc/smpl_ep000001_h025_000000_lidar.npz --mode html
```

查看目录中的第 4 个点云：

```bash
python view_lidar_pointcloud.py lidar_pc --index 3 --mode html
```

如果额外安装了 Open3D，也可以用交互窗口：

```bash
uv pip install open3d
python view_lidar_pointcloud.py lidar_pc/smpl_ep000001_h025_000000_lidar.npz --mode open3d
```

## 输入数据格式

每个输入 `.npz` 至少需要包含：

- `root_pos`，shape `[1, 3]`
- `root_rot_xyzw`，shape `[1, 4]`
- `dof_pos`，shape `[1, 69]`
- `smpl_betas`，shape `[1, >=10]`

这与原始 `smpl_random_pc_viz.py` 的输入约定保持一致。

## 内部分享说明

此包已经包含 `SMPL_NEUTRAL.pkl`，仅用于内部研究分享。若要公开发布，请移除该模型文件，并让使用者从 SMPL 官方渠道自行下载模型后放到：

```text
human2humanoid/data/smpl/SMPL_NEUTRAL.pkl
```

