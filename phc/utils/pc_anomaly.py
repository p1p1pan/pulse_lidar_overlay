import os
import sys

import numpy as np
import torch
import torch.nn as nn
from typing import Optional

try:
    import smplx  # type: ignore
except ImportError:
    smplx = None

# PointNet++ 特征维度（与 Pointnet_Pointnet2_pytorch-master 中 PointNet2FeatureExtractor 一致）
POINTNET2_FEAT_DIM = 1024
# ModelNet40 中 "person" 类别索引（modelnet40_shape_names.txt 第 25 行，0-based=24）
MODELNET40_PERSON_CLASS_IDX = 24


def pc_normalize_torch(pc: torch.Tensor) -> torch.Tensor:
    """
    与 ModelNet/PersonVsRest 训练时一致的归一化：去中心 + max 范数缩放到单位球。
    pc: (B, N, 3)，输出同 shape，可微。
    """
    centroid = pc.mean(dim=1, keepdim=True)
    pc = pc - centroid
    m = (pc ** 2).sum(dim=-1).sqrt().max(dim=1, keepdim=True)[0].clamp(min=1e-8).unsqueeze(-1)
    return pc / m


class SmplToPointCloud(nn.Module):
    """
    可微的 SMPL → 点云转换模块。

    约定输入为单帧 SMPL 状态：
    - root_pos:  (B, 3)      世界坐标系下根平移
    - root_rot:  (B, 4)      根四元数，格式为 (x, y, z, w)（Isaac Gym 风格）
    - dof_pos:   (B, J*3)    关节轴角（不含根），与环境中的 dof_pos 一致
    - betas:     (B, nb)     SMPL 形状参数（可以共享或全 0）

    输出：
    - pc:        (B, num_points, 3)
    """

    def __init__(
        self,
        smpl_model_path: str = "data/smpl",
        num_betas: int = 10,
        num_points: int = 1024,
        local_coord: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        if smplx is None:
            raise ImportError(
                "smplx is required for SmplToPointCloud but is not installed. "
                "Please install smplx or disable use_pc_anomaly_loss."
            )

        self.num_betas = num_betas
        self.num_points = num_points
        self.local_coord = local_coord

        self.smpl = smplx.SMPL(
            model_path=smpl_model_path,
            gender="neutral",
            num_betas=num_betas,
            ext="pkl",
        )

        if device is not None:
            self.to(device)

        # 预先固定一个顶点子集用于点云采样（可微 gather）
        # 直接使用 v_template 的顶点数即可，避免在 __init__ 中额外做一次 SMPL forward
        with torch.no_grad():
            num_verts = int(self.smpl.v_template.shape[0])

        if num_points > num_verts:
            raise ValueError(
                f"num_points={num_points} > num_verts={num_verts}; "
                "please reduce num_points."
            )

        # 确定性均匀采样，避免每次 run 得到不同点云（影响 p_human 可复现性）
        indices = torch.linspace(0, num_verts - 1, num_points, dtype=torch.long)
        self.register_buffer("vertex_indices", indices, persistent=False)

        _faces = getattr(self.smpl, "faces", None)
        if _faces is None:
            raise RuntimeError("smplx SMPL model has no faces attribute.")
        if isinstance(_faces, torch.Tensor):
            faces_t = _faces.long().contiguous().cpu()
        else:
            # smplx 常为 numpy.uint32，torch.as_tensor(..., long) 会报错，先落到 int64
            arr = np.asarray(_faces)
            if arr.dtype != np.int64:
                arr = arr.astype(np.int64, copy=False)
            faces_t = torch.from_numpy(np.ascontiguousarray(arr)).long()
        self.register_buffer("_smpl_faces", faces_t, persistent=False)

    @staticmethod
    def _quat_xyzw_to_angle_axis_wxyz(quat_xyzw: torch.Tensor) -> torch.Tensor:
        """
        将 Isaac Gym 风格 (x, y, z, w) 四元数转换为轴角 (angle * axis)。
        """
        # 重新排列为 (w, x, y, z)
        q = quat_xyzw[..., [3, 0, 1, 2]]
        q = torch.nn.functional.normalize(q, p=2, dim=-1)
        w = q[..., 0]
        xyz = q[..., 1:]

        # 避免数值问题
        w_clamped = torch.clamp(w, -1.0, 1.0)
        angle = 2.0 * torch.acos(w_clamped)

        # sin(theta/2) 可能接近 0，需要保护
        sin_half = torch.sqrt(torch.clamp(1.0 - w_clamped * w_clamped, min=1e-8))
        axis = xyz / sin_half.unsqueeze(-1)

        angle_axis = angle.unsqueeze(-1) * axis
        return angle_axis

    def forward(
        self,
        root_pos: torch.Tensor,
        root_rot_xyzw: torch.Tensor,
        dof_pos: torch.Tensor,
        betas: torch.Tensor,
    ) -> torch.Tensor:
        """
        root_pos:      (B, 3)
        root_rot_xyzw: (B, 4)  (x, y, z, w)
        dof_pos:       (B, J*3)  轴角
        betas:         (B, num_betas)

        返回:
        pc:            (B, num_points, 3)
        """
        B = root_pos.shape[0]
        device = root_pos.device

        global_orient = self._quat_xyzw_to_angle_axis_wxyz(root_rot_xyzw)
        body_pose = dof_pos  # 已经是轴角 (J*3)

        if betas.shape[1] != self.num_betas:
            raise ValueError(
                f"betas dim mismatch: got {betas.shape[1]}, expected {self.num_betas}"
            )

        smpl_out = self.smpl(
            global_orient=global_orient.to(device),
            body_pose=body_pose.to(device),
            betas=betas.to(device),
            transl=root_pos.to(device),
        )
        verts = smpl_out.vertices  # (B, V, 3)

        # 固定子集抽样
        verts_sampled = verts[:, self.vertex_indices, :]  # (B, N, 3)

        if self.local_coord:
            # 以根平移为原点做局部坐标
            verts_sampled = verts_sampled - root_pos.unsqueeze(1)

        return verts_sampled

    def mesh_vertices_world(
        self,
        root_pos: torch.Tensor,
        root_rot_xyzw: torch.Tensor,
        dof_pos: torch.Tensor,
        betas: torch.Tensor,
    ) -> torch.Tensor:
        """与 forward 中 SMPL 前向一致：世界坐标系下完整网格顶点 (B, V, 3)，供 LiDAR 射线求交。"""
        device = root_pos.device
        global_orient = self._quat_xyzw_to_angle_axis_wxyz(root_rot_xyzw)
        if betas.shape[1] != self.num_betas:
            raise ValueError(
                f"betas dim mismatch: got {betas.shape[1]}, expected {self.num_betas}"
            )
        smpl_out = self.smpl(
            global_orient=global_orient.to(device),
            body_pose=dof_pos.to(device),
            betas=betas.to(device),
            transl=root_pos.to(device),
        )
        return smpl_out.vertices

    def mesh_faces_numpy(self) -> np.ndarray:
        """三角面索引 (F, 3)，long。"""
        return self._smpl_faces.detach().cpu().numpy().astype(np.int64)


class SimplePointNetBackbone(nn.Module):
    """轻量 PointNet 风格骨干，无 PointNet++ 时可用。"""

    def __init__(self, in_channels: int = 3, feat_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, feat_dim, 1),
            nn.ReLU(inplace=True),
        )
        self.feat_dim = feat_dim

    def forward(self, pc: torch.Tensor) -> torch.Tensor:
        """
        pc: (B, N, 3)
        返回: (B, feat_dim)
        """
        x = pc.transpose(1, 2)  # (B, 3, N)
        x = self.mlp(x)         # (B, feat_dim, N)
        x = torch.max(x, dim=2).values  # (B, feat_dim)
        return x


def _get_pointnet2_extractor_class():
    """延迟导入 Pointnet 仓库中的 PointNet2FeatureExtractor，避免未安装时报错。"""
    pointnet_dir = os.environ.get(
        "PULSE_POINTNET_DIR",
        os.path.join(os.path.dirname(__file__), "..", "..", "Pointnet_Pointnet2_pytorch-master"),
    )
    pointnet_dir = os.path.abspath(pointnet_dir)
    if pointnet_dir not in sys.path:
        sys.path.insert(0, pointnet_dir)
    try:
        from models.pointnet2_feature_extractor import PointNet2FeatureExtractor
        return PointNet2FeatureExtractor
    except Exception as e:
        raise ImportError(
            f"无法导入 PointNet2FeatureExtractor，请确认 Pointnet 代码在 {pointnet_dir}，且该目录下存在 models/pointnet2_feature_extractor.py。错误: {e}"
        ) from e


def _get_pointnet2_cls_msg_class():
    """延迟导入 Pointnet 仓库中的 pointnet2_cls_msg.get_model（40 类 ModelNet 分类器）。"""
    pointnet_dir = os.environ.get(
        "PULSE_POINTNET_DIR",
        os.path.join(os.path.dirname(__file__), "..", "..", "Pointnet_Pointnet2_pytorch-master"),
    )
    pointnet_dir = os.path.abspath(pointnet_dir)
    if pointnet_dir not in sys.path:
        sys.path.insert(0, pointnet_dir)
    try:
        from models.pointnet2_cls_msg import get_model
        return get_model
    except Exception as e:
        raise ImportError(
            f"无法导入 pointnet2_cls_msg，请确认 Pointnet 代码在 {pointnet_dir}。错误: {e}"
        ) from e


class PointNet2ModelNet40PersonWrapper(nn.Module):
    """
    将 Pointnet 仓库中训练好的 40 类 ModelNet 分类器（含 person）封装为
    “人形 / 非人形”二类输出，与 PointCloudMotionClassifier 接口一致：
    输入 (B, N, 3)，输出 (B, 2) logits，softmax 后 [:, 0] 即为 p_human（person 类概率）。
    """

    def __init__(
        self,
        num_class: int = 40,
        person_class_idx: int = MODELNET40_PERSON_CLASS_IDX,
        normal_channel: bool = False,
        weights_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.num_classes = 2
        self.person_class_idx = person_class_idx
        get_model = _get_pointnet2_cls_msg_class()
        self.model = get_model(num_class=num_class, normal_channel=normal_channel)
        if weights_path:
            state = torch.load(weights_path, map_location="cpu")
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            self.model.load_state_dict(state, strict=True)

    def forward(self, pc: torch.Tensor) -> torch.Tensor:
        """pc: (B, N, 3) -> logits (B, 2)，对应 [人形, 非人形]。"""
        x = pc.permute(0, 2, 1)  # (B, 3, N)
        log_prob_40, _ = self.model(x)  # (B, 40) log_softmax
        probs_40 = torch.exp(log_prob_40.clamp(min=-50.0))
        p_person = probs_40[:, self.person_class_idx].clamp(1e-6, 1.0 - 1e-6)
        p_non = (1.0 - p_person).clamp(1e-6, 1.0 - 1e-6)
        logits = torch.stack([torch.log(p_person), torch.log(p_non)], dim=-1)
        return logits


class PointNet2Backbone(nn.Module):
    """Pointnet 仓库 PointNet2FeatureExtractor 封装：输入 (B, N, 3)，输出 (B, 1024)。"""

    def __init__(
        self,
        use_normals: bool = False,
        pretrained_extractor_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        PointNet2FeatureExtractor = _get_pointnet2_extractor_class()
        self.extractor = PointNet2FeatureExtractor(normal_channel=use_normals)
        self.feat_dim = POINTNET2_FEAT_DIM

        if pretrained_extractor_path:
            self._load_pretrained_extractor(pretrained_extractor_path)

    def _load_pretrained_extractor(self, path: str) -> None:
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        # 兼容 PointNetTransformer 保存的 key：feature_extractor.sa1 -> extractor.sa1
        new_state = {}
        for k, v in state.items():
            if k.startswith("feature_extractor."):
                new_state["extractor." + k[len("feature_extractor.") :]] = v
            elif k.startswith("sa1.") or k.startswith("sa2.") or k.startswith("sa3."):
                new_state["extractor." + k] = v
        if new_state:
            self.load_state_dict(new_state, strict=False)

    def forward(self, pc: torch.Tensor) -> torch.Tensor:
        """
        pc: (B, N, 3)
        返回: (B, 1024)
        """
        x = pc.permute(0, 2, 1)  # (B, 3, N)
        return self.extractor(x)


def build_pc_backbone(
    backbone_type: str = "pointnet2",
    feat_dim: int = 256,
    in_channels: int = 3,
    pretrained_extractor_path: Optional[str] = None,
) -> nn.Module:
    """构建点云 backbone：simple 为轻量占位，pointnet2 为 PointNet++（1024 维）。"""
    if backbone_type == "pointnet2":
        return PointNet2Backbone(
            use_normals=False,
            pretrained_extractor_path=pretrained_extractor_path,
        )
    return SimplePointNetBackbone(in_channels=in_channels, feat_dim=feat_dim)


class PointCloudMotionClassifier(nn.Module):
    """单帧/序列点云 → 行人(人类)/非行人 分类。输入 (B, N, 3) 或 (B, T, N, 3)，输出 (B, num_classes) logits。"""

    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        in_channels: int = 3,
        feat_dim: int = 256,
        num_classes: int = 2,
        temporal_pool: str = "mean",
    ) -> None:
        super().__init__()
        if backbone is None:
            backbone = SimplePointNetBackbone(in_channels=in_channels, feat_dim=feat_dim)
        if hasattr(backbone, "feat_dim"):
            feat_dim = int(backbone.feat_dim)

        self.backbone = backbone
        self.num_classes = num_classes
        self.temporal_pool = temporal_pool
        self._feat_dim = feat_dim

        self.head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, pc_seq: torch.Tensor) -> torch.Tensor:
        """单帧 (B, N, 3) 或序列 (B, T, N, 3) → logits (B, num_classes)。"""
        if pc_seq.dim() == 3:
            # 单帧：直接 backbone + head
            feats = self.backbone(pc_seq)
            return self.head(feats)

        B, T, N, C = pc_seq.shape
        pcs = pc_seq.reshape(B * T, N, C)
        feats = self.backbone(pcs)
        feats = feats.view(B, T, -1)
        seq_feat = feats.mean(dim=1) if self.temporal_pool == "mean" else feats.max(dim=1).values
        return self.head(seq_feat)

