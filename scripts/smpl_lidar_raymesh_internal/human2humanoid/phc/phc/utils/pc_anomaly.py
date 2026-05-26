"""SMPL pose to point-cloud utilities.

This module provides the small interface used by the exported pc_smpl_dataset
scripts. It intentionally stays lightweight: it depends only on torch, smplx,
and the local PHC SMPL parser.
"""

from __future__ import annotations

import math
import inspect
import warnings

import numpy as np
import torch

# The original SMPL pickle often imports chumpy. chumpy is unmaintained and
# still expects Python/Numpy aliases removed in newer runtimes.
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
for _name, _value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "str": str,
    "unicode": str,
}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)
warnings.filterwarnings(
    "ignore",
    message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
    category=Warning,
    module=r"smplx\.body_models",
)

from phc.smpllib.smpl_parser import SMPL_Parser


def _quat_xyzw_to_axis_angle(quat_xyzw: torch.Tensor) -> torch.Tensor:
    quat = quat_xyzw.float()
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    xyz = quat[..., :3]
    w = quat[..., 3:4].clamp(-1.0, 1.0)
    sin_half = xyz.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, w)
    axis = xyz / sin_half.clamp_min(1e-8)
    return axis * angle


def _sample_mesh_surface(vertices: torch.Tensor, faces: torch.Tensor, num_points: int) -> torch.Tensor:
    batch_size = vertices.shape[0]
    faces = faces.to(device=vertices.device, dtype=torch.long)
    tri = vertices[:, faces]  # [B, F, 3, 3]

    edge1 = tri[:, :, 1] - tri[:, :, 0]
    edge2 = tri[:, :, 2] - tri[:, :, 0]
    areas = torch.linalg.cross(edge1, edge2, dim=-1).norm(dim=-1).clamp_min(1e-12)
    face_idx = torch.multinomial(areas, num_points, replacement=True)

    batch_idx = torch.arange(batch_size, device=vertices.device)[:, None]
    chosen = tri[batch_idx, face_idx]
    u = torch.rand(batch_size, num_points, 1, device=vertices.device, dtype=vertices.dtype)
    v = torch.rand(batch_size, num_points, 1, device=vertices.device, dtype=vertices.dtype)
    flip = (u + v) > 1.0
    u = torch.where(flip, 1.0 - u, u)
    v = torch.where(flip, 1.0 - v, v)
    return chosen[:, :, 0] + u * (chosen[:, :, 1] - chosen[:, :, 0]) + v * (chosen[:, :, 2] - chosen[:, :, 0])


class SmplToPointCloud(torch.nn.Module):
    """Convert SMPL pose tensors to sampled body-surface point clouds.

    Parameters match the interface used by smpl_random_pc_viz.py:
    - root_pos: [B, 3]
    - root_rot_xyzw: [B, 4] quaternion in xyzw order
    - dof_pos: [B, 69] SMPL body pose in axis-angle format
    - betas: [B, >=num_betas]
    """

    def __init__(
        self,
        smpl_model_path: str,
        num_betas: int = 10,
        num_points: int = 1024,
        local_coord: bool = True,
        device: str | torch.device = "cpu",
        gender: str = "neutral",
    ) -> None:
        super().__init__()
        self.num_betas = int(num_betas)
        self.num_points = int(num_points)
        self.local_coord = bool(local_coord)
        self.device = torch.device(device)
        self.smpl = SMPL_Parser(
            model_path=smpl_model_path,
            gender=gender,
            num_betas=self.num_betas,
            create_transl=False,
        ).to(self.device)
        faces = torch.as_tensor(self.smpl.faces.astype("int64"), dtype=torch.long)
        self.register_buffer("faces_tensor", faces, persistent=False)

    def forward(
        self,
        root_pos: torch.Tensor,
        root_rot_xyzw: torch.Tensor,
        dof_pos: torch.Tensor,
        betas: torch.Tensor,
    ) -> torch.Tensor:
        vertices = self.vertices_from_pose(root_pos, root_rot_xyzw, dof_pos, betas)
        return _sample_mesh_surface(vertices, self.faces_tensor, self.num_points)

    def vertices_from_pose(
        self,
        root_pos: torch.Tensor,
        root_rot_xyzw: torch.Tensor,
        dof_pos: torch.Tensor,
        betas: torch.Tensor,
    ) -> torch.Tensor:
        """Return posed SMPL mesh vertices with the same coordinate convention."""

        root_pos = root_pos.to(self.device).float()
        root_rot_xyzw = root_rot_xyzw.to(self.device).float()
        dof_pos = dof_pos.to(self.device).float()
        betas = betas.to(self.device).float()[..., : self.num_betas]

        if dof_pos.shape[-1] != 69:
            raise ValueError(f"dof_pos must have 69 values per sample, got {dof_pos.shape[-1]}")

        global_orient = _quat_xyzw_to_axis_angle(root_rot_xyzw)
        smpl_output = self.smpl(
            betas=betas,
            global_orient=global_orient,
            body_pose=dof_pos,
            transl=None,
        )
        vertices = smpl_output.vertices
        if not self.local_coord:
            vertices = vertices + root_pos[:, None, :]
        return vertices

    def faces(self) -> torch.Tensor:
        """Return SMPL triangle face indices as a tensor."""

        return self.faces_tensor


__all__ = ["SmplToPointCloud"]
