from __future__ import annotations

import torch


def quat_to_mat(quat: torch.Tensor) -> torch.Tensor:
    """Convert [x, y, z, w] quaternions to 3x3 rotation matrices."""
    x, y, z, w = quat.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w

    return torch.stack(
        [
            1 - 2 * yy - 2 * zz,
            2 * xy - 2 * zw,
            2 * xz + 2 * yw,
            2 * xy + 2 * zw,
            1 - 2 * xx - 2 * zz,
            2 * yz - 2 * xw,
            2 * xz - 2 * yw,
            2 * yz + 2 * xw,
            1 - 2 * xx - 2 * yy,
        ],
        dim=-1,
    ).reshape(quat.shape[:-1] + (3, 3))


def build_covariance(scale_raw: torch.Tensor, q_raw: torch.Tensor) -> torch.Tensor:
    """Build world-space covariance matrices from raw 3DGS scale/rotation data."""
    scale = torch.exp(scale_raw).clamp_min(1e-6)
    q = q_raw / (torch.linalg.norm(q_raw, dim=-1, keepdim=True) + 1e-9)
    rotation = quat_to_mat(q)
    scale_matrix = torch.diag_embed(scale)
    return rotation @ scale_matrix @ scale_matrix @ rotation.transpose(1, 2)
