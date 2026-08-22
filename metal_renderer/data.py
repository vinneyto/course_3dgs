from __future__ import annotations

from dataclasses import dataclass
from math import isqrt
from pathlib import Path

import torch

from util import build_covariance


_MAX_SH_LEVELS = 4


@dataclass(frozen=True)
class GaussianData:
    """Renderer-facing Gaussian data with a canonical SH coefficient layout."""

    positions: torch.Tensor
    sh_coefficients: torch.Tensor
    opacity_raw: torch.Tensor
    sigma: torch.Tensor
    sh_levels: int

    def __post_init__(self) -> None:
        if not 1 <= self.sh_levels <= _MAX_SH_LEVELS:
            raise ValueError(f"sh_levels must be in [1, {_MAX_SH_LEVELS}]")

        num_gaussians = self.positions.shape[0]
        expected_coefficients = self.sh_levels * self.sh_levels

        if self.positions.shape != (num_gaussians, 3):
            raise ValueError("positions must have shape [N, 3]")
        if self.sh_coefficients.shape != (num_gaussians, expected_coefficients, 3):
            raise ValueError(
                "sh_coefficients must have shape "
                f"[N, {expected_coefficients}, 3] for sh_levels={self.sh_levels}"
            )
        if self.opacity_raw.numel() != num_gaussians:
            raise ValueError("opacity_raw must contain one value per Gaussian")
        if self.sigma.shape != (num_gaussians, 3, 3):
            raise ValueError("sigma must have shape [N, 3, 3]")

        device = self.positions.device
        tensors = (self.sh_coefficients, self.opacity_raw, self.sigma)
        if any(tensor.device != device for tensor in tensors):
            raise ValueError("all Gaussian tensors must live on the same device")
        if any(tensor.dtype != torch.float32 for tensor in (self.positions, *tensors)):
            raise TypeError("the Metal renderer currently expects float32 Gaussian tensors")

    @property
    def num_gaussians(self) -> int:
        return self.positions.shape[0]

    @property
    def sh_coefficient_count(self) -> int:
        return self.sh_levels * self.sh_levels

    @classmethod
    def from_flat_tensors(
        cls,
        *,
        positions: torch.Tensor,
        f_dc: torch.Tensor,
        f_rest: torch.Tensor | None,
        opacity_raw: torch.Tensor,
        sigma: torch.Tensor,
        sh_levels: int | None = None,
    ) -> "GaussianData":
        """
        Adapt flat checkpoint SH data to renderer-facing [N, K, 3].

        f_dc is [N, 3]. f_rest is [N, 3 * (K - 1)] with all red
        coefficients first, then green, then blue. If sh_levels is omitted, the
        highest complete level present in f_rest is used.
        """
        positions = positions.to(dtype=torch.float32).contiguous()
        f_dc = f_dc.to(device=positions.device, dtype=torch.float32).contiguous()
        opacity_raw = opacity_raw.to(
            device=positions.device,
            dtype=torch.float32,
        ).contiguous()
        sigma = sigma.to(device=positions.device, dtype=torch.float32).contiguous()

        num_gaussians = positions.shape[0]
        if f_dc.shape != (num_gaussians, 3):
            raise ValueError("f_dc must have shape [N, 3]")

        if f_rest is None:
            available_levels = 1
            rest_per_channel = 0
        else:
            f_rest = f_rest.to(
                device=positions.device,
                dtype=torch.float32,
            ).contiguous()
            if f_rest.ndim != 2 or f_rest.shape[0] != num_gaussians:
                raise ValueError("f_rest must have shape [N, 3 * (K - 1)]")
            if f_rest.shape[1] % 3 != 0:
                raise ValueError("f_rest width must be divisible by 3")

            rest_per_channel = f_rest.shape[1] // 3
            available_coefficient_count = rest_per_channel + 1
            available_levels = isqrt(available_coefficient_count)
            if available_levels * available_levels != available_coefficient_count:
                raise ValueError("f_rest does not contain a complete number of SH levels")
            if available_levels > _MAX_SH_LEVELS:
                raise ValueError(f"only up to {_MAX_SH_LEVELS} SH levels are supported")

        selected_levels = available_levels if sh_levels is None else sh_levels
        if not 1 <= selected_levels <= available_levels:
            raise ValueError(
                f"sh_levels={selected_levels} is not available; "
                f"checkpoint provides {available_levels} level(s)"
            )

        coefficient_count = selected_levels * selected_levels
        selected_rest_count = coefficient_count - 1
        coefficients = torch.empty(
            (num_gaussians, coefficient_count, 3),
            device=positions.device,
            dtype=torch.float32,
        )
        coefficients[:, 0] = f_dc

        if selected_rest_count:
            assert f_rest is not None
            for channel in range(3):
                source_start = channel * rest_per_channel
                source_end = source_start + selected_rest_count
                coefficients[:, 1:, channel] = f_rest[:, source_start:source_end]

        return cls(
            positions=positions,
            sh_coefficients=coefficients.contiguous(),
            opacity_raw=opacity_raw,
            sigma=sigma,
            sh_levels=selected_levels,
        )

    @classmethod
    def from_checkpoint(
        cls,
        directory: str | Path,
        *,
        device: torch.device | str = "cpu",
        sh_levels: int | None = None,
    ) -> "GaussianData":
        """Load the tensor checkpoint layout used by the sample 3DGS scene."""
        directory = Path(directory)
        device = torch.device(device)

        positions = _load_float_tensor(directory / "pos_param.pt", device)
        opacity_raw = _load_float_tensor(directory / "alpha_raw_param.pt", device)
        f_dc = _load_float_tensor(directory / "f_dc.pt", device)
        f_rest_path = directory / "f_rest.pt"
        f_rest = _load_float_tensor(f_rest_path, device) if f_rest_path.exists() else None
        scale_raw = _load_float_tensor(directory / "scale_raw.pt", device)
        rot_raw = _load_float_tensor(directory / "rot_raw.pt", device)
        sigma = build_covariance(scale_raw, rot_raw)

        return cls.from_flat_tensors(
            positions=positions,
            f_dc=f_dc,
            f_rest=f_rest,
            opacity_raw=opacity_raw,
            sigma=sigma,
            sh_levels=sh_levels,
        )

    def evaluate_color(self, c2w: torch.Tensor) -> torch.Tensor:
        """PyTorch SH evaluation useful for validating the Metal renderer."""
        camera_origin = c2w[:3, 3]
        view_direction = self.positions - camera_origin.unsqueeze(0)
        view_direction = view_direction / (
            torch.linalg.norm(view_direction, dim=-1, keepdim=True) + 1e-8
        )
        basis = _spherical_harmonics_basis(view_direction, self.sh_levels)
        return torch.sigmoid(
            (self.sh_coefficients * basis.unsqueeze(-1)).sum(dim=1)
        )


def _load_float_tensor(path: Path, device: torch.device) -> torch.Tensor:
    value = torch.load(path, weights_only=False)
    tensor = value if isinstance(value, torch.Tensor) else torch.from_numpy(value)
    return tensor.to(device=device, dtype=torch.float32)


def _spherical_harmonics_basis(
    view_direction: torch.Tensor,
    sh_levels: int,
) -> torch.Tensor:
    x, y, z = view_direction.unbind(dim=-1)

    basis = [torch.full_like(x, 0.28209479177387814)]
    if sh_levels == 1:
        return torch.stack(basis, dim=1)

    basis.extend(
        (
            -0.4886025119029199 * y,
            +0.4886025119029199 * z,
            -0.4886025119029199 * x,
        )
    )
    if sh_levels == 2:
        return torch.stack(basis, dim=1)

    x2 = x * x
    y2 = y * y
    z2 = z * z
    xy = x * y
    yz = y * z
    xz = x * z
    x2_minus_y2 = x2 - y2

    basis.extend(
        (
            +1.0925484305920792 * xy,
            +1.0925484305920792 * yz,
            +0.31539156525252005 * (3.0 * z2 - 1.0),
            +1.0925484305920792 * xz,
            +0.5462742152960396 * x2_minus_y2,
        )
    )
    if sh_levels == 3:
        return torch.stack(basis, dim=1)

    four_z2_minus_x2_minus_y2 = 4.0 * z2 - x2 - y2
    basis.extend(
        (
            +0.5900435899266435 * y * (3.0 * x2 - y2),
            +2.890611442640554 * xy * z,
            +0.4570457994644658 * y * four_z2_minus_x2_minus_y2,
            +0.3731763325901154
            * z
            * (2.0 * z2 - 3.0 * x2 - 3.0 * y2),
            +0.4570457994644658 * x * four_z2_minus_x2_minus_y2,
            +1.445305721320277 * z * x2_minus_y2,
            +0.5900435899266435 * x * (x2 - 3.0 * y2),
        )
    )
    return torch.stack(basis, dim=1)
