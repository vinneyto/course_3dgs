from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import torch

from .data import GaussianData


_SCAN_BLOCK_SIZE = 256
_DISPATCH_SIZE = 256


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _read_kernel(name: str) -> str:
    return (
        files("course_3dgs.metal_renderer")
        .joinpath("kernels", name)
        .read_text(encoding="utf-8")
    )


@lru_cache(maxsize=None)
def _compile_kernel_file(name: str):
    return torch.mps.compile_shader(_read_kernel(name))


@lru_cache(maxsize=None)
def _compile_gaussian_setup(sh_levels: int):
    source = _read_kernel("gaussian_setup.metal")
    source = source.replace("__SH_LEVELS__", str(sh_levels))
    return torch.mps.compile_shader(source)


@lru_cache(maxsize=None)
def _compile_tile_rasterizer(tile_size: int):
    if tile_size * tile_size > 256:
        raise ValueError("tile_size * tile_size must be <= 256")
    source = _read_kernel("tile_rasterizer.metal")
    source = source.replace("__TILE_SIZE__", str(tile_size))
    source = source.replace("__THREADS_PER_TILE__", str(tile_size * tile_size))
    return torch.mps.compile_shader(source)


def _exclusive_scan_int32(values: torch.Tensor) -> torch.Tensor:
    """Hierarchical exclusive scan implemented entirely by Metal kernels."""
    if values.dtype != torch.int32 or values.device.type != "mps":
        raise TypeError("exclusive scan expects an int32 MPS tensor")

    n = values.numel()
    output = torch.empty_like(values)
    if n == 0:
        return output

    scan_lib = _compile_kernel_file("scan.metal")
    num_blocks = (n + _SCAN_BLOCK_SIZE - 1) // _SCAN_BLOCK_SIZE
    block_sums = torch.empty(num_blocks, device=values.device, dtype=torch.int32)

    scan_lib.scan_blocks(
        values,
        output,
        block_sums,
        n,
        threads=[num_blocks * _SCAN_BLOCK_SIZE, 1, 1],
        group_size=[_SCAN_BLOCK_SIZE, 1, 1],
    )

    if num_blocks > 1:
        block_offsets = _exclusive_scan_int32(block_sums)
        scan_lib.add_block_offsets(
            output,
            block_offsets,
            n,
            threads=[_round_up(n, _DISPATCH_SIZE), 1, 1],
            group_size=[_DISPATCH_SIZE, 1, 1],
        )

    return output


def _scan_total(counts: torch.Tensor, offsets: torch.Tensor) -> int:
    if counts.numel() == 0:
        return 0
    scan_lib = _compile_kernel_file("scan.metal")
    total = torch.empty(1, device=counts.device, dtype=torch.int32)
    scan_lib.write_scan_total(
        counts,
        offsets,
        total,
        counts.numel(),
        threads=[1, 1, 1],
        group_size=[1, 1, 1],
    )
    # Dynamic K still requires one host-visible scalar before the variable-size
    # intersection buffers can be allocated. render() intentionally does not
    # call torch.mps.synchronize(); callers synchronize only when they need it.
    return int(total.item())


def _lsd_radix_sort(
    tile_ids: torch.Tensor,
    depth_bits: torch.Tensor,
    gaussian_ids: torch.Tensor,
    num_tiles: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Stable 4-bit LSD radix sort by depth, then tile id."""
    n = tile_ids.numel()
    if n <= 1:
        return tile_ids, depth_bits, gaussian_ids, 0

    radix = _compile_kernel_file("radix_sort.metal")
    block_items = 256
    radix_bins = 16
    num_blocks = (n + block_items - 1) // block_items

    block_histograms = torch.empty(
        num_blocks * radix_bins,
        device=tile_ids.device,
        dtype=torch.int32,
    )
    block_prefixes = torch.empty_like(block_histograms)
    digit_totals = torch.empty(radix_bins, device=tile_ids.device, dtype=torch.int32)
    digit_offsets = torch.empty_like(digit_totals)

    out_tile = torch.empty_like(tile_ids)
    out_depth = torch.empty_like(depth_bits)
    out_gaussian = torch.empty_like(gaussian_ids)
    in_tile, in_depth, in_gaussian = tile_ids, depth_bits, gaussian_ids

    tile_bits = max(1, (num_tiles - 1).bit_length())
    passes = [(0, shift) for shift in range(0, 32, 4)]
    passes.extend((1, shift) for shift in range(0, tile_bits, 4))

    block_threads = [_round_up(num_blocks, _DISPATCH_SIZE), 1, 1]
    block_group = [_DISPATCH_SIZE, 1, 1]

    for key_kind, shift in passes:
        radix.radix_histogram_blocks(
            in_tile,
            in_depth,
            block_histograms,
            n,
            key_kind,
            shift,
            threads=block_threads,
            group_size=block_group,
        )
        radix.radix_scan_block_histograms(
            block_histograms,
            block_prefixes,
            digit_totals,
            num_blocks,
            threads=[radix_bins, 1, 1],
            group_size=[radix_bins, 1, 1],
        )
        radix.radix_scan_digit_totals(
            digit_totals,
            digit_offsets,
            threads=[1, 1, 1],
            group_size=[1, 1, 1],
        )
        radix.radix_scatter_blocks(
            in_tile,
            in_depth,
            in_gaussian,
            block_prefixes,
            digit_offsets,
            out_tile,
            out_depth,
            out_gaussian,
            n,
            key_kind,
            shift,
            threads=block_threads,
            group_size=block_group,
        )
        in_tile, out_tile = out_tile, in_tile
        in_depth, out_depth = out_depth, in_depth
        in_gaussian, out_gaussian = out_gaussian, in_gaussian

    return in_tile, in_depth, in_gaussian, len(passes)


class MetalRenderer:
    """Handwritten Metal 3DGS renderer whose per-frame input is only c2w."""

    def __init__(
        self,
        data: GaussianData,
        *,
        H: int,
        W: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        near: float = 2e-3,
        far: float = 100.0,
        pix_guard: int = 64,
        tile_size: int = 16,
        min_conis: float = 1e-6,
        chi_square_clip: float = 9.21,
        alpha_max: float = 0.99,
        alpha_cutoff: float = 1 / 255.0,
    ) -> None:
        if not torch.backends.mps.is_available():
            raise RuntimeError("The Metal renderer requires an available MPS device")
        if data.positions.device.type != "mps":
            raise ValueError("GaussianData must live on MPS")
        if tile_size * tile_size > 256:
            raise ValueError("tile_size * tile_size must be <= 256")

        self.data = data
        self.H = H
        self.W = W
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.near = near
        self.far = far
        self.pix_guard = pix_guard
        self.tile_size = tile_size
        self.min_conis = min_conis
        self.chi_square_clip = chi_square_clip
        self.alpha_max = alpha_max
        self.alpha_cutoff = alpha_cutoff
        self._last_stats: dict[str, float | int] = {}

        # Compile an SH-specialized setup kernel once for this data layout.
        self._setup = _compile_gaussian_setup(data.sh_levels)
        self._raster = _compile_tile_rasterizer(tile_size)

    @property
    def last_stats(self) -> dict[str, float | int]:
        return dict(self._last_stats)

    def render(self, c2w: torch.Tensor) -> torch.Tensor:
        if c2w.device.type != "mps" or c2w.dtype != torch.float32:
            raise TypeError("c2w must be a float32 MPS tensor")
        if c2w.shape != (4, 4):
            raise ValueError("c2w must have shape [4, 4]")

        data = self.data
        device = data.positions.device
        n = data.num_gaussians
        T = self.tile_size
        num_tiles_u = (self.W + T - 1) // T
        num_tiles_v = (self.H + T - 1) // T
        num_tiles = num_tiles_u * num_tiles_v
        c2w = c2w.contiguous()

        gaussians = torch.empty((n, 9), device=device, dtype=torch.float32)
        depths = torch.empty(n, device=device, dtype=torch.float32)
        tile_bounds = torch.empty((n, 4), device=device, dtype=torch.int32)
        tile_counts = torch.empty(n, device=device, dtype=torch.int32)

        image_parameters = torch.tensor(
            (self.W, self.H, T, self.pix_guard),
            device=device,
            dtype=torch.int32,
        )
        camera_parameters = torch.tensor(
            (self.fx, self.fy, self.cx, self.cy, self.near, self.far, self.min_conis),
            device=device,
            dtype=torch.float32,
        )

        self._setup.project_gaussians(
            data.positions,
            data.sh_coefficients,
            data.opacity_raw,
            data.sigma,
            c2w,
            gaussians,
            depths,
            tile_bounds,
            tile_counts,
            image_parameters,
            camera_parameters,
            n,
            threads=[_round_up(n, _DISPATCH_SIZE), 1, 1],
            group_size=[_DISPATCH_SIZE, 1, 1],
        )

        intersection_offsets = _exclusive_scan_int32(tile_counts)
        num_intersections = _scan_total(tile_counts, intersection_offsets)
        if num_intersections == 0:
            self._last_stats = {
                "gaussians": n,
                "image_tiles": num_tiles,
                "gaussian_tile_intersections": 0,
                "radix_passes": 0,
                "sh_levels": data.sh_levels,
            }
            return torch.zeros((self.H, self.W, 3), device=device, dtype=torch.float32)

        tile_ids = torch.empty(num_intersections, device=device, dtype=torch.int32)
        depth_bits = torch.empty(num_intersections, device=device, dtype=torch.int32)
        gaussian_ids = torch.empty(num_intersections, device=device, dtype=torch.int32)

        binning = _compile_kernel_file("binning.metal")
        binning.emit_intersections(
            tile_bounds,
            tile_counts,
            intersection_offsets,
            depths,
            tile_ids,
            depth_bits,
            gaussian_ids,
            n,
            num_tiles_u,
            threads=[_round_up(n, _DISPATCH_SIZE), 1, 1],
            group_size=[_DISPATCH_SIZE, 1, 1],
        )

        tile_ids, depth_bits, gaussian_ids, radix_passes = _lsd_radix_sort(
            tile_ids,
            depth_bits,
            gaussian_ids,
            num_tiles,
        )

        counts_per_tile = torch.empty(num_tiles, device=device, dtype=torch.int32)
        binning.fill_int(
            counts_per_tile,
            0,
            num_tiles,
            threads=[_round_up(num_tiles, _DISPATCH_SIZE), 1, 1],
            group_size=[_DISPATCH_SIZE, 1, 1],
        )
        binning.count_sorted_tiles(
            tile_ids,
            counts_per_tile,
            num_intersections,
            threads=[_round_up(num_intersections, _DISPATCH_SIZE), 1, 1],
            group_size=[_DISPATCH_SIZE, 1, 1],
        )

        tile_starts = _exclusive_scan_int32(counts_per_tile)
        tile_offsets = torch.empty(num_tiles + 1, device=device, dtype=torch.int32)
        binning.finalize_tile_offsets(
            counts_per_tile,
            tile_starts,
            tile_offsets,
            num_tiles,
            threads=[_round_up(num_tiles + 1, _DISPATCH_SIZE), 1, 1],
            group_size=[_DISPATCH_SIZE, 1, 1],
        )

        final_image = torch.empty((self.H, self.W, 3), device=device, dtype=torch.float32)
        raster_parameters_i32 = torch.tensor(
            (self.W, self.H, num_tiles_u),
            device=device,
            dtype=torch.int32,
        )
        raster_parameters_f32 = torch.tensor(
            (self.chi_square_clip, self.alpha_max, self.alpha_cutoff),
            device=device,
            dtype=torch.float32,
        )

        threads_per_tile = T * T
        self._raster.tile_rasterizer_kernel(
            gaussians,
            gaussian_ids,
            tile_offsets,
            final_image,
            raster_parameters_i32,
            raster_parameters_f32,
            threads=[num_tiles * threads_per_tile, 1, 1],
            group_size=[threads_per_tile, 1, 1],
        )

        self._last_stats = {
            "gaussians": n,
            "image_tiles": num_tiles,
            "gaussian_tile_intersections": num_intersections,
            "radix_passes": radix_passes,
            "sh_levels": data.sh_levels,
        }
        return final_image
