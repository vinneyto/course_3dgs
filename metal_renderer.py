from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from time import perf_counter

import torch


_KERNEL_DIR = Path(__file__).with_name("metal_kernels")
_SCAN_BLOCK_SIZE = 256
_DISPATCH_SIZE = 256

_last_metal_stats: dict[str, float | int] = {}


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _read_kernel(name: str) -> str:
    return (_KERNEL_DIR / name).read_text()


@lru_cache(maxsize=None)
def _compile_kernel_file(name: str):
    return torch.mps.compile_shader(_read_kernel(name))


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
    # Dynamic K requires one host-visible scalar before intersection buffers
    # can be allocated. All heavy work before and after this point stays on GPU.
    return int(total.item())


def _lsd_radix_sort(
    tile_ids: torch.Tensor,
    depth_bits: torch.Tensor,
    gaussian_ids: torch.Tensor,
    num_tiles: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Stable 4-bit LSD radix sort.

    Depth is sorted first, then tile id. Because each pass is stable, the final
    order is equivalent to ORDER BY tile_id, depth for positive finite z.
    """
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


def get_last_metal_stats() -> dict[str, float | int]:
    return dict(_last_metal_stats)


def gaussian_rasterization_metal(
    pos: torch.Tensor,
    color: torch.Tensor,
    opacity_raw: torch.Tensor,
    sigma: torch.Tensor,
    c2w: torch.Tensor,
    H: int,
    W: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    near: float = 2e-3,
    far: float = 100.0,
    pix_guard: int = 64,
    T: int = 16,
    min_conis: float = 1e-6,
    chi_square_clip: float = 9.21,
    alpha_max: float = 0.99,
    alpha_cutoff: float = 1 / 255.0,
) -> torch.Tensor:
    """
    Full experimental 3DGS render front-end + rasterizer implemented with
    handwritten Metal compute kernels.

    PyTorch is used only as the tensor owner / allocator and Python dispatch
    layer. Projection, covariance projection, rectangular AABB construction,
    scan, Gaussian-tile emission, radix sorting, tile-offset construction and
    pixel rasterization execute in Metal.
    """
    if not torch.backends.mps.is_available():
        raise RuntimeError("The Metal renderer requires an available MPS device")
    if pos.device.type != "mps":
        raise ValueError("pos must be on MPS")
    if pos.dtype != torch.float32:
        raise TypeError("The Metal pipeline currently expects float32 tensors")
    if color.dtype != torch.float32 or sigma.dtype != torch.float32:
        raise TypeError("color and sigma must be float32")
    if opacity_raw.dtype != torch.float32 or c2w.dtype != torch.float32:
        raise TypeError("opacity_raw and c2w must be float32")
    if T * T > 256:
        raise ValueError("T * T must be <= 256")

    started_at = perf_counter()
    device = pos.device
    n = pos.shape[0]
    num_tiles_u = (W + T - 1) // T
    num_tiles_v = (H + T - 1) // T
    num_tiles = num_tiles_u * num_tiles_v

    pos = pos.contiguous()
    color = color.contiguous()
    opacity_raw = opacity_raw.contiguous()
    sigma = sigma.contiguous()
    c2w = c2w.contiguous()

    # Per-Gaussian outputs. Invisible Gaussians keep tile_count == 0.
    gaussians = torch.empty((n, 9), device=device, dtype=torch.float32)
    depths = torch.empty(n, device=device, dtype=torch.float32)
    tile_bounds = torch.empty((n, 4), device=device, dtype=torch.int32)
    tile_counts = torch.empty(n, device=device, dtype=torch.int32)

    image_parameters = torch.tensor(
        (W, H, T, pix_guard),
        device=device,
        dtype=torch.int32,
    )
    camera_parameters = torch.tensor(
        (fx, fy, cx, cy, near, far, min_conis),
        device=device,
        dtype=torch.float32,
    )

    setup = _compile_kernel_file("gaussian_setup.metal")
    setup.project_gaussians(
        pos,
        color,
        opacity_raw,
        sigma,
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
        return torch.zeros((H, W, 3), device=device, dtype=torch.float32)

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

    final_image = torch.empty((H, W, 3), device=device, dtype=torch.float32)
    raster_parameters_i32 = torch.tensor(
        (W, H, num_tiles_u),
        device=device,
        dtype=torch.int32,
    )
    raster_parameters_f32 = torch.tensor(
        (chi_square_clip, alpha_max, alpha_cutoff),
        device=device,
        dtype=torch.float32,
    )

    raster = _compile_tile_rasterizer(T)
    threads_per_tile = T * T
    raster.tile_rasterizer_kernel(
        gaussians,
        gaussian_ids,
        tile_offsets,
        final_image,
        raster_parameters_i32,
        raster_parameters_f32,
        threads=[num_tiles * threads_per_tile, 1, 1],
        group_size=[threads_per_tile, 1, 1],
    )

    torch.mps.synchronize()
    elapsed = perf_counter() - started_at

    _last_metal_stats.clear()
    _last_metal_stats.update(
        gaussians=n,
        image_tiles=num_tiles,
        gaussian_tile_intersections=num_intersections,
        radix_passes=radix_passes,
        total_seconds=elapsed,
    )

    return final_image
