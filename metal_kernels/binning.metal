#include <metal_stdlib>
using namespace metal;

kernel void emit_intersections(
    const device int* tile_bounds [[buffer(0)]],
    const device int* tile_counts [[buffer(1)]],
    const device int* intersection_offsets [[buffer(2)]],
    const device float* depths [[buffer(3)]],
    device uint* tile_ids [[buffer(4)]],
    device uint* depth_bits [[buffer(5)]],
    device int* gaussian_ids [[buffer(6)]],
    constant uint& num_gaussians [[buffer(7)]],
    constant uint& num_tiles_u [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_gaussians) {
        return;
    }

    const int count = tile_counts[gid];
    if (count <= 0) {
        return;
    }

    const uint bbase = gid * 4;
    const int min_u = tile_bounds[bbase];
    const int max_u = tile_bounds[bbase + 1];
    const int min_v = tile_bounds[bbase + 2];
    const int max_v = tile_bounds[bbase + 3];

    const int n_v = max_v - min_v + 1;
    const int output_start = intersection_offsets[gid];
    const uint depth_key = as_type<uint>(depths[gid]);

    // v changes fastest, matching the original PyTorch enumeration.
    for (int local_id = 0; local_id < count; ++local_id) {
        const int local_u = local_id / n_v;
        const int local_v = local_id - local_u * n_v;
        const uint tile_u = uint(min_u + local_u);
        const uint tile_v = uint(min_v + local_v);
        const uint output_index = uint(output_start + local_id);

        tile_ids[output_index] = tile_v * num_tiles_u + tile_u;
        depth_bits[output_index] = depth_key;
        gaussian_ids[output_index] = int(gid);
    }
}

kernel void fill_int(
    device int* output [[buffer(0)]],
    constant int& value [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index < n) {
        output[index] = value;
    }
}

kernel void count_sorted_tiles(
    const device uint* sorted_tile_ids [[buffer(0)]],
    device atomic_uint* counts_per_tile [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= n) {
        return;
    }

    atomic_fetch_add_explicit(
        &counts_per_tile[sorted_tile_ids[index]],
        1u,
        memory_order_relaxed
    );
}

kernel void finalize_tile_offsets(
    const device int* counts [[buffer(0)]],
    const device int* starts [[buffer(1)]],
    device int* offsets [[buffer(2)]],
    constant uint& num_tiles [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index < num_tiles) {
        offsets[index] = starts[index];
    } else if (index == num_tiles && num_tiles > 0) {
        offsets[num_tiles] =
            starts[num_tiles - 1] + counts[num_tiles - 1];
    }
}
