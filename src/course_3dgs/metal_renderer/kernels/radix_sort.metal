#include <metal_stdlib>
using namespace metal;

#define RADIX_BITS 4
#define RADIX_BINS 16
#define RADIX_BLOCK_ITEMS 256

inline uint select_key(
    const device uint* tile_ids,
    const device uint* depth_bits,
    uint index,
    uint key_kind
) {
    return key_kind == 0 ? depth_bits[index] : tile_ids[index];
}

kernel void radix_histogram_blocks(
    const device uint* tile_ids [[buffer(0)]],
    const device uint* depth_bits [[buffer(1)]],
    device uint* block_histograms [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& key_kind [[buffer(4)]],
    constant uint& shift [[buffer(5)]],
    uint block_id [[thread_position_in_grid]]
) {
    const uint block_start = block_id * RADIX_BLOCK_ITEMS;
    if (block_start >= n) {
        return;
    }

    uint histogram[RADIX_BINS] = {
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0
    };

    const uint block_end = min(block_start + uint(RADIX_BLOCK_ITEMS), n);
    for (uint index = block_start; index < block_end; ++index) {
        const uint key = select_key(tile_ids, depth_bits, index, key_kind);
        const uint digit = (key >> shift) & (RADIX_BINS - 1);
        histogram[digit] += 1;
    }

    const uint output_base = block_id * RADIX_BINS;
    for (uint digit = 0; digit < RADIX_BINS; ++digit) {
        block_histograms[output_base + digit] = histogram[digit];
    }
}

kernel void radix_scan_block_histograms(
    const device uint* block_histograms [[buffer(0)]],
    device uint* block_prefixes [[buffer(1)]],
    device uint* digit_totals [[buffer(2)]],
    constant uint& num_blocks [[buffer(3)]],
    uint digit [[thread_position_in_grid]]
) {
    if (digit >= RADIX_BINS) {
        return;
    }

    uint running = 0;
    for (uint block = 0; block < num_blocks; ++block) {
        const uint index = block * RADIX_BINS + digit;
        block_prefixes[index] = running;
        running += block_histograms[index];
    }
    digit_totals[digit] = running;
}

kernel void radix_scan_digit_totals(
    const device uint* digit_totals [[buffer(0)]],
    device uint* digit_offsets [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    if (index != 0) {
        return;
    }

    uint running = 0;
    for (uint digit = 0; digit < RADIX_BINS; ++digit) {
        digit_offsets[digit] = running;
        running += digit_totals[digit];
    }
}

kernel void radix_scatter_blocks(
    const device uint* tile_ids_in [[buffer(0)]],
    const device uint* depth_bits_in [[buffer(1)]],
    const device int* gaussian_ids_in [[buffer(2)]],
    const device uint* block_prefixes [[buffer(3)]],
    const device uint* digit_offsets [[buffer(4)]],
    device uint* tile_ids_out [[buffer(5)]],
    device uint* depth_bits_out [[buffer(6)]],
    device int* gaussian_ids_out [[buffer(7)]],
    constant uint& n [[buffer(8)]],
    constant uint& key_kind [[buffer(9)]],
    constant uint& shift [[buffer(10)]],
    uint block_id [[thread_position_in_grid]]
) {
    const uint block_start = block_id * RADIX_BLOCK_ITEMS;
    if (block_start >= n) {
        return;
    }

    uint local_counts[RADIX_BINS] = {
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0
    };

    const uint block_end = min(block_start + uint(RADIX_BLOCK_ITEMS), n);
    const uint prefix_base = block_id * RADIX_BINS;

    // One GPU thread owns one 256-item block and walks it in input order.
    // That preserves stability without atomics.
    for (uint index = block_start; index < block_end; ++index) {
        const uint key = select_key(
            tile_ids_in,
            depth_bits_in,
            index,
            key_kind
        );
        const uint digit = (key >> shift) & (RADIX_BINS - 1);
        const uint destination =
            digit_offsets[digit]
            + block_prefixes[prefix_base + digit]
            + local_counts[digit];

        local_counts[digit] += 1;

        tile_ids_out[destination] = tile_ids_in[index];
        depth_bits_out[destination] = depth_bits_in[index];
        gaussian_ids_out[destination] = gaussian_ids_in[index];
    }
}
