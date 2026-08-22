#include <metal_stdlib>
using namespace metal;

#define SCAN_BLOCK_SIZE 256

kernel void scan_blocks(
    const device int* input [[buffer(0)]],
    device int* output [[buffer(1)]],
    device int* block_sums [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    threadgroup int scratch[SCAN_BLOCK_SIZE];

    const uint group_start = group_id * SCAN_BLOCK_SIZE;
    const uint index = group_start + tid;
    const int input_value = index < n ? input[index] : 0;
    scratch[tid] = input_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Hillis-Steele inclusive scan inside the block.
    for (uint offset = 1; offset < SCAN_BLOCK_SIZE; offset <<= 1) {
        int addend = 0;
        if (tid >= offset) {
            addend = scratch[tid - offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid >= offset) {
            scratch[tid] += addend;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (index < n) {
        output[index] = scratch[tid] - input_value;
    }

    if (tid == 0 && group_start < n) {
        const uint valid_count = min(uint(SCAN_BLOCK_SIZE), n - group_start);
        block_sums[group_id] = scratch[valid_count - 1];
    }
}

kernel void add_block_offsets(
    device int* output [[buffer(0)]],
    const device int* block_offsets [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= n) {
        return;
    }

    const uint block_id = index / SCAN_BLOCK_SIZE;
    output[index] += block_offsets[block_id];
}

kernel void write_scan_total(
    const device int* counts [[buffer(0)]],
    const device int* offsets [[buffer(1)]],
    device int* total [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index != 0) {
        return;
    }

    total[0] = n == 0 ? 0 : offsets[n - 1] + counts[n - 1];
}
