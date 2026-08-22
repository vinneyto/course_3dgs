#include <metal_stdlib>
using namespace metal;

#define TILE_SIZE __TILE_SIZE__
#define THREADS_PER_TILE __THREADS_PER_TILE__
#define GAUSSIAN_COMPONENTS 9

kernel void tile_rasterizer_kernel(
    const device float* gaussians [[buffer(0)]],
    const device int* gaussian_ids [[buffer(1)]],
    const device int* tile_offsets [[buffer(2)]],
    device float* final_image [[buffer(3)]],
    const device int* image_parameters [[buffer(4)]],
    const device float* render_parameters [[buffer(5)]],
    uint tile_id [[threadgroup_position_in_grid]],
    uint pixel_in_tile [[thread_index_in_threadgroup]]
) {
    threadgroup float gaussians_shared[
        THREADS_PER_TILE * GAUSSIAN_COMPONENTS
    ];

    const uint image_width = uint(image_parameters[0]);
    const uint image_height = uint(image_parameters[1]);
    const uint num_tiles_u = uint(image_parameters[2]);

    const float chi_square_clip = render_parameters[0];
    const float alpha_max = render_parameters[1];
    const float alpha_cutoff = render_parameters[2];

    const uint tile_x = tile_id % num_tiles_u;
    const uint tile_y = tile_id / num_tiles_u;
    const uint pixel_x = tile_x * TILE_SIZE + pixel_in_tile % TILE_SIZE;
    const uint pixel_y = tile_y * TILE_SIZE + pixel_in_tile / TILE_SIZE;
    const bool valid_pixel =
        pixel_x < image_width && pixel_y < image_height;

    const int tile_start = tile_offsets[tile_id];
    const int tile_end = tile_offsets[tile_id + 1];

    float transmittance = 1.0f;
    float3 accumulated_color = float3(0.0f);

    for (
        int batch_start = tile_start;
        batch_start < tile_end;
        batch_start += THREADS_PER_TILE
    ) {
        const int batch_count = min(
            THREADS_PER_TILE,
            tile_end - batch_start
        );

        if (int(pixel_in_tile) < batch_count) {
            const int gaussian_id =
                gaussian_ids[batch_start + int(pixel_in_tile)];
            const int source_base = gaussian_id * GAUSSIAN_COMPONENTS;
            const int shared_base =
                int(pixel_in_tile) * GAUSSIAN_COMPONENTS;

            #pragma clang loop unroll(full)
            for (
                int component = 0;
                component < GAUSSIAN_COMPONENTS;
                ++component
            ) {
                gaussians_shared[shared_base + component] =
                    gaussians[source_base + component];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid_pixel) {
            for (
                int local_gaussian_id = 0;
                local_gaussian_id < batch_count;
                ++local_gaussian_id
            ) {
                const int gaussian_base =
                    local_gaussian_id * GAUSSIAN_COMPONENTS;
                const float du =
                    float(pixel_x) - gaussians_shared[gaussian_base];
                const float dv =
                    float(pixel_y) - gaussians_shared[gaussian_base + 1];

                const float q =
                    gaussians_shared[gaussian_base + 6] * du * du
                    + 2.0f * gaussians_shared[gaussian_base + 7] * du * dv
                    + gaussians_shared[gaussian_base + 8] * dv * dv;

                if (q <= chi_square_clip) {
                    const float alpha = min(
                        gaussians_shared[gaussian_base + 5]
                            * exp(-0.5f * q),
                        alpha_max
                    );

                    if (alpha >= alpha_cutoff) {
                        const float weight = transmittance * alpha;
                        accumulated_color += weight * float3(
                            gaussians_shared[gaussian_base + 2],
                            gaussians_shared[gaussian_base + 3],
                            gaussians_shared[gaussian_base + 4]
                        );
                        transmittance *= 1.0f - alpha;
                    }
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid_pixel) {
        const uint output_base =
            (pixel_y * image_width + pixel_x) * 3;
        const float3 result = clamp(
            accumulated_color,
            float3(0.0f),
            float3(1.0f)
        );
        final_image[output_base] = result.x;
        final_image[output_base + 1] = result.y;
        final_image[output_base + 2] = result.z;
    }
}
