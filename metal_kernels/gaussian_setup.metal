#include <metal_stdlib>
using namespace metal;

kernel void project_gaussians(
    const device float* positions [[buffer(0)]],
    const device float* colors [[buffer(1)]],
    const device float* opacity_raw [[buffer(2)]],
    const device float* sigma_world [[buffer(3)]],
    const device float* c2w [[buffer(4)]],
    device float* gaussians [[buffer(5)]],
    device float* depths [[buffer(6)]],
    device int* tile_bounds [[buffer(7)]],
    device int* tile_counts [[buffer(8)]],
    const device int* image_parameters [[buffer(9)]],
    const device float* camera_parameters [[buffer(10)]],
    constant uint& num_gaussians [[buffer(11)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_gaussians) {
        return;
    }

    const int W = image_parameters[0];
    const int H = image_parameters[1];
    const int tile_size = image_parameters[2];
    const int pix_guard = image_parameters[3];

    const float fx = camera_parameters[0];
    const float fy = camera_parameters[1];
    const float cx = camera_parameters[2];
    const float cy = camera_parameters[3];
    const float near_plane = camera_parameters[4];
    const float far_plane = camera_parameters[5];
    const float min_conis = camera_parameters[6];

    const uint pbase = gid * 3;
    const float3 p = float3(
        positions[pbase],
        positions[pbase + 1],
        positions[pbase + 2]
    );

    const float3 camera_origin = float3(c2w[3], c2w[7], c2w[11]);
    const float3 dp = p - camera_origin;

    // c2w stores R_cw row-major. These are the rows of R_wc = R_cw^T.
    const float3 r0 = float3(c2w[0], c2w[4], c2w[8]);
    const float3 r1 = float3(c2w[1], c2w[5], c2w[9]);
    const float3 r2 = float3(c2w[2], c2w[6], c2w[10]);

    const float x = dot(r0, dp);
    const float y = dot(r1, dp);
    const float z = dot(r2, dp);
    depths[gid] = z;

    if (!(z > near_plane && z < far_plane) || !isfinite(z)) {
        tile_counts[gid] = 0;
        return;
    }

    const float inv_z = 1.0f / z;
    const float u = fx * x * inv_z + cx;
    const float v = fy * y * inv_z + cy;

    if (
        !(u > -float(pix_guard) && u < float(W + pix_guard) &&
          v > -float(pix_guard) && v < float(H + pix_guard)) ||
        !isfinite(u) || !isfinite(v)
    ) {
        tile_counts[gid] = 0;
        return;
    }

    const uint sbase = gid * 9;
    const float s00 = sigma_world[sbase];
    const float s01 = sigma_world[sbase + 1];
    const float s02 = sigma_world[sbase + 2];
    const float s10 = sigma_world[sbase + 3];
    const float s11 = sigma_world[sbase + 4];
    const float s12 = sigma_world[sbase + 5];
    const float s20 = sigma_world[sbase + 6];
    const float s21 = sigma_world[sbase + 7];
    const float s22 = sigma_world[sbase + 8];

    // sigma_camera = R_wc * sigma_world * R_wc^T.
    const float3 sr0 = float3(
        s00 * r0.x + s01 * r0.y + s02 * r0.z,
        s10 * r0.x + s11 * r0.y + s12 * r0.z,
        s20 * r0.x + s21 * r0.y + s22 * r0.z
    );
    const float3 sr1 = float3(
        s00 * r1.x + s01 * r1.y + s02 * r1.z,
        s10 * r1.x + s11 * r1.y + s12 * r1.z,
        s20 * r1.x + s21 * r1.y + s22 * r1.z
    );
    const float3 sr2 = float3(
        s00 * r2.x + s01 * r2.y + s02 * r2.z,
        s10 * r2.x + s11 * r2.y + s12 * r2.z,
        s20 * r2.x + s21 * r2.y + s22 * r2.z
    );

    const float c00 = dot(r0, sr0);
    const float c01 = dot(r0, sr1);
    const float c02 = dot(r0, sr2);
    const float c10 = dot(r1, sr0);
    const float c11 = dot(r1, sr1);
    const float c12 = dot(r1, sr2);
    const float c20 = dot(r2, sr0);
    const float c21 = dot(r2, sr1);
    const float c22 = dot(r2, sr2);

    const float inv_z2 = inv_z * inv_z;
    const float3 j0 = float3(fx * inv_z, 0.0f, -fx * x * inv_z2);
    const float3 j1 = float3(0.0f, fy * inv_z, -fy * y * inv_z2);

    const float3 cj0 = float3(
        c00 * j0.x + c01 * j0.y + c02 * j0.z,
        c10 * j0.x + c11 * j0.y + c12 * j0.z,
        c20 * j0.x + c21 * j0.y + c22 * j0.z
    );
    const float3 cj1 = float3(
        c00 * j1.x + c01 * j1.y + c02 * j1.z,
        c10 * j1.x + c11 * j1.y + c12 * j1.z,
        c20 * j1.x + c21 * j1.y + c22 * j1.z
    );

    float a = dot(j0, cj0);
    float b = 0.5f * (dot(j0, cj1) + dot(j1, cj0));
    float d = dot(j1, cj1);

    if (!isfinite(a) || !isfinite(b) || !isfinite(d)) {
        tile_counts[gid] = 0;
        return;
    }

    // Analytic symmetric 2x2 eigendecomposition. Clamp eigenvalues, then
    // reconstruct sigma_uv in screen coordinates.
    const float midpoint = 0.5f * (a + d);
    const float radius = sqrt(
        0.25f * (a - d) * (a - d) + b * b
    );
    const float lambda_min = clamp(midpoint - radius, 1e-6f, 1e4f);
    const float lambda_max = clamp(midpoint + radius, 1e-6f, 1e4f);

    const float theta = 0.5f * atan2(2.0f * b, a - d);
    const float cs = cos(theta);
    const float sn = sin(theta);

    a = lambda_min * sn * sn + lambda_max * cs * cs;
    b = (lambda_max - lambda_min) * cs * sn;
    d = lambda_min * cs * cs + lambda_max * sn * sn;

    if (!isfinite(a) || !isfinite(b) || !isfinite(d)) {
        tile_counts[gid] = 0;
        return;
    }

    // Rectangular axis-aligned 3-sigma AABB. This is the optimization that
    // previously lived in the closed AABB PR.
    const int radius_u = int(ceil(3.0f * sqrt(clamp(a, 1e-12f, 1e4f))));
    const int radius_v = int(ceil(3.0f * sqrt(clamp(d, 1e-12f, 1e4f))));

    int umin = int(floor(u - float(radius_u)));
    int umax = int(floor(u + float(radius_u)));
    int vmin = int(floor(v - float(radius_v)));
    int vmax = int(floor(v + float(radius_v)));

    if (umax < 0 || umin >= W || vmax < 0 || vmin >= H) {
        tile_counts[gid] = 0;
        return;
    }

    umin = clamp(umin, 0, W - 1);
    umax = clamp(umax, 0, W - 1);
    vmin = clamp(vmin, 0, H - 1);
    vmax = clamp(vmax, 0, H - 1);

    const int umin_tile = umin / tile_size;
    const int umax_tile = umax / tile_size;
    const int vmin_tile = vmin / tile_size;
    const int vmax_tile = vmax / tile_size;

    const int n_u = umax_tile - umin_tile + 1;
    const int n_v = vmax_tile - vmin_tile + 1;
    const int count = n_u * n_v;

    const float det = a * d - b * b;
    const float safe_det = max(det, 1e-12f);
    const float A11 = max(d / safe_det, min_conis);
    const float A12 = -b / safe_det;
    const float A22 = max(a / safe_det, min_conis);

    const float opacity = clamp(
        1.0f / (1.0f + exp(-opacity_raw[gid])),
        0.0f,
        0.999f
    );

    const uint cbase = gid * 3;
    const uint gbase = gid * 9;
    gaussians[gbase] = u;
    gaussians[gbase + 1] = v;
    gaussians[gbase + 2] = colors[cbase];
    gaussians[gbase + 3] = colors[cbase + 1];
    gaussians[gbase + 4] = colors[cbase + 2];
    gaussians[gbase + 5] = opacity;
    gaussians[gbase + 6] = A11;
    gaussians[gbase + 7] = A12;
    gaussians[gbase + 8] = A22;

    const uint bbase = gid * 4;
    tile_bounds[bbase] = umin_tile;
    tile_bounds[bbase + 1] = umax_tile;
    tile_bounds[bbase + 2] = vmin_tile;
    tile_bounds[bbase + 3] = vmax_tile;
    tile_counts[gid] = count;
}
