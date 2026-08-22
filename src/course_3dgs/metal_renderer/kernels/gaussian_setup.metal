#include <metal_stdlib>
using namespace metal;

#define SH_LEVELS __SH_LEVELS__
#define SH_COEFFICIENT_COUNT (SH_LEVELS * SH_LEVELS)

inline float3 load_sh_coefficient(
    const device float* sh_coefficients,
    uint gaussian_id,
    uint coefficient_id
) {
    const uint base =
        (gaussian_id * SH_COEFFICIENT_COUNT + coefficient_id) * 3;
    return float3(
        sh_coefficients[base],
        sh_coefficients[base + 1],
        sh_coefficients[base + 2]
    );
}

inline float3 evaluate_sh_color(
    const device float* sh_coefficients,
    uint gaussian_id,
    float3 view_direction
) {
    const float x = view_direction.x;
    const float y = view_direction.y;
    const float z = view_direction.z;

    float3 value =
        load_sh_coefficient(sh_coefficients, gaussian_id, 0)
        * 0.28209479177387814f;

#if SH_LEVELS >= 2
    value += load_sh_coefficient(sh_coefficients, gaussian_id, 1)
        * (-0.4886025119029199f * y);
    value += load_sh_coefficient(sh_coefficients, gaussian_id, 2)
        * (+0.4886025119029199f * z);
    value += load_sh_coefficient(sh_coefficients, gaussian_id, 3)
        * (-0.4886025119029199f * x);
#endif

#if SH_LEVELS >= 3
    const float x2 = x * x;
    const float y2 = y * y;
    const float z2 = z * z;
    const float xy = x * y;
    const float yz = y * z;
    const float xz = x * z;
    const float x2_minus_y2 = x2 - y2;

    value += load_sh_coefficient(sh_coefficients, gaussian_id, 4)
        * (+1.0925484305920792f * xy);
    value += load_sh_coefficient(sh_coefficients, gaussian_id, 5)
        * (+1.0925484305920792f * yz);
    value += load_sh_coefficient(sh_coefficients, gaussian_id, 6)
        * (+0.31539156525252005f * (3.0f * z2 - 1.0f));
    value += load_sh_coefficient(sh_coefficients, gaussian_id, 7)
        * (+1.0925484305920792f * xz);
    value += load_sh_coefficient(sh_coefficients, gaussian_id, 8)
        * (+0.5462742152960396f * x2_minus_y2);
#endif

#if SH_LEVELS >= 4
    const float four_z2_minus_x2_minus_y2 = 4.0f * z2 - x2 - y2;

    value += load_sh_coefficient(sh_coefficients, gaussian_id, 9)
        * (+0.5900435899266435f * y * (3.0f * x2 - y2));
    value += load_sh_coefficient(sh_coefficients, gaussian_id, 10)
        * (+2.890611442640554f * xy * z);
    value += load_sh_coefficient(sh_coefficients, gaussian_id, 11)
        * (+0.4570457994644658f * y * four_z2_minus_x2_minus_y2);
    value += load_sh_coefficient(sh_coefficients, gaussian_id, 12)
        * (+0.3731763325901154f
           * z
           * (2.0f * z2 - 3.0f * x2 - 3.0f * y2));
    value += load_sh_coefficient(sh_coefficients, gaussian_id, 13)
        * (+0.4570457994644658f * x * four_z2_minus_x2_minus_y2);
    value += load_sh_coefficient(sh_coefficients, gaussian_id, 14)
        * (+1.445305721320277f * z * x2_minus_y2);
    value += load_sh_coefficient(sh_coefficients, gaussian_id, 15)
        * (+0.5900435899266435f * x * (x2 - 3.0f * y2));
#endif

    return 1.0f / (1.0f + exp(-value));
}

kernel void project_gaussians(
    const device float* positions [[buffer(0)]],
    const device float* sh_coefficients [[buffer(1)]],
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

    const float midpoint = 0.5f * (a + d);
    const float radius = sqrt(0.25f * (a - d) * (a - d) + b * b);
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

    const float3 view_direction = dp / (length(dp) + 1e-8f);
    const float3 color = evaluate_sh_color(sh_coefficients, gid, view_direction);

    const uint gbase = gid * 9;
    gaussians[gbase] = u;
    gaussians[gbase + 1] = v;
    gaussians[gbase + 2] = color.x;
    gaussians[gbase + 3] = color.y;
    gaussians[gbase + 4] = color.z;
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
