import torch

from util import eigh_2x2, inv2x2, project_points


def gaussian_rasterization_pytorch(
    pos,
    color,
    opacity_raw,
    sigma,
    c2w,
    H,
    W,
    fx,
    fy,
    cx,
    cy,
    near=2e-3,
    far=100.0,
    pix_guard=64,
    T=16,
    min_conis=1e-6,
    chi_square_clip=9.21,
    alpha_max=0.99,
    alpha_cutoff=1 / 255.0,
):
    """PyTorch reference renderer used only to validate the Metal experiment."""
    device = pos.device
    dt = pos.dtype

    uv, x, y, z = project_points(pos, c2w, H, W, fx, fy, cx, cy)
    u, v = uv[:, 0], uv[:, 1]

    frustum = (
        (u > -pix_guard)
        & (u < W + pix_guard)
        & (v > -pix_guard)
        & (v < H + pix_guard)
        & (z > near)
        & (z < far)
    )

    uv = uv[frustum]
    color = color[frustum]
    opacity = torch.sigmoid(opacity_raw[frustum]).squeeze(-1).clamp(0, 0.999)
    x = x[frustum]
    y = y[frustum]
    z = z[frustum]
    sigma = sigma[frustum]

    Rcw = c2w[:3, :3]
    Rwc = Rcw.T

    J = torch.zeros((uv.shape[0], 2, 3), device=device, dtype=dt)
    J[:, 0, 0] = fx / z
    J[:, 1, 1] = fy / z
    J[:, 0, 2] = -fx * x / (z * z)
    J[:, 1, 2] = -fy * y / (z * z)

    sigma_camera = Rwc.unsqueeze(0) @ sigma @ Rwc.T.unsqueeze(0)
    sigma_uv = J @ sigma_camera @ J.transpose(1, 2)
    sigma_uv = 0.5 * (sigma_uv + sigma_uv.transpose(1, 2))

    evals, evecs = eigh_2x2(sigma_uv)
    evals = torch.clamp(evals, min=1e-6, max=1e4)
    sigma_uv = (
        evecs
        @ torch.diag_embed(evals)
        @ evecs.transpose(1, 2)
    )

    keep = torch.isfinite(
        sigma_uv.reshape(sigma_uv.shape[0], -1)
    ).all(dim=-1)

    uv = uv[keep]
    color = color[keep]
    opacity = opacity[keep]
    z = z[keep]
    sigma_uv = sigma_uv[keep]

    z, order = torch.sort(z, descending=False)
    uv = uv[order]
    color = color[order]
    opacity = opacity[order]
    sigma_uv = sigma_uv[order]

    u = uv[:, 0]
    v = uv[:, 1]

    # Rectangular 3-sigma AABB, matching the Metal setup kernel.
    variance_u = sigma_uv[:, 0, 0].clamp_min(1e-12).clamp_max(1e4)
    variance_v = sigma_uv[:, 1, 1].clamp_min(1e-12).clamp_max(1e4)
    radius_u = torch.ceil(3.0 * torch.sqrt(variance_u)).to(torch.int64)
    radius_v = torch.ceil(3.0 * torch.sqrt(variance_v)).to(torch.int64)

    umin = torch.floor(u - radius_u).to(torch.int64)
    umax = torch.floor(u + radius_u).to(torch.int64)
    vmin = torch.floor(v - radius_v).to(torch.int64)
    vmax = torch.floor(v + radius_v).to(torch.int64)

    on_screen = (
        (umax >= 0)
        & (umin < W)
        & (vmax >= 0)
        & (vmin < H)
    )
    if not on_screen.any():
        raise RuntimeError("there are no gaussians on screen")

    u, v = u[on_screen], v[on_screen]
    color = color[on_screen]
    opacity = opacity[on_screen]
    sigma_uv = sigma_uv[on_screen]
    umin, umax = umin[on_screen], umax[on_screen]
    vmin, vmax = vmin[on_screen], vmax[on_screen]

    umin = umin.clamp(0, W - 1)
    umax = umax.clamp(0, W - 1)
    vmin = vmin.clamp(0, H - 1)
    vmax = vmax.clamp(0, H - 1)

    umin_tile = (umin // T).to(torch.int64)
    umax_tile = (umax // T).to(torch.int64)
    vmin_tile = (vmin // T).to(torch.int64)
    vmax_tile = (vmax // T).to(torch.int64)

    n_u = umax_tile - umin_tile + 1
    n_v = vmax_tile - vmin_tile + 1
    num_tiles_per_gaussian = n_u * n_v

    num_gaussians = umin_tile.shape[0]
    K = int(num_tiles_per_gaussian.sum().item())

    gaussian_ids = torch.repeat_interleave(
        torch.arange(num_gaussians, device=device, dtype=torch.int64),
        num_tiles_per_gaussian,
        output_size=K,
    )

    starts = torch.cumsum(num_tiles_per_gaussian, dim=0)
    starts = starts - num_tiles_per_gaussian

    local_tile_ids = (
        torch.arange(K, device=device, dtype=torch.int64)
        - starts[gaussian_ids]
    )
    local_tile_u = local_tile_ids // n_v[gaussian_ids]
    local_tile_v = local_tile_ids % n_v[gaussian_ids]

    flat_tile_u = umin_tile[gaussian_ids] + local_tile_u
    flat_tile_v = vmin_tile[gaussian_ids] + local_tile_v
    num_tiles_u = (W + T - 1) // T
    flat_tile_id = flat_tile_v * num_tiles_u + flat_tile_u

    tile_ids_1d, perm = torch.sort(flat_tile_id, stable=True)
    gaussian_ids = gaussian_ids[perm]

    unique_tile_ids, counts = torch.unique_consecutive(
        tile_ids_1d,
        return_counts=True,
    )
    start = torch.zeros_like(unique_tile_ids)
    start[1:] = torch.cumsum(counts[:-1], dim=0)
    end = start + counts

    inverse_covariance = inv2x2(sigma_uv)
    inverse_covariance[:, 0, 0] = torch.clamp(
        inverse_covariance[:, 0, 0],
        min=min_conis,
    )
    inverse_covariance[:, 1, 1] = torch.clamp(
        inverse_covariance[:, 1, 1],
        min=min_conis,
    )

    final_image = torch.zeros((H * W, 3), device=device, dtype=dt)

    for tile_id, s0, s1 in zip(
        unique_tile_ids.tolist(),
        start.tolist(),
        end.tolist(),
    ):
        txi = tile_id % num_tiles_u
        tyi = tile_id // num_tiles_u

        tile_gaussian_ids = gaussian_ids[s0:s1]
        x0, y0 = txi * T, tyi * T
        x1 = min((txi + 1) * T, W)
        y1 = min((tyi + 1) * T, H)
        if x0 >= x1 or y0 >= y1:
            continue

        xs = torch.arange(x0, x1, device=device, dtype=dt)
        ys = torch.arange(y0, y1, device=device, dtype=dt)
        pu, pv = torch.meshgrid(xs, ys, indexing="xy")
        px_u = pu.reshape(-1)
        px_v = pv.reshape(-1)

        pixel_idx = (px_v * W + px_u).to(torch.int64)

        gaussian_u = u[tile_gaussian_ids]
        gaussian_v = v[tile_gaussian_ids]
        gaussian_color = color[tile_gaussian_ids]
        gaussian_opacity = opacity[tile_gaussian_ids]
        conic = inverse_covariance[tile_gaussian_ids]

        du = px_u.unsqueeze(0) - gaussian_u.unsqueeze(-1)
        dv = px_v.unsqueeze(0) - gaussian_v.unsqueeze(-1)

        A11 = conic[:, 0, 0].unsqueeze(-1)
        A12 = conic[:, 0, 1].unsqueeze(-1)
        A22 = conic[:, 1, 1].unsqueeze(-1)
        q = A11 * du * du + 2 * A12 * du * dv + A22 * dv * dv

        inside = q <= chi_square_clip
        g = torch.exp(-0.5 * torch.clamp(q, max=chi_square_clip))
        g = torch.where(inside, g, torch.zeros_like(g))

        alpha = (gaussian_opacity.unsqueeze(-1) * g).clamp_max(alpha_max)
        alpha = torch.where(
            alpha >= alpha_cutoff,
            alpha,
            torch.zeros_like(alpha),
        )

        one_minus_alpha = 1 - alpha
        transmittance = torch.cumprod(one_minus_alpha, dim=0)
        transmittance = torch.concatenate(
            [
                torch.ones(
                    (1, alpha.shape[-1]),
                    device=device,
                    dtype=dt,
                ),
                transmittance[:-1],
            ],
            dim=0,
        )

        weights = alpha * transmittance
        tile_color = (
            weights.unsqueeze(-1)
            * gaussian_color.unsqueeze(1)
        ).sum(dim=0)

        final_image[pixel_idx] = tile_color

    return final_image.reshape((H, W, 3)).clamp(0, 1)
