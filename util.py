import torch

def project_points(pc, c2w, H, W, fx, fy, cx, cy):
    w2c = torch.eye(4)
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    w2c[:3, :3] = R.T
    w2c[:3, 3] = -R.T @ t

    # 4, 4 @ 4, N -> 4, N -> N, 4
    pcc = ((w2c @ torch.concatenate([pc, torch.ones_like(pc[:, :1])], dim=-1).T).T)[:, :3]

    # camera space
    x = pcc[:, 0]
    y = pcc[:, 1]
    z = pcc[:, 2]

    uv = torch.stack([
        fx * x / z + cx,
        fy * y / z + cy
    ], dim=-1)

    return uv, x, y, z