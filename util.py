from pathlib import Path
import numpy as np
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


def quat_to_mat(quat: torch.Tensor):
    """
    :param quat: [N, 4]
    """
    x, y, z, w = quat.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w

    return torch.stack([
        1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw,
        2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw,
        2 * xz - 2 * yw, 2 * yz  + 2 * xw, 1 - 2 * xx  - 2 * yy 
    ], dim=-1).reshape(quat.shape[:-1] + (3, 3))


def scale_intrinsics(H, W, H_src, W_src, fx, fy, cx, cy):
    scale_x = W / W_src
    scale_y = H / H_src
    fx_scaled = fx * scale_x
    fy_scaled = fy * scale_y
    cx_scaled = cx * scale_x
    cy_scaled = cy * scale_y
    return fx_scaled, fy_scaled, cx_scaled, cy_scaled


def load_cameras(cameras_path, images_root, device="cpu", dtype=torch.float32):
    cams = np.load(cameras_path, allow_pickle=True)
    cams = sorted(cams, key=lambda x: x['id'])
    
    c2ws = []
    images_paths = []
    
    for cam in cams:
        quat = torch.from_numpy(cam['q'])
        R = quat_to_mat(quat)
        T = torch.from_numpy(cam['t'])
        
        c2w = torch.eye(4, device=device, dtype=dtype)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ T
        c2ws.append(c2w)
        
        image_path = Path(images_root) / cam['name']
        images_paths.append(image_path)
        
    return c2ws, images_paths


def inv2x2(M, eps=1e-12):
    """
    :param M: [N, 2, 2]
    """
    a = M[:, 0, 0]
    b = M[:, 0, 1]
    c = M[:, 1, 0]
    d = M[:, 1, 1]
    det = a * d - b * c
    safe_det = torch.clamp(det, min=eps)
    
    inv = torch.empty_like(M)
    inv[:, 0, 0] = d / safe_det
    inv[:, 0, 1] = -b / safe_det
    inv[:, 1, 0] = -c / safe_det
    inv[:, 1, 1] = a / safe_det
    return inv