import torch

def get_rays(H: int, W: int, focal: float, c2w: torch.Tensor, device=None):
    """
    Generate ray origins and directions for one camera pose.
    - H, W: image size
    - focal: scalar focal length (pixels)
    - c2w: (4,4) camera-to-world
    Returns:
      rays_o: (H*W, 3)
      rays_d: (H*W, 3) normalized
    Convention matches TinyNeRF tutorials: camera looks along -z.
    """
    device = device or c2w.device
    i, j = torch.meshgrid(
        torch.arange(W, device=device),
        torch.arange(H, device=device),
        indexing="xy",
    )
    # pixel coords to camera directions
    dirs = torch.stack([
        (i - W * 0.5) / focal,
        -(j - H * 0.5) / focal,
        -torch.ones_like(i, dtype=torch.float32, device=device)
    ], dim=-1).to(torch.float32)  # (H, W, 3)

    # Rotate and translate: rays_d = (R @ dirs), rays_o = t
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    rays_d = torch.matmul(dirs.reshape(-1, 3), R.T)  # (H*W,3)
    rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
    rays_o = t.expand_as(rays_d)                     # (H*W,3)
    return rays_o, rays_d
