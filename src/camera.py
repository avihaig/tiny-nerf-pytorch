import torch, math

# builds a simple path around a reference camera
def spiral_poses(c2w_ref: torch.Tensor, n_frames: int = 60, radius: float = 0.3):
    device = c2w_ref.device
    out = []
    for t in torch.linspace(0, 2*math.pi, n_frames, device=device):
        c = torch.cos(t); s = torch.sin(t)
        T = torch.eye(4, device=device, dtype=c2w_ref.dtype)
        T[:3, 3] = torch.tensor([radius*c, radius*s, 0.0], device=device, dtype=c2w_ref.dtype)
        out.append(c2w_ref @ T)
    return torch.stack(out, dim=0)
