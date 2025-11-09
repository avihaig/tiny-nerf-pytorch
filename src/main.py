import time
import os
import imageio.v2 as imageio
import numpy as np
import torch

from data import load_tiny_nerf_npz
from encoding import PositionalEncoding
from nerf import TinyNeRF
from rays import get_rays
from sampling import stratified_samples
from volume import volume_render

@torch.no_grad()
def test_render_once(model, encoder, H, W, focal, pose, device, n_samples=64, near=2.0, far=6.0, chunk=8192):
    model.eval()
    # Rays
    rays_o, rays_d = get_rays(H, W, focal, pose.to(device), device=device)  # (HW,3)
    # Batched for memory
    all_rgb = []
    for i in range(0, rays_o.shape[0], chunk):
        ro = rays_o[i:i+chunk]
        rd = rays_d[i:i+chunk]
        z_vals, pts = stratified_samples(near, far, n_samples, ro, rd, randomized=False)
        # Encode xyz
        xenc = encoder(pts.reshape(-1, 3))
        # MLP
        rgb, sigma = model(xenc)
        rgb = rgb.reshape(pts.shape[0], n_samples, 3)
        sigma = sigma.reshape(pts.shape[0], n_samples, 1)
        comp_rgb, _, _, _ = volume_render(rgb, sigma, z_vals, rd)
        all_rgb.append(comp_rgb)
    img = torch.cat(all_rgb, dim=0).reshape(H, W, 3).clamp(0., 1.)
    return img

def main():
    # ----- Device & seeds -----
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device} torch={torch.__version__}")

    # ----- Load data -----
    d = load_tiny_nerf_npz("data/tiny_nerf_data.npz")
    images = torch.from_numpy(d["images"])  # (N, H, W, 3)
    poses  = torch.from_numpy(d["poses"])   # (N, 4, 4)
    focal  = float(d["focal"])
    N, H, W, _ = images.shape
    print(f"[data] N={N} H={H} W={W} focal={focal:.2f}")

    # ----- Create encoder & model -----
    encoder = PositionalEncoding(num_freqs=10, include_input=True).to(device)
    model = TinyNeRF(in_dim=encoder.out_dim, hidden=128, depth=4, skip_at=2).to(device)

    # ----- Test render one pose (index 0) -----
    os.makedirs("outputs", exist_ok=True)
    t0 = time.time()
    img = test_render_once(model, encoder, H, W, focal, poses[0], device, n_samples=64, near=2.0, far=6.0, chunk=8192)
    dt = time.time() - t0
    out_path = "outputs/preview.png"
    imageio.imwrite(out_path, (img.cpu().numpy() * 255).astype(np.uint8))
    print(f"[render] wrote {out_path} in {dt:.2f}s (untrained model; expect noisy image)")

if __name__ == "__main__":
    main()
