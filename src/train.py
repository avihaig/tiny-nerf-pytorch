import os, time
from dataclasses import dataclass
from typing import Optional

import imageio.v2 as imageio
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import tyro

from data import load_tiny_nerf_npz
from encoding import PositionalEncoding
from nerf import TinyNeRF
from rays import get_rays
from sampling import stratified_samples
from volume import volume_render
from utils import mse2psnr

@dataclass
class Config:
    iters: int = 20000  # total training iterations/steps
    n_rand: int = 2048  # number of random pixels/rays sampled per step
    n_samples: int = 64 # number of samples along each ray
    lr: float = 5e-4
    near: float = 2.0
    far: float = 6.0
    log_every: int = 50
    preview_every: int = 500
    ckpt_every: int = 1000  # save checkpoint every N steps
    ckpt_path: str = "checkpoints/tinynerf_latest.pth"
    out_dir: str = "outputs"
    resume: bool = True
    preview_pose: Optional[int] = None  # if None, use (step+1)%N

@torch.no_grad()
def render_one(model: nn.Module,
               encoder: nn.Module,
               H: int, W: int, focal: float,
               pose: torch.Tensor,
               device: torch.device,
               n_samples: int = 64,
               near: float = 2.0, far: float = 6.0,
               chunk: int = 8192) -> torch.Tensor:
    model.eval()
    rays_o, rays_d = get_rays(H, W, focal, pose.to(device), device=device)
    out_rgb = []
    for i in range(0, rays_o.shape[0], chunk):
        ro = rays_o[i:i+chunk]
        rd = rays_d[i:i+chunk]
        z_vals, pts = stratified_samples(near, far, n_samples, ro, rd, randomized=False)
        xenc = encoder(pts.reshape(-1, 3))
        rgb, sigma = model(xenc)
        rgb = rgb.reshape(pts.shape[0], n_samples, 3)
        sigma = sigma.reshape(pts.shape[0], n_samples, 1)
        comp_rgb, _, _, _ = volume_render(rgb, sigma, z_vals, rd)
        out_rgb.append(comp_rgb)
    img = torch.cat(out_rgb, dim=0).reshape(H, W, 3).clamp(0., 1.)
    return img

def main(cfg: Config):
    # Setup
    torch.manual_seed(0); np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(cfg.ckpt_path), exist_ok=True)
    print(f"[device] {device} torch={torch.__version__}")

    # Data
    d = load_tiny_nerf_npz("data/tiny_nerf_data.npz")
    images = torch.from_numpy(d["images"]).to(device)  # (N,H,W,3)
    poses  = torch.from_numpy(d["poses"]).to(device)   # (N,4,4)
    focal  = float(d["focal"])
    N, H, W, _ = images.shape
    print(f"[data] N={N} H={H} W={W} focal={focal:.2f}")

    # Model
    encoder = PositionalEncoding(num_freqs=10, include_input=True).to(device)
    model = TinyNeRF(in_dim=encoder.out_dim, hidden=128, depth=4, skip_at=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Resume
    start_step = 0
    if cfg.resume and os.path.exists(cfg.ckpt_path):
        ckpt = torch.load(cfg.ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "opt" in ckpt:
            optimizer.load_state_dict(ckpt["opt"])
        if "step" in ckpt:
            start_step = int(ckpt["step"])
        print(f"[resume] loaded {cfg.ckpt_path} from step {start_step}")

    # Precompute rays
    all_rays_o, all_rays_d = [], []
    for i in range(N):
        ro, rd = get_rays(H, W, focal, poses[i], device=device)
        all_rays_o.append(ro); all_rays_d.append(rd)
    all_rays_o = torch.stack(all_rays_o, dim=0)  # (N, HW, 3)
    all_rays_d = torch.stack(all_rays_d, dim=0)  # (N, HW, 3)
    pixels = images.view(N, H*W, 3)

    # Train
    t0 = time.time()
    pbar = tqdm(range(start_step, cfg.iters), desc="train")
    for step in pbar:
        model.train()
        img_i = step % N
        inds = torch.randint(0, H*W, (cfg.n_rand,), device=device)
        ro = all_rays_o[img_i, inds]
        rd = all_rays_d[img_i, inds]
        target = pixels[img_i, inds]

        z_vals, pts = stratified_samples(cfg.near, cfg.far, cfg.n_samples, ro, rd, randomized=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            xenc = encoder(pts.reshape(-1, 3))
            rgb, sigma = model(xenc)
            rgb = rgb.reshape(cfg.n_rand, cfg.n_samples, 3)
            sigma = sigma.reshape(cfg.n_rand, cfg.n_samples, 1)
            comp_rgb, _, _, _ = volume_render(rgb, sigma, z_vals, rd)
            loss = torch.mean((comp_rgb - target) ** 2)
            psnr = mse2psnr(loss)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if (step + 1) % cfg.log_every == 0:
            pbar.set_postfix(loss=float(loss.item()), psnr=float(psnr.item()))

        if (step + 1) % cfg.preview_every == 0:
            with torch.no_grad():
                pose_idx = img_i + 1 if cfg.preview_pose is None else cfg.preview_pose
                pose_idx %= N
                img = render_one(model, encoder, H, W, focal, poses[pose_idx], device,
                                 n_samples=cfg.n_samples, near=cfg.near, far=cfg.far)
                out_path = f"{cfg.out_dir}/preview_{step+1:06d}.png"
                imageio.imwrite(out_path, (img.cpu().numpy() * 255).astype(np.uint8))

        if (step + 1) % cfg.ckpt_every == 0:
            torch.save({"model": model.state_dict(),
                        "opt": optimizer.state_dict(),
                        "step": step + 1,
                        "in_dim": encoder.out_dim,
                        "cfg": dict(hidden=128, depth=4, skip_at=2)},
                       cfg.ckpt_path)

    dt = time.time() - t0
    torch.save({"model": model.state_dict(),
                "opt": optimizer.state_dict(),
                "step": cfg.iters,
                "in_dim": encoder.out_dim,
                "cfg": dict(hidden=128, depth=4, skip_at=2)},
               cfg.ckpt_path)
    img = render_one(model, encoder, H, W, focal, poses[-1], device,
                     n_samples=cfg.n_samples, near=cfg.near, far=cfg.far)
    imageio.imwrite(f"{cfg.out_dir}/final.png", (img.cpu().numpy() * 255).astype(np.uint8))
    print(f"[done] {cfg.iters} iters in {dt/60:.2f} min | saved {cfg.ckpt_path} and {cfg.out_dir}/final.png")

if __name__ == "__main__":
    cfg = tyro.cli(Config)  # <-- parse dataclass at top-level, so flags are --iters, --n-rand, ...
    main(cfg)
