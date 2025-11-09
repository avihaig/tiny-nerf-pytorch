import os, imageio.v2 as imageio, numpy as np, torch
from data import load_tiny_nerf_npz
from encoding import PositionalEncoding
from nerf import TinyNeRF
from camera import spiral_poses
from train import render_one

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = load_tiny_nerf_npz("data/tiny_nerf_data.npz")
    images = torch.from_numpy(d["images"]).to(device)
    poses  = torch.from_numpy(d["poses"]).to(device)
    focal  = float(d["focal"])
    N, H, W, _ = images.shape

    enc = PositionalEncoding(num_freqs=10, include_input=True).to(device)
    ckpt = torch.load("checkpoints/tinynerf_latest.pth", map_location=device)
    model = TinyNeRF(in_dim=enc.out_dim, **ckpt.get("cfg", dict(hidden=128, depth=4, skip_at=2))).to(device)
    model.load_state_dict(ckpt["model"])

    path = spiral_poses(poses[0], n_frames=60, radius=0.3)
    frames = []
    for i, p in enumerate(path):
        img = render_one(model, enc, H, W, focal, p, device, n_samples=64, near=2.0, far=6.0)
        frames.append((img.cpu().numpy()*255).astype(np.uint8))
        print(f"[render] {i+1}/{len(path)}")
    os.makedirs("outputs", exist_ok=True)
    out = "outputs/novel_views.gif"
    imageio.mimsave(out, frames, fps=15, loop=0)
    print(f"[ok] wrote {out}")
if __name__ == "__main__":
    main()
