# 🧠 Tiny NeRF (PyTorch) — From Coordinates to 3D Scene

This project is a **from-scratch reimplementation of TinyNeRF**, a compact version of the original **Neural Radiance Fields (NeRF)** model, written in **PyTorch** and designed for educational purposes.

It’s meant to be **readable, hackable, and extendable** — especially for researchers who want to understand how coordinate-based neural representations work (e.g., for robotics, affordances, implicit fields, or 3D perception).

---

## 🚀 What Is NeRF?

**NeRF** stands for **Neural Radiance Fields** — a model that learns to represent a 3D scene as a continuous function.

Instead of storing a 3D grid of voxels or point clouds, NeRF trains an **MLP (Multi-Layer Perceptron)** to learn a mapping:

\[
F_\theta: (x, y, z) \rightarrow (r, g, b, \sigma)
\]

where:
- **(x, y, z)** → 3D coordinate in space  
- **(r, g, b)** → color at that point  
- **σ (sigma)** → density or opacity (how much light is absorbed there)

By sampling many points along **camera rays**, and integrating the results using **volume rendering**, we can “render” realistic novel views of a scene — even from unseen viewpoints.

---

## 🧩 Project Structure

tiny-nerf-pytorch/
├─ data/ # dataset (tiny_nerf_data.npz)
├─ outputs/ # previews, final renders, gifs
├─ checkpoints/ # model checkpoints
├─ scripts/
│ └─ get_data.sh # downloads dataset
├─ src/
│ ├─ main.py # simple test render (no training)
│ ├─ train.py # training loop + PSNR logging
│ ├─ data.py # load .npz dataset
│ ├─ encoding.py # positional encoding (Fourier features)
│ ├─ nerf.py # MLP definition
│ ├─ rays.py # ray generation (camera model)
│ ├─ sampling.py # stratified sampling along rays
│ ├─ volume.py # differentiable volume rendering
│ ├─ utils.py # helper functions (e.g., PSNR)
│ ├─ camera.py # generate spiral camera paths
│ └─ make_gif.py # render novel-view GIFs


---

## 🛠️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/avihaig/tiny-nerf-pytorch.git
cd tiny-nerf-pytorch
