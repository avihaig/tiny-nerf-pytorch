# 🧠 Tiny NeRF (PyTorch) — From Coordinates to 3D Scene

This project is a **from-scratch reimplementation of TinyNeRF**, a compact version of the original **Neural Radiance Fields (NeRF)** model, written in **PyTorch** and designed for educational purposes.

It’s meant to be **readable, hackable, and extendable** — especially for researchers who want to understand how coordinate-based neural representations work (e.g., for robotics, affordances, implicit fields, or 3D perception).

---

## 🚀 What Is NeRF?

**NeRF** stands for **Neural Radiance Fields** — a model that learns to represent a 3D scene as a continuous function using a neural network.

Instead of storing a discrete 3D grid (like voxels or point clouds), NeRF trains an **MLP (Multi-Layer Perceptron)** to learn a continuous mapping between **spatial coordinates** and **visual properties** of the scene.

---

### 🧠 The Original NeRF (Full Model)

In the original paper (*Mildenhall et al., ECCV 2020*), NeRF learns a function that depends on both **position** and **view direction**:

$$F_\theta(\mathbf{x}, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma)$$

where:
- $\mathbf{x} = (x, y, z)$ → 3D coordinate in world space  
- $\mathbf{d}$ → normalized viewing direction (derived from the camera pose)  
- $\mathbf{c} = (r, g, b)$ → emitted color  
- $\sigma$ → density or opacity at that point  

This dual input allows NeRF to model **view-dependent lighting effects**, such as **reflections, specular highlights**, and subtle changes in appearance as the camera moves.  
It effectively learns a *radiance field* — how light travels through the scene depending on both location and direction.

---

### ⚙️ TinyNeRF (This Implementation)

In this repository, we simplify the formulation to focus on the **core spatial representation**:

$$F_\theta(\mathbf{x}) \rightarrow (\mathbf{c}, \sigma)$$

Here, the network only takes the **3D position** as input — omitting the viewing direction $\mathbf{d}$.  
This simplification makes the model much lighter, faster to train, and easier to understand while still demonstrating all essential NeRF components:
- **Positional encoding** (Fourier features)  
- **Ray sampling and marching**  
- **Differentiable volume rendering**

The trade-off is that our TinyNeRF cannot reproduce **view-dependent effects** like specularities or reflections — colors are fixed per spatial location.  
However, it still reconstructs the overall scene geometry and diffuse appearance accurately, making it ideal as an educational and foundational implementation.

---

### 💡 Why This Simplification

We intentionally start from this reduced version because it captures the **mathematical and conceptual essence of NeRF** — how an MLP can represent a continuous 3D field — without the added complexity of directional conditioning.  
Once this version is understood, it becomes straightforward to extend it to the full formulation by adding **view-direction encoding** and a **split MLP architecture**, exactly as in the original paper.

---

## 🧩 Project Structure

```text
tiny-nerf-pytorch/
├── data/                     # Dataset (tiny_nerf_data.npz)
├── outputs/                  # Rendered previews, final results, and GIFs
├── checkpoints/              # Saved model weights (.pth)
├── scripts/
│   └── get_data.sh           # Download script for dataset
├── src/
│   ├── data.py               # Load the dataset
│   ├── encoding.py           # Positional (Fourier) encoding
│   ├── nerf.py               # Core TinyNeRF MLP model
│   ├── rays.py               # Ray generation from camera poses
│   ├── sampling.py           # Stratified sampling along rays
│   ├── tiny_nerf_min.py      # Single-file, minimal TinyNeRF (read-first version)
│   ├── volume.py             # Volume rendering (alpha compositing)
│   ├── utils.py              # Utility functions (PSNR)
│   ├── train.py              # Main training loop (with Tyro CLI)
│   ├── main.py               # Minimal forward pass / test render
│   ├── camera.py             # (optional) Generate spiral camera paths
│   └── make_gif.py           # (optional) Render a GIF of novel views
├── requirements.txt          # Dependencies
├── README.md                 # This file 🙂
└── LICENSE

````

---

## 🧠 The Intuition Behind NeRF (Simplified)

Imagine you have several 2D images of a 3D object (say, a Lego or a robot arm) taken from different angles.  
What if we could train a neural network to **remember** the entire scene — not as pixels, but as a function that tells you:

> “If I look at coordinate (x, y, z), what color and density should I see?”

That’s **NeRF**. It learns a continuous representation of a 3D scene using a **neural field**.

---

## 🧮 Core Equations (in simple terms)

### 1. The Scene Function
$$F_\theta: (x, y, z) \rightarrow (r, g, b, \sigma)$$
An MLP parameterized by θ maps each 3D coordinate to:
- RGB color
- Density (σ)

---

### 2. Positional Encoding (Fourier Features)
Small networks can’t easily learn fine spatial detail, so we map coordinates to a higher frequency space:

$$\gamma(x) = [\sin(2^0 x), \cos(2^0 x), \sin(2^1 x), \cos(2^1 x), \dots, \sin(2^{L-1} x), \cos(2^{L-1} x)]$$

This lets the network represent high-frequency variations (sharp edges, textures).

---

### 3. Ray Marching (Sampling Along Rays)
For each camera ray, we sample **N** points between a near and far bound:
$$\mathbf{x}_i = \mathbf{o} + t_i \mathbf{d}, \quad t_i \in [t_{near}, t_{far}]$$
where:
- **o** = ray origin (camera center)
- **d** = ray direction
- **tᵢ** = distance along the ray

We query the network at each sampled point to get colors and densities.

---

### 4. Volume Rendering (Compositing)
We combine these samples along the ray using the **volume rendering equation**:

$$C(\mathbf{r}) = \sum_i T_i \, (1 - e^{-\sigma_i \delta_i}) \, c_i$$
where:
- $T_i = \prod_{j < i} e^{-\sigma_j \delta_j}$ = accumulated transmittance  
- $\delta_i$ = distance between samples  
- $c_i$ = predicted color  
- $\sigma_i$ = predicted density


This gives us the final pixel color \( C(\mathbf{r}) \).

---

### 5. Optimization (Training)
We train the network by minimizing mean-squared error (MSE) between rendered colors and ground truth pixels:
$$\mathcal{L} = \frac{1}{N} \sum_{\text{pixels}} || C_\theta(\mathbf{r}) - C_{gt}(\mathbf{r}) ||^2$$

We also report **PSNR (Peak Signal-to-Noise Ratio)**:
$$\text{PSNR} = -10 \log_{10}(\text{MSE})$$

---

## 📈 Understanding PSNR (Image Quality Metric)

**PSNR (Peak Signal-to-Noise Ratio)** is the main quantitative metric used in the original NeRF paper to measure how close a rendered image is to the real one.

It compares the **pixel-wise error** between the predicted image $\hat{I}$ and the ground truth $I$ using the **Mean Squared Error (MSE)**:

$$
\text{MSE} = \frac{1}{N} \sum_i (I_i - \hat{I}_i)^2
$$

From that, PSNR is computed as:

$$
\text{PSNR} = 10 \log_{10} \left( \frac{1}{\text{MSE}} \right)
$$

---

### 🧠 Intuition
- **MSE** measures how far off your reconstruction is — smaller is better.  
- **PSNR** converts that error into decibels (dB): higher PSNR means better visual quality.  
- Each **+1 dB** usually corresponds to a noticeably clearer or more accurate image.

---

### 📊 Typical Ranges

| PSNR (dB) | Interpretation |
|------------|----------------|
| 15–20 | Blurry or poor reconstruction |
| 20–25 | Reasonable quality |
| 25–30 | Good reconstruction |
| 30 +  | Excellent, nearly indistinguishable from ground truth |

---

### 💡 Why NeRF Uses PSNR
The original NeRF paper reports PSNR as a **simple, interpretable measure** of image fidelity.  
It’s easy to compute and lets researchers **compare model performance** across datasets and NeRF variants in a consistent way.

---


## 🧱 Step-by-Step Pipeline

1. **Load data**: RGB images + camera poses + focal length.  
2. **Generate rays**: For each pixel, compute a ray (origin, direction).  
3. **Sample points**: Along each ray, stratified between near/far bounds.  
4. **Encode positions**: Apply Fourier positional encoding to 3D coords.  
5. **Predict**: Use the MLP to predict RGB + σ at each point.  
6. **Render**: Integrate along the ray with the volume rendering formula.  
7. **Compare**: Compute loss against ground-truth pixel color.  
8. **Train**: Backpropagate and update weights.  
9. **Render**: After training, render novel views for evaluation.

---

## ⚙️ How to Run Everything

### 1️⃣ Environment Setup
```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
````

### 2️⃣ Get the Data

```bash
./scripts/get_data.sh
```

### 3️⃣ Test Your GPU & Data

```bash
python src/main.py
```

You should see CUDA info + dataset shapes.

### 4️⃣ Train the Model

```bash
python src/train.py --iters 8000 --n-rand 2048 --n-samples 64 --preview-every 1000
```

This will:

* Train the TinyNeRF MLP
* Save checkpoints in `checkpoints/`
* Save intermediate renders in `outputs/`

### 5️⃣ Render a Novel-View GIF

```bash
python src/make_gif.py
```

You’ll get `outputs/novel_views.gif` — a camera flying around the reconstructed scene.

---
### ⚡️ TinyNeRF — Minimal Single-File Version

If you prefer a **super small, linear script** to learn from before diving into the full modular code, use:

- `src/tiny_nerf_min.py` — a self-contained TinyNeRF with the same algorithmic steps (rays → samples → encode → MLP → volume render → loss) but fewer abstractions and minimal dependencies.

**When to use this version**
- You want to **read everything in one file** with top-to-bottom execution.
- You’re teaching or learning and want **maximum readability** with fewer moving parts.
- You don’t need the extras (Tyro CLI, GIF rendering, camera path utilities, etc.).

**What’s different vs the full repo**
- ✅ Same math: positional encoding, stratified sampling, volume rendering, MSE/PSNR.
- ✅ Same training recipe (Adam, mixed precision optional, preview renders).
- 🧩 Fewer modules (no separate `rays.py`, `volume.py`, etc.—the logic is inline).
- 🧪 Fewer knobs (hard-coded hyperparameters for clarity).
- 🚫 No GIF/camera path helpers; just **train + preview**.

**Run the minimal version**
```bash
python src/tiny_nerf_min.py
```

This will:

- 📥 **Download or load** `tiny_nerf_data.npz` (if it’s not already present)
- 🧠 **Train** a lightweight TinyNeRF model for a short schedule
- 🖼️ **Save** preview and final render images in the `outputs/` directory

> 💡 **Tip:** Use the minimal file first to understand the **full NeRF flow** end-to-end.
> Once comfortable, switch to the **modular version** (`src/train.py`) to experiment
> with components or extend the model (e.g., add view-direction inputs).
---

## 🧾 Cheat Sheet

| Symbol        | Meaning                          | Shape  | File          |
| ------------- | -------------------------------- | ------ | ------------- |
| $(x, y, z)$   | 3D coordinate                    | (3,)   | anywhere      |
| $\gamma(x)$   | Positional encoding              | (63,)  | `encoding.py` |
| $F_\theta$    | TinyNeRF MLP                     | —      | `nerf.py`     |
| $c_i$         | color at sample i                | (3,)   | model output  |
| $\sigma_i$    | density at sample i              | (1,)   | model output  |
| $T_i$         | transmittance                    | scalar | `volume.py`   |
| $\delta_i$    | distance between samples         | scalar | `volume.py`   |
| $C(\mathbf{r})$ | rendered color                 | (3,)   | `volume.py`   |
| **PSNR**      | quality metric (higher = better) | scalar | `utils.py`    |


---

## 📚 References

* **Original Paper**:
  *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*
  *Ben Mildenhall, et al. (ECCV 2020)*
  [📄 Paper](https://arxiv.org/abs/2003.08934) | [🌐 Project Page](https://www.matthewtancik.com/nerf)

* **TinyNeRF Tutorial (TensorFlow)** — Google Research
  [https://colab.research.google.com/github/bmild/nerf/blob/master/tiny_nerf.ipynb](https://colab.research.google.com/github/bmild/nerf/blob/master/tiny_nerf.ipynb)

---
