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

```markdown
## 🧩 Project Structure

```

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
\[
F_\theta: (x, y, z) \rightarrow (r, g, b, \sigma)
\]
An MLP parameterized by θ maps each 3D coordinate to:
- RGB color
- Density (σ)

---

### 2. Positional Encoding (Fourier Features)
Small networks can’t easily learn fine spatial detail, so we map coordinates to a higher frequency space:

\[
\gamma(x) = [\sin(2^0 x), \cos(2^0 x), \sin(2^1 x), \cos(2^1 x), \dots, \sin(2^{L-1} x), \cos(2^{L-1} x)]
\]

This lets the network represent high-frequency variations (sharp edges, textures).

---

### 3. Ray Marching (Sampling Along Rays)
For each camera ray, we sample **N** points between a near and far bound:
\[
\mathbf{x}_i = \mathbf{o} + t_i \mathbf{d}, \quad t_i \in [t_{near}, t_{far}]
\]
where:
- **o** = ray origin (camera center)
- **d** = ray direction
- **tᵢ** = distance along the ray

We query the network at each sampled point to get colors and densities.

---

### 4. Volume Rendering (Compositing)
We combine these samples along the ray using the **volume rendering equation**:

\[
C(\mathbf{r}) = \sum_i T_i \, (1 - e^{-\sigma_i \delta_i}) \, c_i
\]
where:
- \( T_i = \prod_{j < i} e^{-\sigma_j \delta_j} \) = accumulated transmittance  
- \( \delta_i \) = distance between samples  
- \( c_i \) = predicted color  
- \( \sigma_i \) = predicted density

This gives us the final pixel color \( C(\mathbf{r}) \).

---

### 5. Optimization (Training)
We train the network by minimizing mean-squared error (MSE) between rendered colors and ground truth pixels:
\[
\mathcal{L} = \frac{1}{N} \sum_{\text{pixels}} || C_\theta(\mathbf{r}) - C_{gt}(\mathbf{r}) ||^2
\]

We also report **PSNR (Peak Signal-to-Noise Ratio)**:
\[
\text{PSNR} = -10 \log_{10}(\text{MSE})
\]

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

## 🧾 Cheat Sheet

| Symbol        | Meaning                          | Shape  | File          |
| ------------- | -------------------------------- | ------ | ------------- |
| ( (x, y, z) ) | 3D coordinate                    | (3,)   | anywhere      |
| ( \gamma(x) ) | Positional encoding              | (63,)  | `encoding.py` |
| ( F_\theta )  | TinyNeRF MLP                     | —      | `nerf.py`     |
| ( c_i )       | color at sample i                | (3,)   | model output  |
| ( \sigma_i )  | density at sample i              | (1,)   | model output  |
| ( T_i )       | transmittance                    | scalar | `volume.py`   |
| ( \delta_i )  | distance between samples         | scalar | `volume.py`   |
| ( C(r) )      | rendered color                   | (3,)   | `volume.py`   |
| PSNR          | quality metric (higher = better) | scalar | `utils.py`    |

---

## 📚 References

* **Original Paper**:
  *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*
  *Ben Mildenhall, et al. (ECCV 2020)*
  [📄 Paper](https://arxiv.org/abs/2003.08934) | [🌐 Project Page](https://www.matthewtancik.com/nerf)

* **TinyNeRF Tutorial (TensorFlow)** — Google Research
  [https://colab.research.google.com/github/bmild/nerf/blob/master/tiny_nerf.ipynb](https://colab.research.google.com/github/bmild/nerf/blob/master/tiny_nerf.ipynb)

---

## 🧩 Extending This Repo (For Research)

This implementation is minimal but extensible:

* 🔧 **Add view-direction conditioning** for specular surfaces.
* 🪞 **Hierarchical sampling** for coarse→fine rendering.
* 🤖 **Affordance fields**: Replace (rgb, σ) output with multi-affordance vectors for robotics research.
* 🧠 **Feature MLPs**: Encode latent embeddings for tasks like grasp planning or motion prediction.

---

## ❤️ Acknowledgments

Built as a learning-first project inspired by the original NeRF authors.
Special thanks to the open-source community for their educational repos and tutorials.

---

> “Understanding is reimplementation.” — this repo is designed so you can *read it top to bottom and understand every line* before you scale it up to your PhD affordance field model.

```
```
