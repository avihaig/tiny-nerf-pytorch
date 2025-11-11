import os, time
import numpy as np
import torch
import torch.nn as nn
import imageio.v2 as imageio

# =========================
# 1) CONFIG & DEVICE SETUP
# =========================
SEED = 0
ITERS = 8000          # training steps (increase to ~20k for better results)
N_RAND = 2048         # rays per step
N_SAMPLES = 64        # samples along each ray
NEAR, FAR = 2.0, 6.0  # sampling bounds
LR = 5e-4             # learning rate
PREVIEW_EVERY = 1000  # save a preview image every N steps

torch.manual_seed(SEED)
np.random.seed(SEED)
# We set random seeds for reproducibility.
# Many parts of NeRF training involve randomness:
#  - selecting random rays from an image,
#  - sampling random depth points along each ray,
#  - initializing neural network weights.
#
# Both PyTorch and NumPy have their own random number generators,
# so we seed both of them to make sure we can get the same training
# behavior and results every time we rerun the script.
#
# (If we only seed one, operations from the other library could still
# produce different random values across runs.)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] {device} torch={torch.__version__}")

os.makedirs("outputs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# =======================
# 2) LOAD TINY NERF DATA
# =======================
def load_tiny_nerf_npz(path="data/tiny_nerf_data.npz"):
    """
    Loads the TinyNeRF dataset (.npz file) and ensures the data types are GPU-friendly.

    Explanation:
    When we load data with NumPy, most arrays are stored as float64 (double precision).
    PyTorch models, however, use float32 by default on both CPU and GPU.
    Using float64 in the model would:
      • double the memory usage
      • slow down computations on the GPU (since most GPU operations are optimized for float32).

    Therefore, we convert any float64 arrays to float32 before turning them into torch tensors.
    This keeps training efficient and avoids type mismatch errors later.
    """
    d = np.load(path)
    out = {}
    for k, v in d.items():
        if hasattr(v, "dtype") and v.dtype == np.float64:
            v = v.astype(np.float32)
        out[k] = v
    return out

data = load_tiny_nerf_npz("data/tiny_nerf_data.npz")
# Move all relevant arrays to the same device (CPU or GPU) to speed up computation.
# Each variable plays a specific role in the NeRF pipeline:

images = torch.from_numpy(data["images"]).to(device)
# (N, H, W, 3)
# All training RGB images, already normalized in [0,1].
# N = number of camera views (images).
# Each pixel’s color will serve as the ground truth target for rendering loss.

poses = torch.from_numpy(data["poses"]).to(device)
# (N, 4, 4)
# 4X4 is the camera-to-world transformation matrix for each image.
# Each 4x4 matrix encodes the position (translation) and orientation (rotation)
# of that camera in 3D space — letting us compute rays for every pixel.

focal = float(data["focal"])
# Scalar focal length of the virtual pinhole camera (in pixels).
# Used to compute the ray directions from pixel coordinates.

N, H, W, _ = images.shape
print(f"[data] N={N} H={H} W={W} focal={focal:.2f}")

# ==========================
# 3) CAMERA RAYS PER POSE
# ==========================
def get_rays(H, W, focal, c2w, device):
    """
    Generate one *ray* per pixel for a single camera pose.

    Each ray has:
      - an origin (the camera center in world space)
      - a direction (where that pixel "looks" into the 3D scene)

    Returns:
      rays_o: (H*W, 3)  world-space ray origins  → same for all pixels (the camera position)
      rays_d: (H*W, 3)  world-space ray directions → different for each pixel

    Intuition (geometry)
    --------------------
    Think of a **pinhole camera**: each pixel (i, j) corresponds to a direction
    through a virtual image plane located at z = -1 in the *camera coordinate frame*.

    If the focal length is f (in pixels), the direction vector of pixel (i, j)
    in camera space is roughly:

        d_cam = [ (i - W/2) / f,  -(j - H/2) / f,  -1 ]

    - Subtracting W/2 and H/2 centers the coordinate system at the image middle.
    - Dividing by focal scales by the "zoom" of the camera (focal length).
    - The negative z means the camera looks *forward* along -z in its local frame.

    After computing all these directions, we:
      1. Rotate + translate them into *world space/coords* using the camera-to-world matrix (c2w):
        [ R | t ]  (4x4)
        [ 0 | 1 ]
        where R is a 3x3 rotation, t is a 3x1 camera position in world space.
      2. Normalize them (so all rays have unit direction length).

    ---------------------------------------------------------------
    Understanding the PyTorch building blocks:
    ------------------------------------------
    - torch.arange(W) → [0, 1, 2, ..., W-1]
      Creates a 1D tensor of equally spaced integers.
      (Same as range(), but returns a Tensor usable for math on GPU.)

    - torch.meshgrid(a, b, indexing='xy')
      Takes two 1D tensors (a and b) and produces coordinate grids:
        i, j = torch.meshgrid(torch.arange(W), torch.arange(H))
      Here, 'i' will hold x coordinates (columns), 'j' y coordinates (rows).

      Example for W=3, H=2:
         i = [[0,1,2],
              [0,1,2]]
         j = [[0,0,0],
              [1,1,1]]

      So (i[x,y], j[x,y]) is the pixel coordinate.

    - torch.stack([...], dim=-1)
      Combines multiple 2D tensors into one tensor along a new last axis.
      For example:
          a = [[1,2],
               [3,4]]
          b = [[5,6],
               [7,8]]
          torch.stack([a,b], dim=-1)
          → [[[1,5],
              [2,6]],
             [[3,7],
              [4,8]]]   (shape: 2x2x2)

      Here, we stack x, y, and z directions to get (H, W, 3).

    - @ R.T
      Matrix multiplication to rotate directions from camera → world coordinates.

    - torch.nn.functional.normalize(v, dim=-1)
      Normalizes each row of v to unit length (||v|| = 1). Makes each ray direction
      a *unit vector*, so “1.0” along the ray corresponds to “1 meter” (or 1 world unit).

    ---------------------------------------------------------------
    Example of shapes
    -----------------
    Suppose the image is 2x2 pixels (H=2, W=2).

      (i, j) pairs:
         (0,0)  (1,0)
         (0,1)  (1,1)

    We'll get 4 rays total, one per pixel.
    Each has 3 coordinates (x, y, z).

      rays_o shape → (4, 3): same origin [camera_x, camera_y, camera_z] repeated 4 times.
      rays_d shape → (4, 3): direction vectors, one per pixel.

    Example (conceptually):
      rays_o = [[0,0,0],
                [0,0,0],
                [0,0,0],
                [0,0,0]]
      rays_d = [[-0.3,  0.2, -1.0],
                [ 0.3,  0.2, -1.0],
                [-0.3, -0.2, -1.0],
                [ 0.3, -0.2, -1.0]]

    NOTE: NeRF’s “camera looks along -z” convention is common in tutorials.
    """
    # (1) Build a grid of pixel coordinates across the image plane.
    i, j = torch.meshgrid(
        torch.arange(W, device=device),  # horizontal pixel positions: 0..W-1
        torch.arange(H, device=device),  # vertical pixel positions:   0..H-1
        indexing='xy'
    )

    # (2) Compute the camera-space direction vector for each pixel.
    dirs = torch.stack([
        (i - W * 0.5) / focal,             # horizontal offset (x-axis)
        -(j - H * 0.5) / focal,            # vertical offset (y-axis, flipped)
        -torch.ones_like(i, device=device) # z = -1 (forward direction)
    ], dim=-1)  # → shape: (H, W, 3)

    # (3) Transform directions from camera → world space.
    R = c2w[:3, :3]                        # rotation (3x3)
    t = c2w[:3, 3]                         # translation (camera position)
    rays_d = (dirs @ R.T).reshape(-1, 3)   # rotate each direction
    rays_o = t.expand(rays_d.shape)        # origin = camera center for all pixels

    # (4) Normalize the direction vectors.
    rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
    return rays_o, rays_d  # (H*W, 3), (H*W, 3)

# Precompute rays for each training image (this avoids recomputing per step).
# Shapes:
#   all_rays_o: (N, H*W, 3)
#   all_rays_d: (N, H*W, 3)
#   pixels    : (N, H*W, 3)  (flattened images as RGB targets)
all_rays_o, all_rays_d = [], []
for i in range(N):
    ro, rd = get_rays(H, W, focal, poses[i], device=device)
    all_rays_o.append(ro)
    all_rays_d.append(rd)
all_rays_o = torch.stack(all_rays_o, dim=0)  # (N, HW, 3)
all_rays_d = torch.stack(all_rays_d, dim=0)  # (N, HW, 3)
pixels = images.view(N, H * W, 3)            # (N, HW, 3)


# =======================================
# 4) POSITIONAL ENCODING (FOURIER FEATS)
# =======================================
class PositionalEncoding(nn.Module):
    """
    Turns each 3D coordinate (x, y, z) into a *richer feature vector*
    containing multiple sine and cosine functions at increasing frequencies.

    Why we need this:
    -----------------
    A small MLP (like TinyNeRF) has trouble learning sharp edges or detailed
    patterns directly from raw coordinates like (0.23, -0.91, 0.55). It tends
    to produce only *smooth* color fields.

    To fix that, we map each coordinate to a *higher-dimensional signal* that
    contains multiple frequency versions of it — kind of like decomposing
    the coordinate into sine and cosine waves. This gives the network access
    to both low-frequency (broad shapes) and high-frequency (fine detail)
    variations.

    Mathematically:
    ---------------
    For one coordinate x:
        γ(x) = [sin(2⁰x), cos(2⁰x),
                 sin(2¹x), cos(2¹x),
                 sin(2²x), cos(2²x),
                 ...,
                 sin(2^(L-1)x), cos(2^(L-1)x)]
    where L = num_freqs (number of frequency bands).

    We do this for x, y, z and concatenate everything:
        γ(x,y,z) = concat(γ(x), γ(y), γ(z))

    Optionally, we can also include the original (x,y,z) at the beginning.
    The final output dimension becomes:
        out_dim = 3 * 2 * L + (3 if include_input else 0)

    Example:
    --------
    Suppose num_freqs = 2, include_input = True, and the input is (x,y,z) = (1, 2, 3).
    Then:
        freq_bands = 2.0 ** torch.arange(num_freqs).float() =>  2**[0,1] = [1, 2]
        freq_bands = [1, 2]
        γ(x,y,z) = [x, y, z,
                    sin(1*x), cos(1*x), sin(1*y), cos(1*y), sin(1*z), cos(1*z),
                    sin(2*x), cos(2*x), sin(2*y), cos(2*y), sin(2*z), cos(2*z)]
    → shape: (15,)

    So instead of giving the MLP only (1,2,3), we give it 15 numbers that
    represent both position and frequency structure — this massively improves
    the network’s ability to model detailed 3D scenes.

    ---------------------------------------------------------------
    Understanding the PyTorch operations:
    ------------------------------------
    - torch.arange(num_freqs)
        → [0, 1, 2, ..., num_freqs-1]
        This represents the power indices for 2^k.

    - 2.0 ** torch.arange(num_freqs)
        → [1, 2, 4, 8, ...]
        These are the actual frequency multipliers we’ll apply to x,y,z.

    - self.register_buffer("freq_bands", ...)
        Adds a tensor to the module that is *not a parameter* (it won’t be trained)
        but automatically moves to the right device (CPU or GPU).

    - for f in self.freq_bands:
          torch.sin(f * x), torch.cos(f * x)
        For each frequency f, we multiply every coordinate by f, then compute
        its sine and cosine. Each operation is *elementwise* and returns a
        tensor of the same shape as x.

    - torch.cat(enc, dim=-1)
        Concatenates all the results (original x,y,z + all sine/cosine pairs)
        along the last dimension, so the output keeps the same shape as x
        except the last dimension grows.
    
    ---------------------------------------------------------------
    Summary Table:
    | Concept               | Purpose                                    | PyTorch Function                 |
    | --------------------- | ------------------------------------------ | -------------------------------- |
    | **Fourier features**  | Allow MLPs to learn high-frequency details | `torch.sin`, `torch.cos`         |
    | **Frequency bands**   | Scale coordinates by powers of 2           | `2.0 ** torch.arange(num_freqs)` |
    | **Stacking features** | Combine all sine/cosine outputs            | `torch.cat(enc, dim=-1)`         |
    | **Module buffer**     | Keep constants on the right device         | `self.register_buffer()`         |
    ---------------------------------------------------------------
    Notes on Python features used:
    ------------------------------

    • super().__init__()
        This calls the parent class's constructor (here: nn.Module).
        It sets up all the internal PyTorch machinery that makes this class
        behave like a real neural-network module — so it can be moved to GPU,
        saved in checkpoints, and have its parameters tracked automatically.

        Without this line, our class wouldn’t properly register as an nn.Module.

    • @property
        Normally, if we define a function like this:

            def out_dim(self):
                return something

        we would have to call it using parentheses:
            self.out_dim()

        But by adding @property above the function, we can access it like a normal variable:
            self.out_dim    ← no parentheses needed

        In other words, @property turns a *method* into a *read-only attribute*.

        We use it here because `out_dim` is not something we manually set;
        it’s a value that is *computed* from other attributes (num_freqs, include_input).
        So it makes sense to read it like a property of the encoder, not a function we call.

    """
    def __init__(self, num_freqs=10, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input

        # Compute all frequency bands: [1, 2, 4, 8, ..., 2^(L-1)]
        # and store them in a buffer so they move with the model.
        freq_bands = 2.0 ** torch.arange(num_freqs).float()
        self.register_buffer("freq_bands", freq_bands)

    @property
    def out_dim(self):
        """
        Compute the output dimensionality dynamically.
        Each of x,y,z gets 2*num_freqs features (sin and cos),
        and optionally the 3 raw coords if include_input=True.
        """
        base = 3 * 2 * self.num_freqs
        return base + (3 if self.include_input else 0)

    def forward(self, x):
        """
        x: (..., 3) tensor of 3D coordinates in world space
        returns: (..., out_dim) tensor of encoded features
        """
        enc = []

        # Optionally include the original coordinates (for low-frequency info)
        if self.include_input:
            enc.append(x)

        # Apply sin and cos to each frequency band for each coordinate
        for f in self.freq_bands:
            # Each of these keeps the same shape as x (...,3)
            enc.append(torch.sin(f * x))
            enc.append(torch.cos(f * x))

        # Concatenate all features along the last axis
        # Example: if x = (batch, 3), output = (batch, out_dim)
        return torch.cat(enc, dim=-1)


# Instantiate encoder and move it to device
enc = PositionalEncoding(num_freqs=10, include_input=True).to(device)
print(f"[encoding] out_dim = {enc.out_dim}")

"""
Quick demo of what happens for one 3D point:
--------------------------------------------
Let’s test with a single coordinate x = [1.0, 2.0, 3.0]:

    >>> enc = PositionalEncoding(num_freqs=2, include_input=True)
    >>> x = torch.tensor([[1.0, 2.0, 3.0]])
    >>> y = enc(x)
    >>> y.shape
    torch.Size([1, 15])

    # Breakdown:
    # include_input = 3
    # sin/cos for each of 3 coords at 2 frequencies = 3 * 2 * 2 = 12
    # total = 15 values per point
"""


# ==========================
# 5) THE TINY NeRF MLP
# ==========================
class TinyNeRF(nn.Module):
    """
    A minimal MLP (Multi-Layer Perceptron) used by NeRF to predict color and density
    from encoded 3D coordinates.

    The network learns a continuous function:
        F_θ(x, y, z) → (r, g, b, σ)

    where:
        (x, y, z)  = 3D position (after positional encoding)
        (r, g, b)  = color at that location, in [0,1]
        σ (sigma)  = density (how much the point absorbs light)

    ---------------------------------------------------------------
    Architecture Overview
    ---------------------------------------------------------------
    Input:
      - encoded 3D point (dim = enc.out_dim, usually 63)
    Layers:
      - several fully-connected (Linear) layers with ReLU activations
      - one "skip connection" layer that re-injects the original input halfway through
    Output:
      - rgb   (3 channels, squashed by Sigmoid to [0,1])
      - sigma (1 channel, non-negative by ReLU)

    ---------------------------------------------------------------
    WHY THE SKIP CONNECTION?
    ---------------------------------------------------------------
    It reintroduces the original input halfway through the network,
    helping the model remember *where* each point is in space.
    Without it, deeper MLPs tend to “forget” positional details.
    ---------------------------------------------------------------
    ---------------------------------------------------------------
    VISUAL REPRESENTATION
    ---------------------------------------------------------------

                ┌──────────────────────────────────────────────────┐
                │             Positional Encoding γ(x)             │
                │        (input: 3D coords → output: 63D)          │
                └──────────────────────────────────────────────────┘
                                     │
                                     ▼
                            ┌───────────────────┐
                            │  Linear(63→128)   │
                            │  + ReLU           │
                            └───────────────────┘
                                     │
                                     ▼
                            ┌───────────────────┐
                            │  Linear(128→128)  │  ← skip connection here
                            │  + ReLU           │     (concat input)
                            └───────────────────┘
                                     │
                              ┌──────┴───────────────────────────┐
                              │   Concatenate h ⊕ γ(x) (adds +63) │
                              └──────┬───────────────────────────┘
                                     ▼
                            ┌───────────────────┐
                            │  Linear(191→128)  │
                            │  + ReLU           │
                            └───────────────────┘
                                     │
                                     ▼
                            ┌───────────────────┐
                            │  Linear(128→128)  │
                            │  + ReLU           │
                            └───────────────────┘
                                     │
                          ┌──────────┴───────────┐
                          ▼                      ▼
               ┌─────────────────┐     ┌────────────────────┐
               │ Linear(128→3)   │     │ Linear(128→1)      │
               │ + Sigmoid → RGB │     │ + ReLU → Density σ │
               └─────────────────┘     └────────────────────┘
    ---------------------------------------------------------------
    
    Understanding the PyTorch parts:
    --------------------------------
    • super().__init__()
        Initializes the parent nn.Module class so PyTorch knows this is a layer.
        Without it, the model couldn’t be moved to GPU or saved properly.

    • nn.Linear(in_dim, out_dim)
        A standard fully-connected layer:
           output = input @ Wᵀ + b
        where W is the weight matrix and b is the bias.

    • nn.ModuleList([...])
        A Python list that registers each layer as part of the module.
        Regular lists wouldn’t let PyTorch find their parameters automatically.

    • nn.Sequential(...)
        A quick way to chain a few layers and activations together.
        Example:
            nn.Sequential(nn.Linear(128, 3), nn.Sigmoid())
        means: apply a Linear layer → then a Sigmoid activation.

    • torch.relu()
        Applies ReLU activation: max(0, x), introducing non-linearity.

    • torch.cat([h, x_enc], dim=-1)
        Concatenates tensors along their last dimension.
        Here, it merges the current activations (h) with the original input (x_enc)
        during the skip connection stage.

    ---------------------------------------------------------------
    Example:
    --------
    Suppose:
      - Input encoding has size 63 (enc.out_dim)
      - hidden = 128
      - depth = 4
      - skip_at = 2

    Then the network looks like:

      Layer 1: Linear(63 → 128)
      Layer 2: Linear(128 → 128)
      [skip connection adds +63 here]
      Layer 3: Linear(191 → 128)
      Layer 4: Linear(128 → 128)
      Heads:
        σ: Linear(128 → 1) + ReLU
        rgb: Linear(128 → 3) + Sigmoid
    ---------------------------------------------------------------
    Summary Table:
    | Component         | Purpose                            | Analogy                             |
    | ----------------- | ---------------------------------- | ----------------------------------- |
    | `nn.Linear`       | Fully-connected layer              | Like y = mx + b for vectors         |
    | `nn.ModuleList`   | Holds layers that PyTorch can see  | Like a “trainable list”             |
    | `nn.Sequential`   | Chains simple layers together      | Like a mini pipeline                |
    | `skip connection` | Re-adds original input mid-network | Helps memory of spatial coordinates |
    | `ReLU`, `Sigmoid` | Activations (shape of output)      | ReLU → non-negative, Sigmoid → 0–1  |
    ---------------------------------------------------------------
    """
    def __init__(self, in_dim, hidden=128, depth=4, skip_at=2):
        super().__init__()
        self.in_dim = in_dim      # input feature size (e.g., 63)
        self.hidden = hidden      # hidden layer width
        self.depth = depth        # number of fully connected layers
        self.skip_at = skip_at    # where to apply the skip connection

        layers = []
        last = in_dim

        # Build the fully-connected backbone
        for i in range(depth):
            layers.append(nn.Linear(last, hidden))
            # After skip layer, next input = previous hidden + original input
            last = hidden if i != (skip_at - 1) else (hidden + in_dim)

        # Register the layers so PyTorch can track them
        self.layers = nn.ModuleList(layers)

        # Two small "heads" for final outputs
        self.sigma = nn.Sequential(nn.Linear(hidden, 1), nn.ReLU(inplace=True))
        self.rgb   = nn.Sequential(nn.Linear(hidden, 3), nn.Sigmoid())

    def forward(self, x_enc):
        """
        Forward pass through the MLP.
        Input:  x_enc (..., in_dim)
        Output: rgb (..., 3), sigma (..., 1)
        """
        h = x_enc
        for i, lin in enumerate(self.layers):
            h = torch.relu(lin(h))
            # Add skip connection after the chosen layer
            if i == (self.skip_at - 1):
                h = torch.cat([h, x_enc], dim=-1)
        sigma = self.sigma(h)  # density prediction (non-negative)
        rgb   = self.rgb(h)    # color prediction (0..1)
        return rgb, sigma

model = TinyNeRF(in_dim=enc.out_dim, hidden=128, depth=4, skip_at=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
print(f"[model] TinyNeRF with {sum(p.numel() for p in model.parameters())} parameters")

# ===========================
# 6) STRATIFIED SAMPLING
# ===========================
def stratified_samples(near, far, n_samples, rays_o, rays_d, randomized=True):
    """
    Sample 3D points along each camera ray between near and far planes.

    ---------------------------------------------------------------
    PURPOSE
    ---------------------------------------------------------------
    For every ray (origin, direction), we want to take N evenly spaced
    samples between the "near" and "far" distances along that ray.
    Each sample gives us a 3D point (x, y, z) that we’ll later feed
    into the NeRF MLP to get its color and density.

    ---------------------------------------------------------------
    VISUAL REPRESENTATION
    ---------------------------------------------------------------
    For one ray:
           near                     far
            |-------------------------|
            o----*----*----*----*----> direction
                z1   z2   z3   z4   z5  ... (N samples)

    Each '*' is a 3D position along the ray where we query the network.

    If `randomized=True`, each sample is slightly “jittered” inside its
    bin (Monte Carlo sampling), which reduces aliasing and improves
    generalization — like adding soft noise during training.

    ---------------------------------------------------------------
    MATHEMATICAL DEFINITION (WITH EXPLANATION)
    ---------------------------------------------------------------
    We first define a normalized parameter t_i ∈ [0, 1], where 0 = near and 1 = far.

        z_i = (1 - t_i) * near + t_i * far

    - This linearly interpolates between `near` and `far`.
    - If near=2.0, far=6.0, and t_i = 0.5 → z_i = (1-0.5)*2.0 + 0.5*6.0 = 4.0

    So we get evenly spaced depth values (z-values) between the near and far planes.

    Once we have those depths, we can get the actual 3D coordinates along each ray:

        p_i = rays_o + z_i * rays_d

    where:
      - rays_o: (N_rays, 3)  → the 3D starting point of each ray (the camera origin)
      - rays_d: (N_rays, 3)  → the normalized 3D direction of each ray
      - z_i: scalar depth along that ray
      - p_i: resulting 3D sample point

    Example of one ray:
        rays_o = (0, 0, 0)
        rays_d = (0, 0, -1)      # ray going straight into the screen
        near = 2, far = 6, n_samples = 4
        z_i = [2.0, 3.33, 4.67, 6.0]

        Then:
        p_i = (0, 0, 0) + z_i * (0, 0, -1)
            = [(0, 0, -2.0),
               (0, 0, -3.33),
               (0, 0, -4.67),
               (0, 0, -6.0)]

    These points are what the MLP will later “see” to predict color/density.

    ---------------------------------------------------------------
    TORCH FUNCTIONS USED
    ---------------------------------------------------------------
    • torch.linspace(start, end, steps)
        → Creates evenly spaced numbers between start and end.
          Example: torch.linspace(0, 1, 5) = [0.00, 0.25, 0.5, 0.75, 1.00]

    • tensor.expand(new_shape)
        → “Pretends” to repeat the tensor along new dimensions without copying it.
          Example: t = [1,2,3] → t.expand(2,3) =
                  [[1,2,3],
                   [1,2,3]]

    • tensor[:, None, :]
        → Adds a *new* dimension (axis) at position 1.
          Example:
              rays_o shape: (N_rays, 3)
              rays_o[:, None, :] → shape: (N_rays, 1, 3)
          This is used so that when we add (rays_o + z_vals * rays_d),
          PyTorch can automatically "broadcast" the values and match
          each ray to all its sample depths without writing loops.

    • torch.rand_like(tensor)
        → Creates random values with the same shape as another tensor.
          Used here to slightly randomize (jitter) each sample’s position.

    ---------------------------------------------------------------
    EXTENDED EXAMPLE
    ---------------------------------------------------------------
    Suppose:
        We have 2 rays, each with 4 samples.
        near = 2.0, far = 6.0

    Step 1: Create normalized t-values
        t = [0.00, 0.33, 0.67, 1.00]

    Step 2: Interpolate z-values (depths)
        z_vals =
          [[2.00, 3.33, 4.67, 6.00],
           [2.00, 3.33, 4.67, 6.00]]      # same for both rays

    Step 3: Compute 3D sample points
        Let's say:
            rays_o[0] = (0, 0, 0)     rays_d[0] = (0, 0, -1)
            rays_o[1] = (1, 0, 0)     rays_d[1] = (0, -1, -1)

        Then:
          pts[0] = [
            (0, 0, -2.00),
            (0, 0, -3.33),
            (0, 0, -4.67),
            (0, 0, -6.00)
          ]

          pts[1] = [
            (1, -2.00, -2.00),
            (1, -3.33, -3.33),
            (1, -4.67, -4.67),
            (1, -6.00, -6.00)
          ]

    The final shapes:
        z_vals → (2, 4)
        pts    → (2, 4, 3)
    """
    N_rays = rays_o.shape[0]

    # Step 1: Create normalized depth positions t_i ∈ [0,1]
    t_vals = torch.linspace(0., 1., steps=n_samples, device=rays_o.device)  # (n_samples,)

    # Step 2: Convert to world-space depth values z_i
    z_vals = near * (1. - t_vals) + far * t_vals                           # (n_samples,)
    z_vals = z_vals.expand(N_rays, n_samples)                              # (N_rays, n_samples)

    # Step 3 (optional): Add stratified (randomized) jitter
    """
    Why add randomness?
    -------------------
    When we sample along each ray, the z-values (depths) are evenly spaced.
    But if we *always* sample at the same fixed positions, the model may
    overfit to those exact locations — causing banding artifacts.

    To fix that, we *perturb* (jitter) each sample slightly inside its
    interval.  This makes the training behave like Monte Carlo integration:
    every epoch, the ray samples change a bit, and the model learns to
    approximate the *continuous* color field rather than memorizing discrete
    bins.

    ---------------------------------------------------------------
    Intuition (1 D visualization)
    ---------------------------------------------------------------
    Suppose we have 4 bins along one ray between near=2 and far=6:

         |----|----|----|----|
         2    3    4    5    6
         ^                 ^
       lower             upper

    Normally we would take samples exactly at the bin edges:
         z_vals = [2.0, 3.33, 4.67, 6.0]

    With jittering, we instead pick one random point *inside each bin*:

         |----|----|----|----|
         2   2.8  3.9  5.5  6
             *     *     *     *

    Every time we run training, the * locations change slightly.
    This is why the network sees a “blurred” version of space and
    learns smoother radiance fields.

    ---------------------------------------------------------------
    How it works in code
    ---------------------------------------------------------------
    mids  = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
        → midpoints between consecutive depth samples.

    upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
        → upper bound of each sampling bin.

    lower = torch.cat([z_vals[:, :1], mids], dim=-1)
        → lower bound of each bin.

    t_rand = torch.rand_like(z_vals)
        → uniform random numbers in [0, 1] with same shape as z_vals.

    z_vals = lower + (upper - lower) * t_rand
        → for each bin [lower_i, upper_i], pick a random point inside it.

    ---------------------------------------------------------------
    Example (numerical)
    ---------------------------------------------------------------
    Let’s say one ray has z_vals = [2, 4, 6]
    so z_vals.shape = (1, 3) each element is a depth sample along that ray - near -> far
    so : 
        - z_vals[:, :-1] all elements except the last one [2, 4]
        - z_vals[:, 1:]  all elements except the first one [4, 6]
        - z_vals[:, :1] first element only [2]
        - z_vals[:, -1:] last element only [6]
    
      mids  = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
            = 0.5 * ([2, 4] + [4, 6])
            = [3, 5]

      lower = torch.cat([z_vals[:, :1], mids], dim=-1)
            = torch.cat([[2], [3, 5]], dim=-1)
            = [2, 3, 5]

      upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
            = torch.cat([[3, 5], [6]], dim=-1)
            = [3, 5, 6]

    If torch.rand_like(z_vals) = [0.2, 0.7, 0.5], then:

      z_vals_new = lower + (upper − lower) * rand
                 = [2 + (3−2)*0.2, 3 + (5−3)*0.7, 5 + (6−5)*0.5]
                 = [2.2, 4.4, 5.5]

    So our new sample positions are slightly randomized inside each bin.
    Over many iterations, this produces a more accurate approximation of
    the volume integral along the ray.

    """
    if randomized:
        mids = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])                     # midpoints between samples
        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)                 # upper bounds of bins
        lower = torch.cat([z_vals[:, :1], mids], dim=-1)                  # lower bounds of bins
        t_rand = torch.rand_like(z_vals)                                  # random noise per bin
        z_vals = lower + (upper - lower) * t_rand                         # jittered depths

    # Step 4: Compute 3D points along each ray
    """
    Goal:
      Turn each depth value z_i on a ray into its actual 3D position:
         p_i = rays_o + z_i * rays_d
      and do this *vectorized* for all rays and all samples at once.

    Shapes before broadcasting:
      rays_o            : (N_rays, 3)          # one origin per ray
      rays_d            : (N_rays, 3)          # one (unit) direction per ray
      z_vals            : (N_rays, n_samples)  # depths along each ray

    Why add singleton dimensions (`None`)?
      • rays_o[:, None, :] → (N_rays, 1,        3)
        "Copy" each origin across its n_samples (broadcasted, no memory copy).
      • rays_d[:, None, :] → (N_rays, 1,        3)
        "Copy" each direction across its n_samples.
      • z_vals[..., None]  → (N_rays, n_samples, 1)
        Make each depth a (…, 1) so it can multiply a 3D vector component-wise.

    Broadcasting rules (PyTorch/NumPy):
      When shapes differ, dimensions of size 1 are automatically expanded
      to match the other tensor. After adding the `None` dimensions, we get:

         rays_o[:, None, :]   → (N_rays, 1,        3)
       + rays_d[:, None, :]   → (N_rays, 1,        3)  *  z_vals[..., None] → (N_rays, n_samples, 1)
       --------------------------------------------------------------------------------------------
         result               → (N_rays, n_samples, 3)

      Concretely:
        - z_vals[..., None] scales each ray’s direction for every sample.
        - Adding rays_o[:, None, :] shifts those scaled directions so they start at the origin.

    The actual code:
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., None]
        # pts has shape (N_rays, n_samples, 3)

    Why directions should be unit length:
      We normalized rays_d earlier. That way, "1.0" in z is "1 world unit" along the ray.
      If rays_d had length ≠ 1, the spacing between points would be distorted.

    ---------------------------------------------------------------
    MINI NUMERICAL EXAMPLE
    ---------------------------------------------------------------
    Suppose:
      N_rays = 2,  n_samples = 3

      rays_o =
        [[0, 0, 0],
         [1, 0, 0]]                           # (2,3)

      rays_d =
        [[0, 0, -1],
         [0, -1, -1]]  (assume already normalized)   # (2,3)

      z_vals =
        [[2.0, 4.0, 6.0],
         [2.0, 4.0, 6.0]]                     # (2,3)

    After adding singleton dims:
      rays_o[:, None, :]  → shape (2,1,3):
        [[[0,0,0]],
         [[1,0,0]]]

      rays_d[:, None, :]  → shape (2,1,3):
        [[[ 0,  0, -1]],
         [[ 0, -1, -1]]]

      z_vals[..., None]   → shape (2,3,1):
        [[[2.0],[4.0],[6.0]],
         [[2.0],[4.0],[6.0]]]

    Multiply & add:
      rays_d[:,None,:] * z_vals[...,None]  → (2,3,3)
        ray 0 directions:
          [0, 0, -1]*2 = [0, 0, -2]
          [0, 0, -1]*4 = [0, 0, -4]
          [0, 0, -1]*6 = [0, 0, -6]
        ray 1 directions:
          [0,-1,-1]*2 = [0,-2,-2]
          [0,-1,-1]*4 = [0,-4,-4]
          [0,-1,-1]*6 = [0,-6,-6]

      Add origins:
        pts =
          ray 0: [0,0,0] + ([0,0,-2], [0,0,-4], [0,0,-6])
               = [[0,0,-2], [0,0,-4], [0,0,-6]]

          ray 1: [1,0,0] + ([0,-2,-2], [0,-4,-4], [0,-6,-6])
               = [[1,-2,-2], [1,-4,-4], [1,-6,-6]]

    Final shape:
      pts → (N_rays, n_samples, 3) = (2, 3, 3)
    """
    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., None]     # (N_rays, n_samples, 3)

    return z_vals, pts



# ===============================
# 7) VOLUME RENDERING (COMPOSITE)
# ===============================
def volume_render(rgb, sigma, z_vals, rays_d, white_bkgd=True):
    """
    Perform **volume rendering**, the core step that turns 3D predictions into a
    final 2D image.

    ---------------------------------------------------------------
    What is Volume Rendering?
    ---------------------------------------------------------------
    Volume rendering is a way to simulate how light travels through a semi-transparent
    3D medium (like fog, smoke, or glass).  
    Instead of having solid surfaces, we imagine that *every point in space* can
    emit and absorb light according to:
      - its color (how it glows or reflects light)
      - its density σ (how much it blocks or absorbs light)

    For each camera ray, we collect contributions from many points along its path —
    as if the ray were passing through multiple thin, translucent layers.  
    Each layer slightly changes the ray’s color and brightness depending on how
    “dense” it is and what color light it emits.

    ---------------------------------------------------------------
    What This Function Does
    ---------------------------------------------------------------
    “Composite per-ray colors from many samples along the ray using the
    standard NeRF volume rendering equation.”

    That means:
    - We take all the 3D samples (points) along one ray.
    - Each sample gives us a *predicted color* (rgb) and a *density* (σ) from the network.
    - We combine (or “composite”) them together from nearest to farthest using the
      physical **volume rendering equation**, which determines how much each
      layer contributes to the final pixel color based on:
        1️⃣ its opacity (α = 1 − exp(−σΔ))  
        2️⃣ how much light passes through previous layers (T = Π(1−α))

    The result is a realistic final color for each pixel that respects visibility,
    transparency, and depth — all computed directly from the neural field.

    Inputs
    ------
    rgb    : (N_rays, N_samples, 3)       predicted colors in [0,1] at each sample
    sigma  : (N_rays, N_samples, 1)       predicted densities (>= 0) at each sample
    z_vals : (N_rays, N_samples)          depth (distance) of each sample along each ray
    rays_d : (N_rays, 3)                  ray directions (ideally unit-length)
    white_bkgd : bool                     if True, composite over a white background

    Outputs
    -------
    comp_rgb : (N_rays, 3)                final per-ray color after alpha compositing
    depth    : (N_rays, 1)                expected depth (weighted avg of z)
    acc      : (N_rays, 1)                accumulated opacity along the ray (sum of weights)
    weights  : (N_rays, N_samples)        contribution of each sample to the final color

    ---------------------------------------------------------------
    Big Picture (Intuition)
    ---------------------------------------------------------------
    Imagine the space in front of the camera as many thin, semi-transparent
    "sheets" stacked along the ray. At each sheet (sample), the network predicts:
      - color c_i (rgb[i])
      - density σ_i (how strongly it absorbs light)

    We convert density into an "opacity" α_i for that small slice:
        α_i = 1 - exp(-σ_i * Δ_i)
      where Δ_i is the spacing (thickness) between consecutive samples in *world units*.

    We also compute the transmittance T_i (how much light gets from the camera to i
    without being absorbed by previous slices):
        T_i = Π_{j < i} (1 - α_j)

    The final per-ray color is the sum of each slice's color * its visibility:
        C(ray) = Σ_i [ T_i * α_i * c_i ]  = Σ_i [ w_i * c_i ]
    where w_i = T_i * α_i are the per-sample weights.

    Depth is computed similarly as a weighted average of z:
        D(ray) = Σ_i [ w_i * z_i ] / Σ_i [ w_i ]   (but we return Σ_i [ w_i * z_i ] and acc = Σ_i w_i)

    ---------------------------------------------------------------
    Why Δ_i is multiplied by |rays_d| ?
    ---------------------------------------------------------------
    If rays_d are normalized, |rays_d| = 1, so Δ_i is just the difference in z.
    If they are not perfectly unit-length, multiplying by |rays_d| scales Δ_i
    into actual world distance along the ray direction, keeping α_i consistent.

    ---------------------------------------------------------------
    Shapes & Tensor Ops
    ---------------------------------------------------------------
    • deltas = z_vals[...,1:] - z_vals[...,:-1]        → (N_rays, N_samples-1)
      spacing between consecutive samples along each ray

    • We append a huge Δ for the last sample (acts like a "back wall"):
        delta_inf = 1e10
        deltas = cat([deltas, delta_inf], -1)          → (N_rays, N_samples)

    • alpha = 1 - exp(-sigma * deltas)                 → (N_rays, N_samples)
      turns densities into opacities given the slice thickness

    • T_i via cumulative product:
        T = cumprod( concat([1, 1-α_i], dim=-1), dim=-1 )[:, :-1]
      Prepend ones so T_0=1. Then drop the extra last element.
      This yields T_i = Π_{j < i} (1 - α_j).

    • weights = T * α                                  → (N_rays, N_samples)
      contribution of each sample to final color

    • comp_rgb = sum(weights[...,None] * rgb, dim=-2)  → (N_rays, 3)

    • depth   = sum(weights * z_vals, dim=-1, keepdim=True)  → (N_rays, 1)

    • acc     = sum(weights, dim=-1, keepdim=True)           → (N_rays, 1)

    • white background:
        C_white = C + (1 - acc)   (adds remaining visibility as white)
      If the background should be black, skip this.

    ---------------------------------------------------------------
    Mini Numeric Sketch (single ray, tiny samples)
    ---------------------------------------------------------------
    Suppose one ray has 3 samples with equal spacing Δ:
      σ = [0.8, 1.2, 3.0],  c = [[1,0,0], [0,1,0], [0,0,1]]

      α_i = 1 - exp(-σ_i * Δ)   (values in (0,1))
      T_0 = 1
      T_1 = (1 - α_0)
      T_2 = (1 - α_0)(1 - α_1)

      weights = [T_0 α_0, T_1 α_1, T_2 α_2]
      C = Σ_i weights[i] * c[i]

    This is exactly like compositing semi-transparent colored layers front-to-back.

    """
    # 1) Compute deltas (thickness of each sample slice along the ray)
    deltas = z_vals[..., 1:] - z_vals[..., :-1]                  # (N_rays, N_samples-1)
    delta_inf = 1e10 * torch.ones_like(deltas[..., :1])          # "infinite" last interval
    deltas = torch.cat([deltas, delta_inf], dim=-1)              # (N_rays, N_samples)

    # Scale by |rays_d| so deltas are in world units if rays_d isn't unit-length
    deltas = deltas * torch.norm(rays_d[:, None, :], dim=-1)     # (N_rays, N_samples)

    # 2) Densities → opacities for each slice: α_i = 1 - exp(-σ_i * Δ_i)
    #    sigma: (N_rays, N_samples, 1) → squeeze last dim to match deltas
    alpha = 1.0 - torch.exp(-sigma.squeeze(-1) * deltas)         # (N_rays, N_samples)

    # 3) Transmittance T_i = Π_{j < i} (1 - α_j)
    #    We implement with a cumulative product. Prepend a 1 so T_0 = 1.
    trans_prefix = torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-10], dim=-1)
    T = torch.cumprod(trans_prefix, dim=-1)[:, :-1]              # (N_rays, N_samples)

    # 4) Per-sample weights: w_i = T_i * α_i
    weights = T * alpha                                          # (N_rays, N_samples)

    # 5) Composite color & depth using the weights
    comp_rgb = torch.sum(weights[..., None] * rgb, dim=-2)       # (N_rays, 3)
    depth    = torch.sum(weights * z_vals, dim=-1, keepdim=True) # (N_rays, 1)
    acc      = torch.sum(weights, dim=-1, keepdim=True)          # (N_rays, 1)

    # 6) Optional white background: add remaining transmittance as white
    if white_bkgd:
        comp_rgb = comp_rgb + (1.0 - acc)

    return comp_rgb, depth, acc, weights


# ==========================
# 8) TRAINING LOOP
# ==========================
def mse2psnr(mse: torch.Tensor) -> torch.Tensor:
    """
    Convert Mean Squared Error (MSE) into PSNR (Peak Signal-to-Noise Ratio),
    a standard image quality metric.

    ---------------------------------------------------------------
    WHY WE USE IT
    ---------------------------------------------------------------
    During training, we minimize the **MSE** between the rendered pixel colors
    and the ground truth image pixels:
        MSE = average((predicted - target)^2)

    This tells us *numerically* how far our predictions are from reality,
    but MSE is hard to interpret — "is 0.002 good or bad?"

    So we convert MSE → **PSNR**, which expresses image quality in decibels (dB).
    It’s a logarithmic measure used in image compression and reconstruction tasks.

    In the original NeRF paper (Mildenhall et al., ECCV 2020),
    **PSNR is the main quantitative metric** used to evaluate how sharp and
    realistic the rendered images are compared to the ground truth.

    ---------------------------------------------------------------
    FORMULA
    ---------------------------------------------------------------
        PSNR = -10 * log10(MSE)

    - The smaller the error (MSE ↓), the higher the PSNR (↑).
    - A PSNR around:
        20 dB  → rough/blurry reconstruction
        25 dB  → reasonable quality
        30 dB+ → very sharp / nearly identical to ground truth

    ---------------------------------------------------------------
    IMPLEMENTATION NOTES
    ---------------------------------------------------------------
    • We use torch.log10() for the base-10 logarithm.
    • clamp_min(1e-10) avoids log(0) which is undefined.
    • Returned PSNR is in decibels (dB).

    In short:
        → MSE measures *error* (lower = better)
        → PSNR measures *visual fidelity* (higher = better)
    """
    return -10.0 * torch.log10(mse.clamp_min(1e-10))



def train():
    """
    End-to-end optimization of the TinyNeRF model.

    ---------------------------------------------------------------
    GOAL
    ---------------------------------------------------------------
    We want the model to learn a function that, when we:
      1) generate camera rays for a training image,
      2) sample 3D points along those rays,
      3) encode points and run them through the MLP,
      4) volume-render the per-ray colors,
    the **rendered pixels** match the **ground-truth pixels**.

    We minimize the Mean Squared Error (MSE) between:
        rendered RGB  vs  dataset RGB
    and we report PSNR for interpretability.

    ---------------------------------------------------------------
    HIGH-LEVEL LOOP (ONE ITERATION)
    ---------------------------------------------------------------
    1) Pick a training image (cycled by index).
    2) Randomly choose N_RAND pixel positions (rays) from that image.
    3) For each ray:
         a) sample depths z in [NEAR, FAR] (with jitter)
         b) compute 3D points p = o + z*d
         c) positional-encode p
         d) predict (rgb, sigma) with the MLP
         e) composite to a pixel color via volume rendering
    4) Compare rendered pixels vs ground truth pixels → MSE → backprop.
    5) Log loss/PSNR; periodically render a preview frame; save checkpoints.

    ---------------------------------------------------------------
    WHY RANDOM SUBSETS OF RAYS?
    ---------------------------------------------------------------
    Each (H×W) image can have ~10k pixels; running all rays each step is slow.
    Instead, we **sample a subset** of rays per step (mini-batch), which:
      • reduces memory and time per step,
      • introduces stochasticity → better generalization (like standard SGD).

    ---------------------------------------------------------------
    MIXED PRECISION (AMP)
    ---------------------------------------------------------------
    We wrap forward pass in:
        with torch.amp.autocast("cuda", enabled=(device.type=="cuda")):
    and scale the loss with GradScaler. This speeds up training on modern GPUs
    and reduces memory usage with minimal code changes.

    ---------------------------------------------------------------
    LOGGING & ARTIFACTS
    ---------------------------------------------------------------
    - Every 100 steps: print loss and PSNR.
    - Every PREVIEW_EVERY steps: render a full image for the next pose, save PNG.
    - After last step: save model checkpoint and a final render.

    ---------------------------------------------------------------
    SHAPES (per step, conceptual)
    ---------------------------------------------------------------
      rays_o, rays_d          : (N_RAND, 3)
      z_vals                  : (N_RAND, N_SAMPLES)
      pts                     : (N_RAND, N_SAMPLES, 3)
      enc(pts.reshape(-1,3))  : (N_RAND*N_SAMPLES, enc_dim)
      rgb                     : (N_RAND*N_SAMPLES, 3)
      sigma                   : (N_RAND*N_SAMPLES, 1)
      reshaped rgb            : (N_RAND, N_SAMPLES, 3)
      reshaped sigma          : (N_RAND, N_SAMPLES, 1)
      comp_rgb (render)       : (N_RAND, 3)
      target (GT)             : (N_RAND, 3)

    """
    t0 = time.time()

    for step in range(ITERS):
        model.train()

        # 1) Choose which training image to use (cycles 0..N-1).
        # Each training step uses one view (pose) from the dataset.
        # Example: at step 5, use image index 5 % N - means "5 modulo N" — it gives the *remainder* when 5 is divided by N.
        # ensures that even if we run thousands of training steps, we keep looping evenly through all N images.
        img_i = step % N

        # 2) Randomly select a subset of rays (pixels) from that image.
        """
        ---------------------------------------------------------------
        SUMMARY (per batch)
        ---------------------------------------------------------------
        For each of N_RAND pixels we now have:
            rays_o  → starting point of the ray in 3D (camera center)
            rays_d  → direction of the ray into the scene
            target  → what color the camera actually saw at that pixel

        The model will later try to reproduce "target" by rendering along
        (rays_o, rays_d) using its learned 3D scene representation.
        """
        #    - torch.randint generates random pixel indices between 0 and H*W-1.
        #    - Each index represents one (x, y) pixel, flattened to a 1D array.
        #    - N_RAND controls the number of sampled pixels (mini-batch size).
        inds = torch.randint(0, H * W, (N_RAND,), device=device)

        # Gather the corresponding ray origins and directions.
        # These were precomputed for every pixel of every image.
        rays_o = all_rays_o[img_i, inds]   # (N_RAND, 3) ray origins
        rays_d = all_rays_d[img_i, inds]   # (N_RAND, 3) ray directions

        # Fetch the ground-truth RGB colors for those same pixels.
        target = pixels[img_i, inds]       # (N_RAND, 3) ground-truth colors

        # 3) Stratified depth samples along each ray → 3D points
        #    z_vals: (N_RAND, N_SAMPLES),  pts: (N_RAND, N_SAMPLES, 3)
        z_vals, pts = stratified_samples(
            NEAR, FAR, N_SAMPLES, rays_o, rays_d, randomized=True
        )

        """
        ---------------------------------------------------------------
        STEP 4–6: Forward pass → Render → Loss → Backprop (with AMP)
        ---------------------------------------------------------------

        WHAT WE'RE DOING (big picture)
        ------------------------------
        For this mini-batch of rays:
          1) We already sampled N_SAMPLES 3D points along each ray (shape: (N_RAND, N_SAMPLES, 3)).
          2) We encode those points (Fourier features), run them through the MLP,
             and get per-point predictions: color (rgb) and density (sigma).
          3) We "composite" those predictions along each ray using volume rendering,
             producing one RGB color per ray → a tiny rendered image patch.
          4) We compare that to the ground-truth pixel colors and compute MSE.
          5) We backpropagate to update the MLP weights.

        WHY THE FLATTEN/RESHAPE?
        ------------------------
        The MLP expects inputs of shape (Batch, Features). We currently have:
            pts: (N_RAND, N_SAMPLES, 3)
        That means "N_RAND rays × N_SAMPLES points per ray".
        We flatten to a single big batch of 3D points:
            pts.reshape(-1, 3) → (N_RAND * N_SAMPLES, 3)
        Then we encode and run the MLP on all points at once (fast & vectorized).
        After predicting, we reshape the outputs back to per-ray, per-sample shapes
        so the volume renderer can combine them along each ray.

        WHAT IS MIXED PRECISION (AMP)?
        ------------------------------
        AMP = Automatic Mixed Precision.
        - It runs parts of the model in float16 (half precision) when safe,
          and float32 when needed — automatically.
        - Pros: faster and uses less GPU memory on modern GPUs.
        - We wrap the forward pass with:
              with torch.amp.autocast("cuda", enabled=(device.type=="cuda")):
          so AMP is only active on CUDA.
        - We also use GradScaler to safely scale the loss before backward()
          to avoid underflow in float16.

        STEP-BY-STEP THROUGH THE LINES
        -------------------------------
        """

        # 4) Forward pass (with optional mixed precision).
        #    We flatten samples so the MLP runs on a big batch of points.
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            # (A) Encode all 3D points (Fourier features).
            #     pts: (N_RAND, N_SAMPLES, 3) → flatten to (N_RAND*N_SAMPLES, 3)
            xenc = enc(pts.reshape(-1, 3))             # → (N_RAND*N_SAMPLES, enc_dim)

            # (B) Predict per-point outputs with the MLP:
            #     rgb:  (N_RAND*N_SAMPLES, 3)   ∈ [0,1]
            #     sigma:(N_RAND*N_SAMPLES, 1)   ≥ 0
            rgb, sigma = model(xenc)                   # flat predictions

            # (C) Reshape back to "per-ray, per-sample" so we can composite.
            #     rgb:   (N_RAND, N_SAMPLES, 3)
            #     sigma: (N_RAND, N_SAMPLES, 1)
            rgb   = rgb.reshape(N_RAND, N_SAMPLES, 3)
            sigma = sigma.reshape(N_RAND, N_SAMPLES, 1)

            # (D) Volume render along each ray:
            #     combine (rgb, sigma) across the N_SAMPLES depths → one color per ray.
            #     comp_rgb: (N_RAND, 3)
            comp_rgb, _, _, _ = volume_render(rgb, sigma, z_vals, rays_d)

            # (E) Compare to ground-truth target colors (same shape).
            #     Loss is mean squared error (MSE) over the mini-batch.
            loss = torch.mean((comp_rgb - target) ** 2)
            psnr = mse2psnr(loss)  # human-friendly quality metric (higher is better)

        """
        BACKPROP WITH GRADIENT SCALING (AMP)
        ------------------------------------
        Why scale the loss? In float16, very small gradients can underflow (become 0).
        GradScaler multiplies the loss by a large number before backward(), then
        divides gradients back afterward — improving numerical stability.

        Call order:
          1) optimizer.zero_grad(set_to_none=True)  → clear previous grads
          2) scaler.scale(loss).backward()         → compute scaled gradients
          3) scaler.step(optimizer)                → apply optimizer step if finite
          4) scaler.update()                       → adjust the scaling factor over time
        """

        # 6) Backprop with gradient scaling (stable AMP).
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        # 7) Console logs for quick monitoring.
        if (step + 1) % 100 == 0:
            print(f"[step {step+1:>5}] loss={loss.item():.5f}  psnr={psnr.item():.2f} dB")

        # 8) Periodically render a full image (slow but informative).
        if (step + 1) % PREVIEW_EVERY == 0:
            with torch.no_grad():
                pose_idx = (img_i + 1) % N
                img = render_image(poses[pose_idx], n_samples=N_SAMPLES)
                imageio.imwrite(
                    f"outputs/preview_{step+1:06d}.png",
                    (img.cpu().numpy() * 255).astype(np.uint8),
                )

    # 9) Save final checkpoint and a final image render.
    torch.save({"model": model.state_dict()}, "checkpoints/tinynerf_latest.pth")
    final = render_image(poses[-1], n_samples=N_SAMPLES)
    imageio.imwrite("outputs/final.png", (final.cpu().numpy() * 255).astype(np.uint8))

    dt_min = (time.time() - t0) / 60.0
    print(f"[done] {ITERS} iters in {dt_min:.2f} min  -> outputs/final.png")

# ==========================
# 9) FULL-IMAGE RENDER
# ==========================
@torch.no_grad()
def render_image(pose, n_samples=64, chunk=8192):
    """
    Render a full (H, W) image for a given camera pose by:
      1) generating one ray per pixel,
      2) sampling N points along each ray,
      3) predicting (rgb, sigma) with the MLP,
      4) volume-rendering along each ray,
      5) stitching all per-ray results into an image.

    We do this in *chunks* to limit peak GPU memory.

    ---------------------------------------------------------------
    WHY CHUNKING?
    ---------------------------------------------------------------
    A 100x100 image has 10,000 pixels, i.e., 10,000 rays.
    With n_samples=64, that's 640,000 3D points in one forward pass.
    Encoding + MLP + rendering for all of them at once can exceed GPU memory.
    So we split the rays into smaller batches ("chunks") and process them
    sequentially. The final image is just the concatenation of all chunks.

    ---------------------------------------------------------------
    INPUTS
    ---------------------------------------------------------------
    pose      : (4, 4) camera-to-world matrix for the desired viewpoint
    n_samples : number of depth samples per ray
    chunk     : number of rays processed at once (memory/speed trade-off)

    ---------------------------------------------------------------
    OUTPUT
    ---------------------------------------------------------------
    img : (H, W, 3) tensor in [0, 1]
        The rendered RGB image for the given pose.

    ---------------------------------------------------------------
    SHAPES INSIDE (per chunk)
    ---------------------------------------------------------------
      ro, rd                 : (R, 3)                 # R = chunk size (≤ H*W)
      z_vals                 : (R, n_samples)
      pts                    : (R, n_samples, 3)
      enc(pts.reshape(-1,3)) : (R*n_samples, enc_dim)
      rgb, sigma             : (R*n_samples, 3/1)  → reshaped to (R, n_samples, 3/1)
      comp_rgb               : (R, 3)

    ---------------------------------------------------------------
    MENTAL MODEL
    ---------------------------------------------------------------
    "Rendering an image" = "Rendering many rays."
    Each ray is independent, so chunking does not change the final image;
    it only controls how many rays we process at once to fit into memory.
    """
    model.eval()  # set to inference mode (affects e.g. dropout/batchnorm if present)

    # 1) Build all rays for this pose (one per pixel), then flatten to (H*W, 3).
    rays_o, rays_d = get_rays(H, W, focal, pose, device=device)  # (H*W,3), (H*W,3)

    # 2) We'll render in pieces and collect results here.
    out_rgb_chunks = []

    # 3) Process rays in chunks to control memory.
    for start in range(0, rays_o.shape[0], chunk):
        end = start + chunk
        ro = rays_o[start:end]  # (R, 3)
        rd = rays_d[start:end]  # (R, 3)

        # 4) Sample depths and compute 3D points along these rays.
        z_vals, pts = stratified_samples(NEAR, FAR, n_samples, ro, rd, randomized=False)

        # 5) Encode and predict for all points in this chunk (flatten → MLP → reshape).
        xenc = enc(pts.reshape(-1, 3))              # (R*n_samples, enc_dim)
        rgb, sigma = model(xenc)                    # flat predictions
        rgb   = rgb.reshape(ro.shape[0], n_samples, 3)
        sigma = sigma.reshape(ro.shape[0], n_samples, 1)

        # 6) Composite along the ray to get one RGB per ray in this chunk.
        comp_rgb, _, _, _ = volume_render(rgb, sigma, z_vals, rd)  # (R, 3)

        out_rgb_chunks.append(comp_rgb)

    # 7) Concatenate all chunks back to (H*W, 3) and reshape to image (H, W, 3).
    img = torch.cat(out_rgb_chunks, dim=0).reshape(H, W, 3).clamp(0.0, 1.0)
    return img
