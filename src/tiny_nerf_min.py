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
