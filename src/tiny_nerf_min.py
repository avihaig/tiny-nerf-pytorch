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
