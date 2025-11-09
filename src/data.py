import numpy as np
from typing import Dict, Any

def load_tiny_nerf_npz(path: str = "data/tiny_nerf_data.npz") -> Dict[str, Any]:
    """
    Loads the standard tiny_nerf_data.npz.
    Returns a dict with keys like: 'images', 'poses', 'focal', etc.
    """
    data = np.load(path)
    # Normalize to float32 to avoid accidental float64 everywhere.
    out = {k: (v.astype(np.float32) if hasattr(v, "dtype") and v.dtype == np.float64 else v)
           for k, v in data.items()}
    return out
