import torch
import torch.nn as nn

class TinyNeRF(nn.Module):
    """
    Minimal NeRF-style MLP (no viewdirs yet):
      input: encoded 3D xyz (PositionalEncoding.out_dim)
      output: rgb (3 in [0,1] via sigmoid), sigma (density, >=0 via ReLU)
    """
    def __init__(self, in_dim: int, hidden: int = 128, depth: int = 4, skip_at: int = 2):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.depth = depth
        self.skip_at = skip_at

        # Build linear layers with one skip concat after `skip_at`
        layers = []
        last = in_dim
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(nn.Linear(last, hidden))
            # after the skip layer, the next layer will see hidden + in_dim
            last = hidden if i != (skip_at - 1) else (hidden + in_dim)

        self.sigma = nn.Sequential(nn.Linear(hidden, 1), nn.ReLU(inplace=True))
        self.rgb   = nn.Sequential(nn.Linear(hidden, 3), nn.Sigmoid())

    def forward(self, x):
        """
        x: (N, in_dim) encoded coordinates
        returns: rgb (N,3), sigma (N,1)
        """
        h = x
        for i, lin in enumerate(self.layers):
            h = torch.relu(lin(h))
            if i == (self.skip_at - 1):
                h = torch.cat([h, x], dim=-1)  # skip connection
        rgb = self.rgb(h)
        sigma = self.sigma(h)
        return rgb, sigma
