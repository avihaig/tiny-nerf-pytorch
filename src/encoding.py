import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Standard NeRF Fourier features for 3D coords.
    If L=10, output dim = 3*(2*L) + (include_input?3:0) = 63 by default.
    """
    def __init__(self, num_freqs: int = 10, include_input: bool = True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        # Precompute frequency bands: [1, 2, 4, ..., 2^{L-1}]
        self.register_buffer("freq_bands", 2.0 ** torch.arange(num_freqs).float())

    @property
    def out_dim(self) -> int:
        base = 3 * 2 * self.num_freqs
        return base + (3 if self.include_input else 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., 3)
        returns: (..., out_dim)
        """
        assert x.shape[-1] == 3, "PositionalEncoding expects (..., 3)"
        enc = []
        if self.include_input:
            enc.append(x)
        for f in self.freq_bands:
            enc.append(torch.sin(x * f))
            enc.append(torch.cos(x * f))
        return torch.cat(enc, dim=-1)
