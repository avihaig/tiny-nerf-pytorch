import torch

# Converts MSE to dB: PSNR = -10 * log10(MSE)
"""
What PSNR numbers mean

~10–15 dB at the very start (random).

~23–26 dB after a few thousand steps for TinyNeRF on this data (without hierarchical sampling or view-dirs).

Higher with longer training and improvements.
"""

def mse2psnr(mse: torch.Tensor) -> torch.Tensor:
    return -10.0 * torch.log10(mse.clamp_min(1e-10))
