import torch

# Converts MSE to dB: PSNR = -10 * log10(MSE)

def mse2psnr(mse: torch.Tensor) -> torch.Tensor:
    return -10.0 * torch.log10(mse.clamp_min(1e-10))
