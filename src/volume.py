import torch

def volume_render(rgb, sigma, z_vals, rays_d, white_bkgd=True):
    """
    Inputs:
      rgb:    (N_rays, N_samples, 3) in [0,1]
      sigma:  (N_rays, N_samples, 1) density >= 0
      z_vals: (N_rays, N_samples)
      rays_d: (N_rays, 3)
    Returns:
      comp_rgb: (N_rays, 3)
      depth:    (N_rays, 1)
      acc:      (N_rays, 1) accumulated opacity
    """
    device = rgb.device
    deltas = z_vals[..., 1:] - z_vals[..., :-1]                         # (N_rays, N_samples-1)
    # infinity delta for the last sample
    delta_inf = 1e10 * torch.ones_like(deltas[..., :1])
    deltas = torch.cat([deltas, delta_inf], dim=-1)                     # (N_rays, N_samples)
    # Account for ray length (|d|)
    deltas = deltas * torch.linalg.norm(rays_d[:, None, :], dim=-1)

    # Volumetric rendering
    alpha = 1. - torch.exp(-sigma.squeeze(-1) * deltas)                 # (N_rays, N_samples)
    # Accumulated transmittance T
    # cumprod_exclusive: prepend 1 then cumprod without last element
    eps = 1e-10
    accum_prod = torch.cumprod(1. - alpha + eps, dim=-1)
    trans = torch.cat([torch.ones_like(accum_prod[..., :1]), accum_prod[..., :-1]], dim=-1)

    weights = alpha * trans                                             # (N_rays, N_samples)
    comp_rgb = torch.sum(weights[..., None] * rgb, dim=-2)              # (N_rays, 3)
    depth = torch.sum(weights * z_vals, dim=-1, keepdim=True)           # (N_rays, 1)
    acc = torch.sum(weights, dim=-1, keepdim=True)                      # (N_rays, 1)

    if white_bkgd:
        comp_rgb = comp_rgb + (1. - acc)  # white background

    return comp_rgb, depth, acc, weights
