import torch

def stratified_samples(near, far, n_samples, rays_o, rays_d, randomized=True):
    """
    Uniformly sample n_samples points between [near, far] along each ray.
    Inputs:
      near, far: floats or tensors broadcastable to (N_rays, 1)
      rays_o, rays_d: (N_rays, 3)
    Returns:
      z_vals: (N_rays, n_samples)
      pts:    (N_rays, n_samples, 3)
    """
    N_rays = rays_o.shape[0]
    device = rays_o.device
    t_vals = torch.linspace(0., 1., steps=n_samples, device=device)
    z_vals = near * (1. - t_vals) + far * t_vals  # (n_samples,)
    z_vals = z_vals.expand(N_rays, n_samples)

    if randomized:
        mids = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
        upper = torch.cat([mids, z_vals[:, -1:]], -1)
        lower = torch.cat([z_vals[:, :1], mids], -1)
        t_rand = torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., None]
    return z_vals, pts
