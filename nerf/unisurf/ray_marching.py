import torch

from nerf.util.util import linspace, where


def find_intersection(rays, tn, tf, model, tau=0.5, N=256, itr=8):

    t = linspace(tn, tf, N)

    sigma = model.evaluate_ray_density(rays, t)

    indices, mask = find_intersection_idx_(sigma, tau)

    # test = sigma.cpu().detach().numpy()

    t1 = torch.gather(t, -2, indices)
    f1 = torch.gather(sigma - tau, -2, indices)

    idx_next = (indices + 1).clamp_max(N - 1)

    t2 = torch.gather(t, -2, idx_next)
    f2 = torch.gather(sigma - tau, -2, idx_next)

    ts, f_mid = bisection_(rays, t1, t2, f1, f2, model, tau, itr)

    return ts, mask


def secant_(rays, t1, t2, f1, f2, model, tau, itr):

    # f(t) = sigma(t) - tau

    for i in range(itr):
        f_delta = f2 - f1

        f_delta = where(f_delta.abs() > 0, f_delta, torch.ones_like(f_delta))

        t_next = t2 - f2 * (t2 - t1) / f_delta
        f_next = model.evaluate_ray_density(rays, t_next) - tau

        t1 = t2
        f1 = f2

        t2 = t_next
        f2 = f_next

    return t_next, f_next


def bisection_(rays, t1, t2, f1, f2, model, tau, itr):

    # f(t) = sigma(t) - tau

    f1_sign = torch.sign(f1)
    # f2_sign = torch.sign(f2)

    for i in range(itr):

        t_mid = (t2 + t1) / 2

        f_mid = model.evaluate_ray_density(rays, t_mid) - tau

        f_mid_sign = torch.sign(f_mid)

        mask = (f1_sign == f_mid_sign).float()

        t1 = mask * t_mid + (1.0 - mask) * t1
        t2 = (1.0 - mask) * t_mid + mask * t2

    return t_mid, f_mid


def find_intersection_idx_(sigma, tau):

    val = sigma - tau

    mask_valid_points = val < 0

    # find first sign change from negatvie to positvie (if any)

    sign_change = torch.sign(val[..., :-1, :] * val[..., 1:, :])
    sign_change = torch.cat(
        (sign_change, torch.ones_like(sign_change[..., :1, :])), dim=-2
    )
    # only introduce sign change from negative to positive
    sign_change = where(mask_valid_points, sign_change, torch.ones_like(sign_change))

    sign_nbr = (
        torch.arange(
            sigma.shape[1], 0, -1, device=sign_change.device, dtype=sign_change.dtype
        )
        .unsqueeze(0)
        .unsqueeze(-1)
    )

    sign_change_idx = sign_change * sign_nbr

    values, indices = torch.min(sign_change_idx, dim=-2, keepdim=True)

    mask = values < 0

    return indices, mask
