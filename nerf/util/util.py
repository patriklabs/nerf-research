from typing import List

import torch


def ray_to_points(ray, t) -> List[torch.Tensor]:

    o, d = torch.split(ray.unsqueeze(-2), [3, 3], dim=-1)

    x = o + t * d

    return x, d


def where(mask, x, y):

    mask = mask.float()

    return mask * x + (1.0 - mask) * y


def linspace(tn, tf, N):

    dt = (tf - tn).unsqueeze(-1)
    tn = tn.unsqueeze(1)

    i = (
        torch.linspace(0, 1, N, device=tn.device, dtype=tn.dtype)
        .unsqueeze(0)
        .unsqueeze(-1)
    )

    return tn + dt * i


def uniform(a, b):

    return a + (b - a) * torch.rand_like(a)


def uniform_sample(tn, tf, N: int):

    dt = (tf - tn).unsqueeze(-1) / N
    tn = tn.unsqueeze(-1)

    i = (
        torch.arange(1, N + 1, device=tn.device, dtype=tn.dtype)
        .unsqueeze(0)
        .unsqueeze(-1)
    )

    a = tn + (i - 1) * dt
    b = tn + i * dt

    return uniform(a, b)


def resample_new(w, t, N):

    c = w.sum(1, keepdim=True)
    w = w / where(c > 0, c, torch.ones_like(c))

    cdf = torch.cumsum(w, dim=1)

    cdf = torch.cat((torch.zeros_like(cdf[:, :1]), cdf), dim=1)

    B, S = w.shape[0:2]

    u = torch.rand((B, N), device=w.device, dtype=w.dtype)

    idx = torch.searchsorted(cdf.squeeze(-1), u, right=True).unsqueeze(-1)

    cdf1 = torch.gather(cdf, 1, (idx - 1).clamp(0, S - 1))
    w1 = torch.gather(w, 1, (idx - 1).clamp(0, S - 1))
    t1 = torch.gather(t, 1, (idx - 1).clamp(0, S - 1))

    t2 = torch.gather(t, 1, idx.clamp(0, S - 1))

    # cdf2 = torch.gather(cdf, 1, (idx).clamp(0, S-1))

    u = u.unsqueeze(-1)

    # u = k*t + m
    w1 = w1
    w1 = where(w1 > 0, w1, torch.ones_like(w1))
    m = cdf1

    tu = (u - m) / w1

    t = t1 + tu * (t2 - t1)

    return t.clamp(t1, t2)


def resample(w, t, N: int):

    t1 = t[:, :-1]
    t2 = t[:, 1:]

    w1 = w[:, :-1]
    w2 = w[:, 1:]

    delta_t = t2 - t1

    k = (w2 - w1) / where(delta_t > 0, delta_t, torch.ones_like(delta_t))
    m = w1 - k * t1

    c1 = 0.5 * k * (t1**2) + m * t1
    c2 = 0.5 * k * (t2**2) + m * t2

    c = c2 - c1

    c = torch.cat((torch.zeros_like(c[:, 0:1]), c), dim=1)

    w_cdf = torch.cumsum(c, dim=1)

    C = w_cdf[:, -1:]
    C = where(C > 0, C, torch.ones_like(C))

    w_cdf = w_cdf / C

    B, S = w_cdf.shape[0:2]

    u = torch.rand((B, N), device=w.device, dtype=w.dtype)

    idx = (
        torch.searchsorted(w_cdf.squeeze(-1), u, right=True)
        .unsqueeze(-1)
        .clamp(0, S - 1)
    )

    u = u.unsqueeze(-1)

    Q = torch.gather(w_cdf, 1, (idx - 1).clamp(0, S - 1))
    t1 = torch.gather(t, 1, (idx - 1).clamp(0, S - 1))

    w1 = torch.gather(w, 1, (idx - 1).clamp(0, S - 1))

    w2 = torch.gather(w, 1, idx)
    t2 = torch.gather(t, 1, idx)

    delta_t = t2 - t1

    k = (w2 - w1) / where(delta_t > 0, delta_t, torch.ones_like(delta_t))
    m = w1 - k * t1

    k = k / C
    m = m / C

    c1 = 0.5 * k * (t1**2) + m * t1

    u = u - Q + c1

    a = 0.5 * k
    b = m
    c = -u

    mask1 = a.abs() > 0

    a = where(a.abs() > 0, a, torch.ones_like(a))

    qq = (b**2 - 4 * a * c).clamp_min(0)

    tu1 = (-b + torch.sqrt(qq)) / (2 * a)
    # tu2 = (-b-torch.sqrt(qq))/(2*a)

    mask2 = b.abs() > 0

    b = where(b.abs() > 0, b, torch.ones_like(b))

    tu3 = -c / b

    tu = where(mask1, tu1, tu3)
    tu = where(mask2.logical_or(mask1), tu, torch.zeros_like(tu))

    # tu = t1 + tu

    return tu.clamp(t1, t2)


def resample_old(w, t, N: int):

    t1 = t[:, :-1]
    t2 = t[:, 1:]

    w1 = w[:, :-1]
    w2 = w[:, 1:]

    delta_t = t2 - t1

    k = (w2 - w1) / where(delta_t > 0, delta_t, torch.ones_like(delta_t))
    m = w1

    c = 0.5 * k * (delta_t**2) + m * delta_t

    c = torch.cat((torch.zeros_like(c[:, 0:1]), c), dim=1)

    w_cdf = torch.cumsum(c, dim=1)

    C = w_cdf[:, -1:]
    C = where(C > 0, C, torch.ones_like(C))

    w_cdf = w_cdf / C

    B, S = w_cdf.shape[0:2]

    u = torch.rand((B, N), device=w.device, dtype=w.dtype)

    idx = (
        torch.searchsorted(w_cdf.squeeze(-1), u, right=True)
        .unsqueeze(-1)
        .clamp(0, S - 1)
    )

    u = u.unsqueeze(-1)

    Q = torch.gather(w_cdf, 1, (idx - 1).clamp(0, S - 1))
    t1 = torch.gather(t, 1, (idx - 1).clamp(0, S - 1))

    w1 = torch.gather(w, 1, (idx - 1).clamp(0, S - 1))

    w2 = torch.gather(w, 1, idx)
    t2 = torch.gather(t, 1, idx)

    delta_t = t2 - t1

    k = (w2 - w1) / where(delta_t > 0, delta_t, torch.ones_like(delta_t))
    m = w1

    u = u - Q

    a = 0.5 * k / C
    b = m / C
    c = -u

    mask1 = a.abs() > 0

    a = where(a.abs() > 0, a, torch.ones_like(a))

    qq = (b**2 - 4 * a * c).clamp_min(0)

    tu1 = (-b + torch.sqrt(qq)) / (2 * a)

    mask2 = b.abs() > 0

    b = where(b.abs() > 0, b, torch.ones_like(b))

    tu3 = -c / b

    tu = where(mask1, tu1, tu3)
    tu = where(mask2.logical_or(mask1), tu, torch.zeros_like(tu))

    tu = t1 + tu

    return tu.clamp(t1, t2)


def to_rays(T, intrinsics, tn, tf, B, H, W):

    rays = generate_rays(T, intrinsics, H, W)

    tn = tn.view(B, 1, 1, 1).expand(-1, 1, H, W)
    tf = tf.view(B, 1, 1, 1).expand(-1, 1, H, W)

    rays = rays.view(B, -1, H * W).permute(0, 2, 1).reshape(B * H * W, -1)
    tn = tn.view(B, -1, H * W).permute(0, 2, 1).reshape(B * H * W, -1)
    tf = tf.view(B, -1, H * W).permute(0, 2, 1).reshape(B * H * W, -1)

    return rays, tn, tf


def to_image(x, B, H, W):

    x = x.reshape(B, H * W, -1).permute(0, 2, 1).view(B, -1, H, W)

    return x


def generate_grid(H, W, **kwargs):

    grid_x, grid_y = torch.meshgrid(
        torch.arange(0, W, **kwargs), torch.arange(0, H, **kwargs), indexing="xy"
    )

    return torch.stack((grid_x, grid_y)).unsqueeze(0)


def generate_rays(T, intrinsics, H, W):

    grid = generate_grid(H, W, device=intrinsics.device, dtype=intrinsics.dtype)

    R = T[:, 0:3, 0:3]
    t = T[:, 0:3, 3:]

    focal, prinicpal = torch.split(intrinsics.view(-1, 4, 1, 1), [2, 2], dim=1)

    d = (grid - prinicpal) / focal

    d = torch.cat((d, torch.ones_like(d[:, 0:1])), dim=1)

    B, C, H, W = d.shape

    d = (torch.transpose(R, -2, -1) @ d.view(B, C, H * W)).view(B, C, H, W)

    d = d / torch.linalg.norm(d, dim=1, keepdim=True)

    o = -torch.transpose(R, -2, -1) @ t

    o = o.view(-1, 3, 1, 1).expand(-1, -1, H, W)

    ray = torch.cat((o, d), dim=1)

    return ray
