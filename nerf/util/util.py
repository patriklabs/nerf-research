from typing import List

import torch


def ray_sphere_intersection_distances_batch(orig, dir, tn_default, tf_default):
    """
    Calculate the distances to both intersections of rays with a unit sphere centered at the origin,
    for a batch of rays.

    Args:
        O (torch.Tensor): The origin points of the rays, shape (batch_size, 3).
        D (torch.Tensor): The direction vectors of the rays, shape (batch_size, 3).
        tn_default (float): Default distance to the first intersection for rays with no intersections.
        tf_default (float): Default distance to the second intersection for rays with no intersections.

    Returns:
        (torch.Tensor, torch.Tensor): Two tensors of shape (batch_size, 1), representing the distances to the first
                                    and second intersection points for each ray, or tn_default, tf_default for rays with no intersections.
    """
    # Coefficients of the quadratic equation
    a = torch.sum(dir**2, dim=1)
    b = 2 * torch.sum(orig * dir, dim=1)
    c = torch.sum(orig**2, dim=1) - 1

    # Calculate the discriminant
    discriminant = b**2 - 4 * a * c

    # Initialize distances with NaN for rays with no intersection
    t1_distances = torch.full_like(discriminant, tn_default)
    t2_distances = torch.full_like(discriminant, tf_default)

    # Case where discriminant == 0 (tangent intersection)
    tangent_intersection = discriminant == 0
    t_tangent = -b / (2 * a)
    t1_distances[tangent_intersection & (t_tangent >= 0)] = t_tangent[
        tangent_intersection & (t_tangent >= 0)
    ]
    t2_distances[tangent_intersection & (t_tangent >= 0)] = t_tangent[
        tangent_intersection & (t_tangent >= 0)
    ]

    # Case where discriminant > 0 (two intersections)
    two_intersections = discriminant > 0
    sqrt_discriminant = torch.sqrt(discriminant[two_intersections])
    t1 = (-b[two_intersections] - sqrt_discriminant) / (2 * a[two_intersections])
    t2 = (-b[two_intersections] + sqrt_discriminant) / (2 * a[two_intersections])

    # Assign both intersection distances
    t1_distances[two_intersections] = t1
    t2_distances[two_intersections] = t2

    t1_distances = t1_distances.clamp(tn_default, tf_default)
    t2_distances = t2_distances.clamp(tn_default, tf_default)

    return t1_distances.unsqueeze(-1), t2_distances.unsqueeze(-1)


def ray_to_points(ray, t) -> List[torch.Tensor]:
    """
    calculates the 3d points on a ray given the origin and direction of the ray and the distances along the ray

    Args:
        ray (torch.Tensor): The origin and direction of the ray, shape (batch_size, 6).
        t (torch.Tensor): The distances along the ray, shape (batch_size, num_samples, 1).

    Returns:
        List[torch.Tensor]: The 3d points on the ray, shape (batch_size, num_samples, 3).
    """

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
    """
    perform importance sampling on the weights w and the corresponding samples t

    Args:
        w (torch.Tensor): The weights of the samples, shape (batch_size, num_samples, 1).
        t (torch.Tensor): The samples, shape (batch_size, num_samples, 1).
        N (int): The number of samples to resample.

    Returns:
        torch.Tensor: The resampled samples, shape (batch_size, N, 1).
    """

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


def to_rays(T, intrinsics, tn, tf, B, H, W):
    """
    generates a grid of rays for a batch of cameras

    Args:

        T (torch.Tensor): The camera poses, shape (batch_size, 4, 4).
        intrinsics (torch.Tensor): The camera intrinsics, shape (batch_size, 4).
        tn (torch.Tensor): The near plane distances, shape (batch_size, 1).
        tf (torch.Tensor): The far plane distances, shape (batch_size, 1).
        B (int): The batch size.
        H (int): The image height.
        W (int): The image width.

    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor): The rays, the near plane distances, and the far plane distances.
    """

    rays = generate_rays(T, intrinsics, H, W)

    tn = tn.view(B, 1, 1, 1).expand(-1, 1, H, W)
    tf = tf.view(B, 1, 1, 1).expand(-1, 1, H, W)

    rays = rays.view(B, -1, H * W).permute(0, 2, 1).reshape(B * H * W, -1)
    tn = tn.view(B, -1, H * W).permute(0, 2, 1).reshape(B * H * W, -1)
    tf = tf.view(B, -1, H * W).permute(0, 2, 1).reshape(B * H * W, -1)

    return rays, tn, tf


def to_image(x, B, H, W):
    """
    reshapes a tensor to an image

    Args:
        x (torch.Tensor): The tensor to reshape.
        B (int): The batch size.
        H (int): The image height.
        W (int): The image width.

    Returns:
        torch.Tensor: The reshaped tensor.
    """

    x = x.reshape(B, H * W, -1).permute(0, 2, 1).view(B, -1, H, W)

    return x


def generate_grid(H, W, **kwargs):
    """
    generates a grid of pixel coordinates

    Args:
        H (int): The image height.
        W (int): The image width.

    Returns:
        torch.Tensor: The grid of pixel coordinates, shape (1, 2, H, W).
    """

    grid_x, grid_y = torch.meshgrid(
        torch.arange(0, W, **kwargs), torch.arange(0, H, **kwargs), indexing="xy"
    )

    return torch.stack((grid_x, grid_y)).unsqueeze(0)


def generate_rays(T, intrinsics, H, W):
    """
    generates a grid of rays for a batch of cameras

    Args:
        T (torch.Tensor): The camera poses, shape (batch_size, 4, 4).
        intrinsics (torch.Tensor): The camera intrinsics, shape (batch_size, 4).
        H (int): The image height.
        W (int): The image width.

    Returns:
        torch.Tensor: The rays, shape (batch_size, 6, H, W).
    """

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
