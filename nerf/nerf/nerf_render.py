import torch
from torch import nn

from nerf.nerf import NerfColor, NerfDensity
from nerf.util.util import ray_to_points, where


def integrate_ray(t: torch.Tensor, sigma, color, infinite: bool = False):

    dt = t[..., 1:, :] - t[..., :-1, :]

    # In the original imp the last distance is infinity.
    # Is this really correct since the integration is between
    # tn and tf where tf is not necessarily inf
    # practical consequence: at least the last color point will
    # receive a high weight even if the last sigma is only slightly positive.

    if infinite:
        dt = torch.cat((dt, 1e10*torch.ones_like(dt[..., 0:1, :])), dim=-2)
    else:
        dt = torch.cat((dt, torch.zeros_like(dt[..., 0:1, :])), dim=-2)

    sdt = sigma*dt

    Ti = torch.exp(-torch.cumsum(sdt, dim=-2))[..., 0:-1, :]

    Ti = torch.cat((torch.ones_like(Ti[..., 0:1, :]), Ti), dim=-2)

    alpha = (1.0 - torch.exp(-sdt))

    wi = Ti*alpha

    return (wi*color).sum(dim=-2), (wi*t).sum(dim=-2), wi, t


class NerfRender(nn.Module):

    def __init__(self, Lp, Ld, homogeneous_projection) -> None:
        super().__init__()

        self.nerf_density = NerfDensity(Lp, homogeneous_projection)

        self.nerf_color = NerfColor(Ld)

    def evaluate_ray(self, ray, t):

        x, d = ray_to_points(ray, t)

        sigma, F = self.nerf_density(x)

        color = self.nerf_color(F, d)

        return sigma, color

    def forward(self, ray, t):

        t, _ = torch.sort(t, dim=-2)

        sigma, color = self.evaluate_ray(ray, t)

        return integrate_ray(t, sigma, color)

    def evaluate(self, x, max_chunk=2048):

        sigma_list = []

        for pos in torch.split(x, max_chunk):

            sigma = self.evaluate_density(pos)

            sigma_list.append(sigma.cpu())

        return torch.cat(sigma_list, dim=0)

    def evaluate_density(self, x):

        return self.nerf_density(x)[0]
