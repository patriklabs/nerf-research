import torch

from torch import nn
from nerf.unisurf import NerfDensity
from nerf.unisurf import NerfColor
from nerf.util.util import ray_to_points, where


class DensityActivation(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        return self.sigmoid(-10.0 * x)


def integrate_ray_nerf(
    t: torch.Tensor, sigma, c, infinite: bool = False, normalize: bool = False
):

    dt = t[..., 1:, :] - t[..., :-1, :]

    # In the original imp the last distance is infinity.
    # Is this really correct since the integration is between
    # tn and tf where tf is not necessarily inf
    # practical consequence: at least the last color point will
    # receive a high weight even if the last sigma is only slightly positive.

    if infinite:
        dt = torch.cat((dt, 1e10 * torch.ones_like(dt[..., 0:1, :])), dim=-2)
    else:
        dt = torch.cat((dt, torch.zeros_like(dt[..., 0:1, :])), dim=-2)

    sdt = sigma * dt

    Ti = torch.exp(-torch.cumsum(sdt, dim=-2))[..., 0:-1, :]

    Ti = torch.cat((torch.ones_like(Ti[..., 0:1, :]), Ti), dim=-2)

    alpha = 1.0 - torch.exp(-sdt)

    wi = Ti * alpha

    if normalize:

        C = wi.sum(dim=-2, keepdim=True)

        C = where(C > 0, C, torch.ones_like(C))

        wi = wi / C

    return (wi * c).sum(dim=-2), (wi * t).sum(dim=-2), wi, t


def integrate_ray(t, sigma, color):

    # test = sigma.cpu().detach().numpy()

    alpha = torch.cat((torch.ones_like(sigma[..., 0:1, :]), 1.0 - sigma), dim=-2)

    wi = sigma * torch.cumprod(alpha, dim=-2)[..., :-1, :]

    return (wi * color).sum(dim=-2), (wi * t).sum(dim=-2), wi, t


class NerfRender(nn.Module):

    def __init__(self, Lp, Ld, homogeneous_projection, nerf_integration=False) -> None:
        super().__init__()

        self.nerf_density = NerfDensity(Lp, homogeneous_projection)

        self.nerf_color = NerfColor(Ld)

        if nerf_integration:
            self.density_activation = nn.ReLU()
        else:
            self.density_activation = DensityActivation()

        self.nerf_integration = nerf_integration

    def forward(self, ray, t, volumetric=True):

        t, _ = torch.sort(t, dim=-2)

        x, d = ray_to_points(ray, t)

        sigma, h, n = self.evaluate_density_gradient(x)

        color = self.evaluate_color(x, n, h, d)

        if volumetric:

            if self.nerf_integration:

                return integrate_ray_nerf(t, sigma, color)

            else:

                return integrate_ray(t, sigma, color)

        return color.squeeze(-2), t.squeeze(-2), None, t

    def evaluate_density_gradient(self, p):

        with torch.enable_grad():
            p.requires_grad_(True)
            y, h = self.nerf_density(p)
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            n = torch.autograd.grad(
                outputs=y,
                inputs=p,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
                allow_unused=True,
            )[0]

        n = n / (torch.norm(n, dim=-1, keepdim=True) + 1e-6)

        return self.density_activation(y), h, n

    def evaluate_color(self, x, n, h, d):

        return self.nerf_color(x, n, h, d)

    def evaluate(self, x, max_chunk=2048):

        sigma_list = []

        for pos in torch.split(x, max_chunk):

            sigma = self.evaluate_density(pos)

            sigma_list.append(sigma.cpu())

        return torch.cat(sigma_list, dim=0)

    def evaluate_density(self, x):

        return self.density_activation(self.nerf_density(x)[0])

    def evaluate_ray_density(self, ray, t):

        x, _ = ray_to_points(ray, t)

        return self.evaluate_density(x)

    def evaluate_normal(self, x):

        _, _, n = self.evaluate_density_gradient(x)

        return n
