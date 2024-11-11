import torch
from torch import nn

from nerf.nerf import NerfColor
from nerf.neus.nerf_density import NerfDensity
from nerf.util.util import ray_to_points
from nerf.nerf.nerf_render import integrate_ray


def integrate_ray_unbiased(
    t: torch.Tensor, sdf: torch.Tensor, color: torch.Tensor, s_inv_log: torch.Tensor
):

    q = (sdf * torch.exp(-s_inv_log)).sigmoid()

    alpha = ((q[..., :-1, :] - q[..., 1:, :] + 1e-5) / (q[..., :-1, :] + 1e-5)).clip(
        0.0, 1.0
    )

    Ti = torch.cumprod(1.0 - alpha, dim=-2)

    Ti = torch.cat((torch.ones_like(Ti[..., 0:1, :]), Ti[..., 0:-1, :]), dim=-2)

    wi = Ti * alpha

    return (
        (wi * color[..., :-1, :]).sum(dim=-2),
        (wi * t[..., :-1, :]).sum(dim=-2),
        wi,
        t[..., :-1, :],
    )


def logistic_density_function(x, s_inv_log):

    s = torch.exp(-s_inv_log)

    sigma = s * torch.exp(-s * x) / (1 + torch.exp(-s * x)) ** 2

    return sigma


def integrate_ray_biased(
    t: torch.Tensor, sdf: torch.Tensor, color: torch.Tensor, s_inv_log: torch.Tensor
):
    return integrate_ray(t, logistic_density_function(sdf, s_inv_log), color)


class NerfRender(nn.Module):

    def __init__(self, Lp, Ld, homogeneous_projection, biased_integration) -> None:
        super().__init__()

        self.nerf_density = NerfDensity(Lp, homogeneous_projection)

        self.nerf_color = NerfColor(Ld)

        self.biased_integration = biased_integration

    def evaluate_ray(self, ray, t, return_eikonal_loss=True):

        x, d = ray_to_points(ray, t)

        if self.training and return_eikonal_loss:
            x.requires_grad_(True)

        sdf_values, F = self.nerf_density(x)

        eikonal_loss = 0

        if self.training and return_eikonal_loss:
            d_output = torch.ones_like(sdf_values)
            gradients = torch.autograd.grad(
                outputs=sdf_values,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            gradient_norms = torch.linalg.norm(gradients, ord=2, dim=-1)

            eikonal_loss = ((gradient_norms - 1) ** 2).mean()

        color = self.nerf_color(F, d)

        return sdf_values, color, eikonal_loss

    def forward(self, ray, t, s_inv_log, return_eikonal_loss=True):

        t, _ = torch.sort(t, dim=-2)

        sigma, color, eikonal_loss = self.evaluate_ray(ray, t, return_eikonal_loss)

        if self.biased_integration:
            return integrate_ray_biased(t, sigma, color, s_inv_log), eikonal_loss
        else:
            return integrate_ray_unbiased(t, sigma, color, s_inv_log), eikonal_loss

    def evaluate(self, x, max_chunk=2048):

        sigma_list = []

        for pos in torch.split(x, max_chunk):

            sigma = self.evaluate_density(pos)

            sigma_list.append(sigma.cpu())

        return torch.cat(sigma_list, dim=0)

    def evaluate_density(self, x):

        return self.nerf_density(x)[0]
