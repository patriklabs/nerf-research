from torch import nn
from distutils.util import strtobool
from nerf.unisurf.nerf_render import NerfRender
from nerf.util.util import uniform_sample, resample
import torch


class Nerf(nn.Module):
    def __init__(
        self,
        Lp=10,
        Ld=4,
        low_res_bins=64,
        high_res_bins=64,
        homogeneous_projection=True,
        nerf_integration=True,
        **kwargs
    ) -> None:
        super().__init__()

        self.low_res_bins = low_res_bins

        self.high_res_bins = high_res_bins

        self.render = NerfRender(Lp, Ld, homogeneous_projection, nerf_integration)

        self.render_low_res = NerfRender(
            Lp, Ld, homogeneous_projection, nerf_integration
        )

    def forward(self, rays, tn, tf, step):

        t = uniform_sample(tn, tf, self.low_res_bins)

        # do one round to find out important sampling regions

        color_low_res, _, w, t = self.render_low_res.forward(rays, t)

        with torch.no_grad():

            # sample according to w
            t_resamp = resample(w, t, self.high_res_bins)

        t_resamp = torch.cat((t, t_resamp), dim=1)

        color_high_res, depth, _, _ = self.render.forward(rays, t_resamp)

        return {
            "color_high_res": color_high_res,
            "depth": depth,
            "color_low_res": color_low_res,
        }
