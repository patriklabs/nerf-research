from distutils.util import strtobool

import torch
from torch import nn

from nerf.nerf.nerf_render import NerfRender
from nerf.util.util import resample, uniform_sample


class Nerf(nn.Module):
    def __init__(
        self,
        Lp=10,
        Ld=4,
        low_res_bins=64,
        high_res_bins=128,
        homogeneous_projection=True,
        **kwargs
    ) -> None:
        super().__init__()

        self.low_res_bins = low_res_bins

        self.high_res_bins = high_res_bins

        self.render = NerfRender(Lp, Ld, homogeneous_projection)

        self.render_low_res = NerfRender(Lp, Ld, homogeneous_projection)

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
