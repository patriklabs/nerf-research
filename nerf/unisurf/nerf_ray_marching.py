
from torch import nn
from distutils.util import strtobool
from nerf.unisurf.nerf_render import NerfRender
from nerf.unisurf.ray_marching import find_intersection
from nerf.util.util import uniform_sample, where, ray_to_points
import torch
import math


class Nerf(nn.Module):
    def __init__(self, Lp, Ld, marching_bins, bins, intersection_itr, tau, noise, delta_max, delta_min, beta, homogeneous_projection, **kwargs) -> None:
        super().__init__()

        self.marching_bins = marching_bins

        self.bins = bins
        self.intersection_itr = intersection_itr
        self.tau = tau
        self.noise = noise
        self.delta_max = delta_max
        self.delta_min = delta_min
        self.beta = beta

        self.render = NerfRender(Lp, Ld, homogeneous_projection)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Nerf")

        parser.add_argument("--Lp", type=int, default=10)
        parser.add_argument("--Ld", type=int, default=4)
        parser.add_argument("--marching_bins", type=int, default=256)
        parser.add_argument("--bins", type=int, default=64)
        parser.add_argument("--intersection_itr", type=int, default=8)

        parser.add_argument("--tau", type=float, default=0.5)
        parser.add_argument("--noise", type=float, default=1e-4)
        parser.add_argument("--delta_max", type=float, default=1)
        parser.add_argument("--delta_min", type=float, default=0.05)
        parser.add_argument("--beta", type=float, default=1e-4)

        parser.add_argument("--homogeneous_projection",
                            type=strtobool, default=True)

        return parent_parser

    def forward(self, ray, tn, tf, step):

        with torch.no_grad():

            ts, mask_ts = find_intersection(ray, tn, tf, self.render,
                                            self.tau, self.marching_bins,
                                            self.intersection_itr)

        if self.training:

            tu = uniform_sample(tn, tf, self.bins)

            delta = max(self.delta_max * math.exp(-step*self.beta),
                        self.delta_min)

            delta = (tf-tn)*delta

            tc = uniform_sample(ts.squeeze(-1)-delta,
                                ts.squeeze(-1)+delta, self.bins)

            mask = mask_ts.logical_and(tc >= 0)
            t_smp = where(mask, tc, tu)

            tu = uniform_sample(tn, ts.squeeze(-1), self.bins//2)

            t = torch.cat((tu, t_smp), dim=-2)

            color_high_res, depth, _, _ = self.render.forward(ray, t)

        else:

            color_high_res, depth, _, _ = self.render.forward(
                ray, ts, volumetric=False)

        x_s = ray_to_points(ray, ts)[0]

        n1 = self.render.evaluate_normal(x_s)
        n2 = self.render.evaluate_normal(x_s+self.noise*torch.randn_like(x_s))

        reg_val = (n1-n2).square().sum(-1)

        return {"color_high_res": color_high_res, "depth": depth, "reg_val": reg_val}
