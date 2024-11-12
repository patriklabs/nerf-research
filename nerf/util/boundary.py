from abc import ABC, abstractmethod
import torch
from nerf.util.util import (
    ray_sphere_intersection_distances_batch,
)


class RayBoundaryInterface(ABC):

    @abstractmethod
    def boundary(self, rays, tn, tf):
        raise NotImplementedError


class RayBoundaryIdentity(RayBoundaryInterface):

    def boundary(self, rays, tn, tf):
        return tn, tf


class RayBoundaryLimits(RayBoundaryInterface):

    def __init__(self, tn=0, tf=3.0) -> None:
        super().__init__()
        self.tn = tn
        self.tf = tf

    def boundary(self, rays, tn, tf):

        tn = self.tn * torch.ones_like(tn)
        tf = self.tf * torch.ones_like(tf)

        return tn, tf


class RayBoundaryUnitSphere(RayBoundaryInterface):

    def boundary(self, rays, tn, tf):
        o, d = torch.split(rays, [3, 3], dim=-1)

        tn, tf = ray_sphere_intersection_distances_batch(o, d, 0, 3)

        return tn, tf
