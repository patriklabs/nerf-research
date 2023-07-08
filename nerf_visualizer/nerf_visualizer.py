import numpy as np
import matplotlib.pyplot as plt
import torch
import mcubes
import os


class NerfVisualizer:
    def __init__(self, device, nerf_render, samples, threshold, name, x_min, x_max, y_min, y_max, z_min, z_max, **kwargs) -> None:
        self.device = device
        self.nerf_render = nerf_render
        self.limits = [x_min, x_max, y_min, y_max, z_min, z_max]
        self.nbr_samples = samples
        self.threshold = threshold
        self.name = name

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Visualizer")

        parser.add_argument("--samples", type=int, default=256)
        parser.add_argument("--threshold", type=float, default=0.5)
        parser.add_argument("--name", type=str, default="output_mesh")
        parser.add_argument("--x_min", type=float, default=-0.5)
        parser.add_argument("--y_min", type=float, default=-0.5)
        parser.add_argument("--z_min", type=float, default=-0.5)
        parser.add_argument("--x_max", type=float, default=0.5)
        parser.add_argument("--y_max", type=float, default=0.5)
        parser.add_argument("--z_max", type=float, default=0.5)

        return parent_parser

    def run(self):

        with torch.no_grad():

            x_min, x_max, y_min, y_max, z_min, z_max = self.limits

            x = np.linspace(x_min, x_max, self.nbr_samples)
            y = np.linspace(y_min, y_max, self.nbr_samples)
            z = np.linspace(z_min, z_max, self.nbr_samples)

            points = np.stack(np.meshgrid(x, y, z), -1)

            points = torch.tensor(points, device=self.device,
                                  dtype=torch.float32)

            H, W, D, C = points.shape

            sigma = self.nerf_render.evaluate(points.view(H*W*D, 1, C))

            sigma = sigma.view(H, W, D)

            sigma = sigma.detach().cpu().numpy()

            print('fraction occupied', np.mean(sigma > self.threshold))

            vertices, triangles = mcubes.marching_cubes(sigma, self.threshold)

            scale = np.array(
                [x_max-x_min, y_max-y_min, z_max-z_min])/(self.nbr_samples-1)
            scale = scale[None, :]

            offset = np.array([x_min, y_min, z_min])
            offset = offset[None, :]

            vertices = scale*vertices + offset

            print('done', vertices.shape, triangles.shape)

            model_folder = "model3d/example"

            if not os.path.exists(model_folder):
                os.makedirs(model_folder)

            mcubes.export_mesh(vertices, triangles, os.path.join(
                model_folder, "{}_{}.dae".format(self.name, self.nbr_samples)), self.name)

            plt.hist(sigma.ravel(), log=True)
            plt.show()
