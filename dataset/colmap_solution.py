import os
import random

import numpy as np
import torch
from PIL import Image as pil_image
from torch import nn

from dataset.util import to_homogen
from thirdparty.colmap.scripts.python.read_write_model import (
    Camera,
    Image,
    Point3D,
    read_model,
    rotmat2qvec,
)
from util.ply import write_ply
from util.statistics import reject_outliers


def image_to_projection_matrix(image: Image):

    R = image.qvec2rotmat()

    t = image.tvec

    P = np.eye(3, 4)
    P[0:3, 0:3] = R
    P[0:3, 3] = t

    return P


def image_to_camera_center(image: Image):

    R = image.qvec2rotmat()

    t = image.tvec

    P = np.eye(4, 4)
    P[0:3, 0:3] = R
    P[0:3, 3] = t

    return np.linalg.inv(P)[0:3, 3]


def projection_matrix_to_image(P, image: Image):

    q = rotmat2qvec(P[0:3, 0:3])
    t = P[0:3, 3]

    return Image(
        id=image.id,
        qvec=q,
        tvec=t,
        camera_id=image.camera_id,
        name=image.name,
        xys=image.xys,
        point3D_ids=image.point3D_ids,
    )


def intrinsic_vec_to_matrix(intrinsics):

    fx, fy, cx, cy = intrinsics

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    return K


import numpy as np


def sphere_point_cloud(radius=1, num_points=1000):
    """
    Generate a point cloud on the surface of a sphere.

    Parameters:
    - radius: Radius of the sphere (default is 1).
    - num_points: Number of points to generate (default is 1000).

    Returns:
    - A NumPy array of shape (num_points, 3) representing the 3D coordinates of points on the sphere.
    """
    # Generate random angles for spherical coordinates
    phi = np.random.uniform(0, np.pi, num_points)  # Angle from the z-axis (0 to pi)
    theta = np.random.uniform(
        0, 2 * np.pi, num_points
    )  # Angle from the x-axis (0 to 2*pi)

    # Convert spherical coordinates to Cartesian coordinates
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    # Combine x, y, z coordinates into a (num_points, 3) array
    points = np.vstack((x, y, z)).T

    return points


def print_points(points, path):

    sphere = sphere_point_cloud(1, 1000)

    points = np.concatenate([points, sphere], axis=0)

    write_ply(points, np.zeros_like(points), path)


class ColmapSolution:

    def __init__(self, path, sol_nbr) -> None:
        self.root_path = path
        self.cameras, self.images, self.points = read_model(
            os.path.join(path, "sparse")
        )

        self.cameras = {
            key: self.convert_camera(camera) for key, camera in self.cameras.items()
        }
        self.img_keys = list(self.images.keys())

        self.error = self.calculate_error()

    def calculate_error(self):

        avg_error = 0
        terms = 0
        for key, point in self.points.items():

            for image_id in point.image_ids:
                image = self.images[image_id]
                camera = self.cameras[image.camera_id]

                if camera.model == "PINHOLE":
                    intrinsics = camera.params
                else:
                    raise Exception("camera model not recognized")

                K = intrinsic_vec_to_matrix(intrinsics)
                P = image_to_projection_matrix(self.images[image_id])

                A = K @ P

                X = to_homogen(point.xyz)

                x = A @ X

                x = x / x[2]

                ids3d = image.point3D_ids

                index = np.where(ids3d == key)[0]

                for idx in index:
                    x_ref = image.xys[idx.item()]

                    error_term = np.linalg.norm(x[0:2] - x_ref)

                    avg_error = avg_error * terms / (terms + 1) + error_term / (
                        terms + 1
                    )
                    terms += 1

        return avg_error

    def unit_transformations(self):

        points = [point.xyz for k, point in self.points.items()]
        # camera_o = [image_to_camera_center(image) for k, image in self.images.items()]

        # points.extend(camera_o)

        x = np.array(points)
        x = np.transpose(x, (1, 0))

        x_mean = np.mean(x, axis=-1, keepdims=True)

        x_cen = x - x_mean

        dists = np.linalg.norm(x_cen[0:3], axis=0)

        dists = reject_outliers(dists)

        s = np.max(dists, keepdims=True)

        # s = iqr(np.abs(x_cen).reshape(-1))[1]

        T = np.eye(4, 4)

        T[0:3, 0:3] = s * np.eye(3, 3)
        T[0:3, 3:] = x_mean

        Tinv = np.linalg.inv(T)

        T = 1 / s * T

        ### test ###
        """

        x = to_homogen(x)
        x = Tinv@x

        x_mean = np.mean(x, axis=-1, keepdims=True)

        x_cen = x - x_mean

        s = np.max(np.linalg.norm(x_cen[0:3], axis=0), keepdims=True)
        """
        return T, Tinv

    def unit_rescale(self):

        error_before = self.calculate_error()

        T, Tinv = self.unit_transformations()

        for key, image in self.images.items():

            P = image_to_projection_matrix(image)

            P = P @ T

            self.images[key] = projection_matrix_to_image(P, image)

        for key, point in self.points.items():

            x = to_homogen(point.xyz)
            x = Tinv @ x

            self.points[key] = Point3D(
                id=point.id,
                xyz=x[0:3],
                rgb=point.rgb,
                error=point.error,
                image_ids=point.image_ids,
                point2D_idxs=point.point2D_idxs,
            )

        error_after = self.calculate_error()

        self.print()

        np.testing.assert_allclose(error_before, error_after, atol=1e-6)

    def print(self):

        xx = np.array([point.xyz for key, point in self.points.items()])

        print_points(xx, os.path.join(self.root_path, "points.ply"))

    def rescale_image(self, size):

        Ho, Wo = size

        for key, camera in self.cameras.items():
            W = camera.width
            H = camera.height

            fx, fy, cx, cy = camera.params
            ax = Wo / W
            ay = Ho / H

            intrinsics = np.array([ax * fx, ay * fy, ax * cx, ay * cy])

            self.cameras[key] = Camera(
                id=camera.id, model=camera.model, width=Wo, height=Ho, params=intrinsics
            )

    def convert_camera(self, camera: Camera):

        intrinsics = None

        if camera.model == "SIMPLE_RADIAL":
            f, cx, cy, k1 = camera.params
            # assuming that k1 is small enough to be ignored!
            intrinsics = np.array([f, f, cx, cy])

        elif camera.model == "PINHOLE":
            intrinsics = camera.params
        else:
            raise Exception("camera model not recognized")

        return Camera(
            id=camera.id,
            model="PINHOLE",
            width=camera.width,
            height=camera.height,
            params=intrinsics,
        )

    def get_points(self, image: Image):

        point3d = to_homogen(
            np.stack(
                [self.points[p_id].xyz for p_id in image.point3D_ids if p_id != -1],
                axis=-1,
            )
        )

        return point3d

    def get_camera(self, image: Image) -> Camera:

        return self.cameras[image.camera_id]

    def load_image(self, image: Image):

        img_path = os.path.join(self.root_path, "images", image.name)

        with pil_image.open(img_path) as im:

            im = np.asarray(im).astype(np.float32) / 255.0

        im = np.transpose(im, [2, 0, 1])

        camera = self.get_camera(image)

        size = [camera.height, camera.width]

        im = (
            nn.functional.interpolate(
                torch.tensor(im).unsqueeze(0),
                size,
                align_corners=True,
                mode="bilinear",
                antialias=True,
            )
            .squeeze(0)
            .numpy()
        )

        return im

    def get_sample(self, image):

        im = self.load_image(image)
        P = image_to_projection_matrix(image)
        point3d = self.get_points(image)
        camera = self.get_camera(image)

        intrinsics = None

        if camera.model == "SIMPLE_RADIAL":
            f, cx, cy, k1 = camera.params
            # assuming that k1 is small enough to be ignored!
            intrinsics = np.array([f, f, cx, cy])

        elif camera.model == "PINHOLE":
            intrinsics = camera.params
        else:
            raise Exception("camera model not recognized")

        return im, P, point3d, intrinsics

    def __len__(self):
        return len(self.img_keys)

    def __getitem__(self, idx):
        return self.get_sample(self.images[self.img_keys[idx]])

    def split(self, ratio=0.8, seed=42):

        keys = self.img_keys

        random.seed(seed)
        random.shuffle(keys)

        train_data = keys[: int((len(keys) + 1) * ratio)]
        test_data = keys[int((len(keys) + 1) * ratio) :]

        splits = {"train_split": train_data, "test_split": test_data}

        return SolutionSplit(self, train_data), SolutionSplit(self, test_data), splits


class SolutionSplit:

    def __init__(self, solution: ColmapSolution, keys) -> None:
        self.solution = solution
        self.keys = keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        return self.solution.get_sample(self.solution.images[self.keys[idx]])
