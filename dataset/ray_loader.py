import numpy as np
from torch.utils.data import Dataset

from dataset.util import to_K_matrix, generate_grid


def calculate_rays(solution):

    total_rays = []

    total_tn = []
    total_tf = []
    total_color = []

    for im, P, point3d, intrinsics in solution:

        K = to_K_matrix(intrinsics)

        point3d = P @ point3d

        distance = np.linalg.norm(point3d[0:3], axis=0)

        tn = np.min(distance)
        tf = np.max(distance)

        T = np.eye(4, 4)
        T[0:3, 0:4] = P

        Tinv = np.linalg.inv(T)
        Kinv = np.linalg.inv(K)

        M = Tinv[0:3, 0:3] @ Kinv[0:3, 0:3]

        H, W = im.shape[1:]
        grid = generate_grid(H, W)
        grid = np.concatenate((grid, np.ones_like(grid[0:1])), axis=0)

        o = Tinv[0:3, 3]

        d = M @ grid.reshape(3, H * W)

        d = d[0:3] / np.linalg.norm(d[0:3], axis=0)

        o = np.repeat(o[:, None], H * W, axis=-1)

        rays = np.concatenate((o, d), axis=0)

        total_rays.append(rays.astype(np.float32))
        total_tn.append(np.repeat(np.expand_dims(np.array(tn), -1), H * W, axis=-1))
        total_tf.append(np.repeat(np.expand_dims(np.array(tf), -1), H * W, axis=-1))

        total_color.append(im.reshape(3, H * W))

    total_rays = np.concatenate(total_rays, axis=-1)
    total_tn = np.expand_dims(np.concatenate(total_tn, axis=-1), 0)
    total_tf = np.expand_dims(np.concatenate(total_tf, axis=-1), 0)
    total_color = np.concatenate(total_color, axis=-1)

    return np.concatenate((total_rays, total_tn, total_tf, total_color), axis=0).astype(
        np.float32
    )


class RayLoader(Dataset):
    def __init__(self, sfm_solution):

        self.rays = calculate_rays(sfm_solution)

    def __len__(self):
        return self.rays.shape[-1]

    def __getitem__(self, idx):

        ray = self.rays[:, idx]

        od, tn, tf, rgb = np.split(ray, [6, 7, 8])

        return {"rgb": rgb, "ray": od, "tn": tn, "tf": tf}
