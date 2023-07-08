

from torch.utils.data import Dataset
import numpy as np


def extract_image(sample):

    (im, P, point3d, intrinsics) = sample

    point3d = P@point3d

    depth = np.linalg.norm(point3d[0:3], axis=0)

    tn = np.min(depth)/2
    tf = np.max(depth)*2

    T = np.eye(4, 4)
    T[0:3, 0:4] = P

    data = {"image": im, "T": T,
            "intrinsics": intrinsics,
            "tn": tn, "tf": tf}

    for key, val in data.items():

        data[key] = val.astype(np.float32)

    return data


class ImageLoader(Dataset):
    def __init__(self, sfm_solution):
        self.solution = sfm_solution

    def __len__(self):
        return len(self.solution)

    def __getitem__(self, idx):
        return extract_image(self.solution[idx])
