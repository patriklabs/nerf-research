from torch.utils.data import Dataset
import numpy as np
from util.statistics import reject_outliers


def extract_image(sample):

    (im, P, point3d, intrinsics) = sample

    point3d = P @ point3d

    distance = np.linalg.norm(point3d[0:3], axis=0)

    distance = reject_outliers(distance)

    tn = np.min(distance) / 1.5
    tf = np.max(distance) * 1.5

    T = np.eye(4, 4)
    T[0:3, 0:4] = P

    data = {"image": im, "T": T, "intrinsics": intrinsics, "tn": tn, "tf": tf}

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
