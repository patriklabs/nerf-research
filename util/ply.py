import numpy as np


def write_ply(points, color, path):
    """
    Convert point cloud to a PLY file.

    Args:
        points (np.ndarray): Points.
        color (np.ndarray): Color.
        path (str): Path.

    Returns:
        None
    """

    header = "ply\nformat ascii 1.0"
    header += "\nelement vertex %d" % len(points)
    header += "\nproperty float32 x\nproperty float32 y\nproperty float32 z"
    header += "\nproperty uchar red\nproperty uchar green\nproperty uchar blue"
    header += "\nend_header\n"

    points = np.hstack([points, (255 * color).astype(np.uint8)])

    np.savetxt(path, points, header=header, comments="", fmt="%8.5g", delimiter=" ")
