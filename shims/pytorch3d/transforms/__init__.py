# Pure-PyTorch subset of pytorch3d.transforms (BSD License, see rotation_conversions.py header).
import math

import torch

from .rotation_conversions import (
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    euler_angles_to_matrix,
    matrix_to_axis_angle,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_apply,
    quaternion_invert,
    quaternion_multiply,
    quaternion_raw_multiply,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
    random_quaternions,
    random_rotation,
    random_rotations,
    rotation_6d_to_matrix,
    standardize_quaternion,
)
from .rotation_conversions import _axis_angle_rotation


class RotateAxisAngle:
    """Small replacement for pytorch3d.transforms.RotateAxisAngle (only transform_points is supported).

    Like PyTorch3D, points are treated as row vectors, so the stored matrix is the transpose of the
    column-vector rotation matrix and `transform_points(p) == p @ R_row`.
    """

    def __init__(self, angle, axis="X", degrees=True, dtype=torch.float32, device=None):
        angle = torch.as_tensor(angle, dtype=dtype, device=device)
        if degrees:
            angle = angle / 180.0 * math.pi
        self._R = _axis_angle_rotation(axis, angle).transpose(-1, -2)

    def get_matrix(self):
        m = torch.eye(4, dtype=self._R.dtype, device=self._R.device)
        m[:3, :3] = self._R
        return m[None]

    def transform_points(self, points):
        return points @ self._R.to(points)


__all__ = [k for k in dir() if not k.startswith("_")]
