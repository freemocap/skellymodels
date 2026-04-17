"""
Typed Trajectory subclasses with dimension-specific validation and methods.

Each subclass enforces a specific dimensionality and provides domain-specific
access patterns and operations.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import model_validator

from skellymodels.core.trajectory.trajectory import Trajectory


class SpatialTrajectory(Trajectory):
    """
    3D spatial trajectory: (F, M, 3) with dimensions representing x, y, z.
    """

    @model_validator(mode="after")
    def _validate_spatial_dimensions(self) -> "SpatialTrajectory":
        if self.num_dimensions != 3:
            raise ValueError(
                f"SpatialTrajectory requires 3 dimensions (x, y, z), "
                f"got {self.num_dimensions}"
            )
        return self

    @property
    def as_dataframe(self) -> pd.DataFrame:
        """Long-form DataFrame with columns [frame, keypoint, x, y, z]."""
        num_frames = self.num_frames
        num_keypoints = self.num_keypoints

        flat = self.array.reshape(num_frames * num_keypoints, 3)
        df = pd.DataFrame(flat, columns=["x", "y", "z"])
        df.insert(0, "frame", np.repeat(np.arange(num_frames), num_keypoints))
        df.insert(1, "keypoint", np.tile(list(self.keypoint_names), num_frames))
        return df


class QuaternionTrajectory(Trajectory):
    """
    Quaternion orientation trajectory: (F, M, 4) with wxyz ordering.

    Validates that all quaternions are unit-norm (within tolerance).
    """

    NORM_TOLERANCE: float = 1e-3

    @model_validator(mode="after")
    def _validate_quaternion_shape_and_norm(self) -> "QuaternionTrajectory":
        if self.num_dimensions != 4:
            raise ValueError(
                f"QuaternionTrajectory requires 4 dimensions (w, x, y, z), "
                f"got {self.num_dimensions}"
            )

        norms = np.linalg.norm(self.array, axis=2)
        bad_mask = np.abs(norms - 1.0) > self.NORM_TOLERANCE
        if np.any(bad_mask):
            bad_indices = np.argwhere(bad_mask)
            first_bad = bad_indices[0]
            bad_norm = norms[first_bad[0], first_bad[1]]
            raise ValueError(
                f"QuaternionTrajectory requires unit quaternions (norm ≈ 1.0). "
                f"First violation at frame={first_bad[0]}, keypoint={first_bad[1]}: "
                f"norm={bad_norm:.6f}"
            )
        return self

    @property
    def as_dataframe(self) -> pd.DataFrame:
        """Long-form DataFrame with columns [frame, keypoint, w, x, y, z]."""
        num_frames = self.num_frames
        num_keypoints = self.num_keypoints

        flat = self.array.reshape(num_frames * num_keypoints, 4)
        df = pd.DataFrame(flat, columns=["w", "x", "y", "z"])
        df.insert(0, "frame", np.repeat(np.arange(num_frames), num_keypoints))
        df.insert(1, "keypoint", np.tile(list(self.keypoint_names), num_frames))
        return df

    def to_rotation_matrices(self) -> NDArray[np.float64]:
        """
        Convert all quaternions to rotation matrices.

        Returns:
            (F, M, 3, 3) array of rotation matrices.
        """
        q = self.array
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

        r00 = 1 - 2 * (y * y + z * z)
        r01 = 2 * (x * y - w * z)
        r02 = 2 * (x * z + w * y)
        r10 = 2 * (x * y + w * z)
        r11 = 1 - 2 * (x * x + z * z)
        r12 = 2 * (y * z - w * x)
        r20 = 2 * (x * z - w * y)
        r21 = 2 * (y * z + w * x)
        r22 = 1 - 2 * (x * x + y * y)

        rotation_matrices = np.stack(
            [
                np.stack([r00, r01, r02], axis=-1),
                np.stack([r10, r11, r12], axis=-1),
                np.stack([r20, r21, r22], axis=-1),
            ],
            axis=-2,
        )
        return rotation_matrices


class AngularVelocityTrajectory(Trajectory):
    """
    Angular velocity trajectory: (F, M, 3) in rad/s.
    """

    @model_validator(mode="after")
    def _validate_angular_velocity_dimensions(self) -> "AngularVelocityTrajectory":
        if self.num_dimensions != 3:
            raise ValueError(
                f"AngularVelocityTrajectory requires 3 dimensions, "
                f"got {self.num_dimensions}"
            )
        return self

    def magnitude(self) -> NDArray[np.float64]:
        """Compute angular speed (norm) at each frame for each keypoint. Returns (F, M)."""
        return np.linalg.norm(self.array, axis=2)


class AngularAccelerationTrajectory(Trajectory):
    """
    Angular acceleration trajectory: (F, M, 3) in rad/s².
    """

    @model_validator(mode="after")
    def _validate_angular_acceleration_dimensions(self) -> "AngularAccelerationTrajectory":
        if self.num_dimensions != 3:
            raise ValueError(
                f"AngularAccelerationTrajectory requires 3 dimensions, "
                f"got {self.num_dimensions}"
            )
        return self

    def magnitude(self) -> NDArray[np.float64]:
        """Compute magnitude at each frame for each keypoint. Returns (F, M)."""
        return np.linalg.norm(self.array, axis=2)
