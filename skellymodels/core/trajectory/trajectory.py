"""
Base Trajectory model.

A Trajectory is a named, validated tensor of shape (F, M, D) where:
  F = number of frames (>= 1)
  M = number of named keypoints (>= 1)
  D = dimensionality per keypoint (3 for spatial, 4 for quaternion, etc.)

Each keypoint has a unique string name. The ordering of names corresponds
to the second axis of the array.
"""

from functools import cached_property

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator

from skellymodels.core.dot_access import DotAccessDict


class Trajectory(BaseModel):
    """
    Immutable container for multi-keypoint trajectory data backed by a numpy array.

    The array has shape (num_frames, num_keypoints, num_dimensions). Keypoints are
    accessed by name via __getitem__. Bulk access is available through as_array,
    as_dict, and as_dataframe.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    name: str
    keypoint_names: tuple[str, ...]
    array: NDArray[np.float64]

    @model_validator(mode="after")
    def _validate_shape_and_names(self) -> "Trajectory":
        if self.array.ndim != 3:
            raise ValueError(
                f"array must be 3-dimensional (frames, keypoints, dimensions), "
                f"got ndim={self.array.ndim}"
            )

        num_frames, num_keypoints, _num_dims = self.array.shape

        if num_frames < 1:
            raise ValueError(f"array must have at least 1 frame, got {num_frames}")

        if num_keypoints < 1:
            raise ValueError(f"array must have at least 1 keypoint, got {num_keypoints}")

        if num_keypoints != len(self.keypoint_names):
            raise ValueError(
                f"array has {num_keypoints} keypoints along axis 1 but "
                f"{len(self.keypoint_names)} keypoint names were provided"
            )

        if len(self.keypoint_names) != len(set(self.keypoint_names)):
            seen: set[str] = set()
            duplicates = [n for n in self.keypoint_names if n in seen or seen.add(n)]  # type: ignore[func-returns-value]
            raise ValueError(f"keypoint_names contains duplicates: {duplicates}")

        return self

    @cached_property
    def _name_to_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.keypoint_names)}

    def __getitem__(self, keypoint_name: str) -> NDArray[np.float64]:
        """Get (F, D) trajectory for a single keypoint by name."""
        if keypoint_name not in self._name_to_index:
            raise KeyError(
                f"Keypoint '{keypoint_name}' not found. "
                f"Available: {sorted(self.keypoint_names)}"
            )
        idx = self._name_to_index[keypoint_name]
        return self.array[:, idx, :]

    def __contains__(self, keypoint_name: str) -> bool:
        return keypoint_name in self._name_to_index

    def __len__(self) -> int:
        """Number of keypoints."""
        return len(self.keypoint_names)

    @property
    def num_frames(self) -> int:
        return self.array.shape[0]

    @property
    def num_keypoints(self) -> int:
        return self.array.shape[1]

    @property
    def num_dimensions(self) -> int:
        return self.array.shape[2]

    @property
    def as_array(self) -> NDArray[np.float64]:
        """Raw (F, M, D) numpy array."""
        return self.array

    @property
    def as_dict(self) -> dict[str, NDArray[np.float64]]:
        """Dict mapping keypoint name → (F, D) array."""
        return {
            name: self.array[:, i, :]
            for i, name in enumerate(self.keypoint_names)
        }

    @cached_property
    def keypoints(self) -> DotAccessDict:
        """
        Dot-access namespace for keypoint data.

        Usage:
            trajectory.keypoints.right_shoulder  # → (F, D) array
            trajectory.keypoints.nose_tip        # → (F, D) array

        Supports tab-completion in IDEs.
        """
        return DotAccessDict(self.as_dict)

    @property
    def as_dataframe(self) -> pd.DataFrame:
        """
        Long-form DataFrame with columns [frame, keypoint, d0, d1, ..., d_{D-1}].

        Subclasses override this to provide named dimension columns (e.g. x, y, z).
        """
        num_frames = self.num_frames
        num_keypoints = self.num_keypoints
        num_dims = self.num_dimensions

        flat = self.array.reshape(num_frames * num_keypoints, num_dims)
        dim_columns = [f"d{i}" for i in range(num_dims)]

        df = pd.DataFrame(flat, columns=dim_columns)
        df.insert(0, "frame", np.repeat(np.arange(num_frames), num_keypoints))
        df.insert(1, "keypoint", np.tile(list(self.keypoint_names), num_frames))
        return df

    def __str__(self) -> str:
        return (
            f"Trajectory(name={self.name!r}, "
            f"frames={self.num_frames}, "
            f"keypoints={self.num_keypoints}, "
            f"dims={self.num_dimensions})"
        )

    def __repr__(self) -> str:
        return self.__str__()
