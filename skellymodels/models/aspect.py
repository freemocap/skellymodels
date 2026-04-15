"""
Aspect model — a modular unit representing a tracked region (body, face, hand).

Aspect is a pure data container. It holds trajectories and metadata but has
no knowledge of anatomy, mappings, or skeleton structure. Those concerns
are handled externally:
  - KeypointMapping.apply() produces mapped data
  - SpatialTrajectory wraps the mapped data
  - Aspect stores the trajectory
"""

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from skellymodels.core.trajectory.trajectory import Trajectory


class ReprojectionError(BaseModel):
    """Per-keypoint per-frame reprojection error from the tracker."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        extra="forbid",
    )

    data: np.ndarray  # (num_frames, num_tracked_points)
    tracked_point_names: list[str]

    def __str__(self) -> str:
        return f"ReprojectionError(frames={self.data.shape[0]}, tracked_points={self.data.shape[1]})"


class Aspect(BaseModel):
    """
    A tracked anatomical region (e.g. body, face, hand).

    Pure data container: holds named trajectories, optional reprojection error,
    and arbitrary metadata. No anatomy knowledge.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
    )

    name: str
    trajectories: dict[str, Trajectory] = Field(default_factory=dict)
    reprojection_error: ReprojectionError | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_trajectory(self, name: str, trajectory: Trajectory) -> None:
        """Add a named trajectory to this aspect."""
        if not isinstance(trajectory, Trajectory):
            raise TypeError(
                f"Expected Trajectory instance for '{name}', "
                f"got {type(trajectory).__name__}"
            )
        self.trajectories[name] = trajectory

    def get_trajectory(self, name: str) -> Trajectory:
        """Get a trajectory by name. Raises KeyError if not found."""
        if name not in self.trajectories:
            raise KeyError(
                f"Trajectory '{name}' not found. "
                f"Available: {sorted(self.trajectories.keys())}"
            )
        return self.trajectories[name]

    def set_reprojection_error(
        self,
        data: np.ndarray,
        tracked_point_names: list[str],
    ) -> None:
        """Attach reprojection error data."""
        if data.ndim != 2:
            raise ValueError(
                f"Reprojection error data must be 2D (frames, tracked_points), "
                f"got ndim={data.ndim}"
            )
        if data.shape[1] != len(tracked_point_names):
            raise ValueError(
                f"Reprojection error has {data.shape[1]} columns but "
                f"{len(tracked_point_names)} tracked point names provided"
            )
        self.reprojection_error = ReprojectionError(
            data=data,
            tracked_point_names=tracked_point_names,
        )

    @property
    def trajectory_names(self) -> list[str]:
        return list(self.trajectories.keys())

    def __str__(self) -> str:
        traj_info = (
            f"{len(self.trajectories)} trajectories: {self.trajectory_names}"
            if self.trajectories else "No trajectories"
        )
        error_info = "Has reprojection error" if self.reprojection_error else "No reprojection error"
        return f"Aspect(name={self.name!r}, {traj_info}, {error_info})"

    def __repr__(self) -> str:
        return self.__str__()
