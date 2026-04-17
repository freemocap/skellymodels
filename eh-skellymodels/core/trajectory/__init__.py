from skellymodels.core.trajectory.trajectory import Trajectory
from skellymodels.core.trajectory.typed_trajectories import (
    AngularAccelerationTrajectory,
    AngularVelocityTrajectory,
    QuaternionTrajectory,
    SpatialTrajectory,
)

__all__ = [
    "Trajectory",
    "SpatialTrajectory",
    "QuaternionTrajectory",
    "AngularVelocityTrajectory",
    "AngularAccelerationTrajectory",
]
