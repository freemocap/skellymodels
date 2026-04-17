from dataclasses import dataclass

import numpy as np

from skellymodels.type_aliases import TrackedPointName, KeypointMappingType, TrackedPointList, \
    WeightedTrackedPoints


@dataclass
class KeypointMapping:

    tracked_points: list[TrackedPointName]
    weights: list[float]

    @classmethod
    def create(cls, mapping: KeypointMappingType):

        if isinstance(mapping, TrackedPointName):
            tracked_points = [mapping]
            weights = [1]
        elif isinstance(mapping, TrackedPointList):
            tracked_points = mapping
            weights = [1 / len(mapping)] * len(mapping)

        elif isinstance(mapping, WeightedTrackedPoints):
            tracked_points = list(mapping.keys())
            weights = list(mapping.values())
        else:
            raise ValueError("Mapping must be a TrackedPointName, TrackedPointList, or WeightedTrackedPoints")


        if len(tracked_points) != len(weights):
            raise ValueError("The number of tracked points must match the number of weights")

        return cls(tracked_points=tracked_points, weights=weights)

    def calculate_trajectory(self, data_fr_name_xyz: np.ndarray, names: list[TrackedPointName]) -> np.ndarray:
        """
        Calculate a trajectory from a mapping of tracked points and their weights.
        """

        if data_fr_name_xyz.shape[1] != len(names):
            raise ValueError("Data shape does not match trajectory names length")
        if not all(tracked_point_name in names for tracked_point_name in self.tracked_points):
            raise ValueError("Not all tracked points in mapping found in trajectory names")

        number_of_frames = data_fr_name_xyz.shape[0]
        number_of_dimensions = data_fr_name_xyz.shape[2]
        trajectories_frame_xyz = np.zeros((number_of_frames, number_of_dimensions), dtype=np.float32)

        for tracked_point_name, weight in zip(self.tracked_points, self.weights):
            if tracked_point_name not in names:
                raise ValueError(f"Key {tracked_point_name} not found in trajectory names")

            keypoint_index = names.index(tracked_point_name)
            keypoint_fr_xyz = data_fr_name_xyz[:, keypoint_index, :]  # slice out the relevant tracked point
            trajectories_frame_xyz += keypoint_fr_xyz * weight  # scale the tracked point by the weight and add to the trajectory

        if np.sum(np.isnan(trajectories_frame_xyz)) == trajectories_frame_xyz.size:
            raise ValueError(f"Trajectory calculation resulted in all NaNs")

        return trajectories_frame_xyz
