"""
Keypoint mapping from tracker namespace to skeleton namespace.

A KeypointMapping defines how raw tracked points (e.g. mediapipe landmarks)
are combined to produce skeleton keypoints (e.g. anatomical landmarks).

Three mapping forms (inferred from YAML shape, not declared):
  str           → 1:1 direct mapping
  list[str]     → equal-weight average of tracker points
  dict[str,float] → weighted sum of tracker points
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator

from skellymodels.type_aliases import (
    KeypointName,
    SkeletonName,
    TrackedPointName,
    TrackerName,
    Weight,
)


# The union type for a single mapping entry
MappingSource = TrackedPointName | list[TrackedPointName] | dict[TrackedPointName, Weight]


class KeypointMapping(BaseModel):
    """
    Mapping from tracker-namespace point names to skeleton-namespace
    keypoint positions, defined as weighted combinations of tracked points.

    Fields:
        tracker_name: name of the tracker (e.g. "mediapipe")
        skeleton_name: name of the target skeleton (e.g. "human_body")
        mappings: dict mapping skeleton keypoint name → source specification
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tracker_name: TrackerName
    skeleton_name: SkeletonName
    mappings: dict[KeypointName, MappingSource]

    @model_validator(mode="after")
    def _validate_mappings(self) -> "KeypointMapping":
        if len(self.mappings) == 0:
            raise ValueError("mappings must contain at least one entry")

        for skeleton_kp, source in self.mappings.items():
            if isinstance(source, str):
                # Direct 1:1 — no further validation needed
                pass
            elif isinstance(source, list):
                if len(source) == 0:
                    raise ValueError(
                        f"Mapping for '{skeleton_kp}': list source must not be empty"
                    )
            elif isinstance(source, dict):
                if len(source) == 0:
                    raise ValueError(
                        f"Mapping for '{skeleton_kp}': dict source must not be empty"
                    )
            else:
                raise ValueError(
                    f"Mapping for '{skeleton_kp}': source must be str, list[str], "
                    f"or dict[str, float], got {type(source).__name__}"
                )

        return self

    @property
    def skeleton_keypoint_names(self) -> list[str]:
        """Ordered list of skeleton keypoint names produced by this mapping."""
        return list(self.mappings.keys())

    @property
    def required_tracker_points(self) -> set[str]:
        """Set of all tracker point names referenced by this mapping."""
        result: set[str] = set()
        for source in self.mappings.values():
            if isinstance(source, str):
                result.add(source)
            elif isinstance(source, list):
                result.update(source)
            elif isinstance(source, dict):
                result.update(source.keys())
        return result

    def apply(
        self,
        tracked_array: NDArray[np.float64],
        tracked_point_names: list[str],
    ) -> NDArray[np.float64]:
        """
        Apply the mapping to produce skeleton keypoint positions.

        Args:
            tracked_array: (F, M_tracker, 3) array of tracker data
            tracked_point_names: ordered list of tracker point names
                matching axis 1 of tracked_array

        Returns:
            (F, M_skeleton, 3) array with skeleton keypoint ordering
            matching self.skeleton_keypoint_names

        Raises:
            ValueError: if tracked_array shape is wrong
            KeyError: if a required tracker point is not in tracked_point_names
        """
        if tracked_array.ndim != 3:
            raise ValueError(
                f"tracked_array must be 3-dimensional (F, M, 3), "
                f"got ndim={tracked_array.ndim}"
            )
        if tracked_array.shape[1] != len(tracked_point_names):
            raise ValueError(
                f"tracked_array axis 1 has {tracked_array.shape[1]} entries "
                f"but {len(tracked_point_names)} names provided"
            )
        if tracked_array.shape[2] != 3:
            raise ValueError(
                f"tracked_array axis 2 must be 3 (xyz), got {tracked_array.shape[2]}"
            )

        # Build name→index lookup
        name_to_idx: dict[str, int] = {
            name: i for i, name in enumerate(tracked_point_names)
        }

        # Check all required tracker points are present
        missing = self.required_tracker_points - set(tracked_point_names)
        if missing:
            raise KeyError(
                f"Tracker points required by mapping but not in input: "
                f"{sorted(missing)}"
            )

        num_frames = tracked_array.shape[0]
        num_skeleton_kps = len(self.mappings)
        output = np.empty(
            (num_frames, num_skeleton_kps, 3), dtype=np.float64
        )

        for out_idx, (skeleton_kp, source) in enumerate(self.mappings.items()):
            if isinstance(source, str):
                # Direct 1:1
                src_idx = name_to_idx[source]
                output[:, out_idx, :] = tracked_array[:, src_idx, :]

            elif isinstance(source, list):
                # Equal-weight average
                src_indices = [name_to_idx[name] for name in source]
                output[:, out_idx, :] = np.mean(
                    tracked_array[:, src_indices, :], axis=1
                )

            elif isinstance(source, dict):
                # Weighted sum
                acc = np.zeros((num_frames, 3), dtype=np.float64)
                for tracker_name, weight in source.items():
                    src_idx = name_to_idx[tracker_name]
                    acc += tracked_array[:, src_idx, :] * weight
                output[:, out_idx, :] = acc

        return output

    def apply_from_tracker(
        self,
        tracked_array: NDArray[np.float64],
        tracker_info: "TrackerModelInfo",
        aspect_name: str,
    ) -> NDArray[np.float64]:
        """
        Apply the mapping using a TrackerModelInfo for name resolution.

        This is the preferred way to apply mappings — the user never needs
        to manually construct a list of tracked point names.

        Args:
            tracked_array: (F, M_tracker, 3) array of tracker data for one aspect
            tracker_info: the tracker's model info (provides point names)
            aspect_name: which aspect to get point names from (e.g. "body")

        Returns:
            (F, M_skeleton, 3) array with skeleton keypoint ordering
        """
        from skellymodels.models.tracking_model_info import TrackerModelInfo

        if aspect_name not in tracker_info.aspects:
            raise KeyError(
                f"Aspect '{aspect_name}' not found in tracker info. "
                f"Available: {sorted(tracker_info.aspects.keys())}"
            )

        tracked_point_names = tracker_info.aspects[aspect_name].tracked_point_names
        return self.apply(
            tracked_array=tracked_array,
            tracked_point_names=tracked_point_names,
        )
