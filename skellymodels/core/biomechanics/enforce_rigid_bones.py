"""
Biomechanics pipeline — orchestrates CoM calculation and rigid bone enforcement
using the new definition types.

All anatomical knowledge comes from explicit arguments (SkeletonDefinition,
CoMDefinition), not from the Aspect. The Aspect is just a data container.
"""

import numpy as np
from numpy.typing import NDArray

from skellymodels.core.trajectory.typed_trajectories import SpatialTrajectory
from skellymodels.core.skeleton.skeleton_definition import SkeletonDefinition


def enforce_rigid_bones(
    trajectory: SpatialTrajectory,
    skeleton: SkeletonDefinition,
) -> SpatialTrajectory:
    """
    Enforce rigid bone length constraints using the skeleton's joint hierarchy.

    Uses median bone lengths as targets and adjusts child positions to
    maintain fixed distances from parents.

    Args:
        trajectory: spatial trajectory with skeleton keypoint names
        skeleton: skeleton definition (provides joint_hierarchy)

    Returns:
        Rigidified SpatialTrajectory with same keypoints
    """
    joint_hierarchy = skeleton.joint_hierarchy
    keypoint_data = trajectory.as_dict

    # Copy data so we can modify in place
    rigid_data: dict[str, NDArray[np.float64]] = {
        name: arr.copy() for name, arr in keypoint_data.items()
    }

    # Find the root keypoint (appears as parent but never as child)
    all_children: set[str] = set()
    for children in joint_hierarchy.values():
        all_children.update(children)
    roots = set(joint_hierarchy.keys()) - all_children

    if len(roots) == 0:
        raise ValueError("No root keypoint found in joint hierarchy (cycle detected?)")

    def _enforce_recursive(parent_name: str) -> None:
        if parent_name not in joint_hierarchy:
            return

        parent_pos = rigid_data[parent_name]  # (F, 3)

        for child_name in joint_hierarchy[parent_name]:
            if child_name not in rigid_data:
                continue

            child_pos = rigid_data[child_name]  # (F, 3)

            # Compute current bone vectors and lengths
            bone_vectors = child_pos - parent_pos  # (F, 3)
            bone_lengths = np.linalg.norm(bone_vectors, axis=1, keepdims=True)  # (F, 1)

            # Target length = median across all frames
            target_length = np.nanmedian(bone_lengths)

            if target_length < 1e-10:
                continue

            # Normalize bone direction, scale to target length
            safe_lengths = np.where(
                bone_lengths < 1e-10, 1.0, bone_lengths
            )
            bone_directions = bone_vectors / safe_lengths
            rigid_data[child_name] = parent_pos + bone_directions * target_length

            # Recurse into children
            _enforce_recursive(child_name)

    for root in roots:
        _enforce_recursive(root)

    # Reconstruct array in original keypoint order
    output_array = np.stack(
        [rigid_data[name] for name in trajectory.keypoint_names],
        axis=1,
    )

    return SpatialTrajectory(
        name="rigid_3d_xyz",
        keypoint_names=trajectory.keypoint_names,
        array=output_array,
    )
