"""
Biomechanics pipeline — orchestrates CoM calculation and rigid bone enforcement
using the new definition types.

All anatomical knowledge comes from explicit arguments (SkeletonDefinition,
CoMDefinition), not from the Aspect. The Aspect is just a data container.
"""

import numpy as np
from numpy.typing import NDArray

from skellymodels.biomechanics.com_definition import CoMDefinition
from skellymodels.core.trajectory.trajectory import Trajectory
from skellymodels.core.trajectory.typed_trajectories import SpatialTrajectory
from skellymodels.skeleton.skeleton_definition import SkeletonDefinition


def calculate_center_of_mass(
    trajectory: SpatialTrajectory,
    skeleton: SkeletonDefinition,
    com_definition: CoMDefinition,
) -> tuple[SpatialTrajectory, SpatialTrajectory]:
    """
    Calculate segment-level and total-body center of mass.

    Args:
        trajectory: spatial trajectory with skeleton keypoint names
        skeleton: skeleton definition (provides segment_connections)
        com_definition: anthropometric mass distribution parameters

    Returns:
        total_body_com: SpatialTrajectory with 1 keypoint ("total_body_center_of_mass")
        segment_com: SpatialTrajectory with 1 keypoint per CoM segment
    """
    keypoint_data = trajectory.as_dict
    segment_connections = skeleton.segment_connections
    num_frames = trajectory.num_frames

    segment_com_positions: dict[str, NDArray[np.float64]] = {}
    segment_names_ordered: list[str] = []

    for seg_name, seg_params in com_definition.segments.items():
        rb_name = seg_params.rigid_body

        if rb_name not in segment_connections:
            raise ValueError(
                f"CoM segment '{seg_name}' references rigid body '{rb_name}' "
                f"which has no segment_connections entry (is it fully constrained? "
                f"CoM currently requires 2-keypoint segments)."
            )

        conn = segment_connections[rb_name]
        proximal_name = conn["proximal"]
        distal_name = conn["distal"]

        if proximal_name not in keypoint_data:
            raise KeyError(
                f"Proximal keypoint '{proximal_name}' for segment '{seg_name}' "
                f"not found in trajectory keypoints"
            )
        if distal_name not in keypoint_data:
            raise KeyError(
                f"Distal keypoint '{distal_name}' for segment '{seg_name}' "
                f"not found in trajectory keypoints"
            )

        proximal = keypoint_data[proximal_name]  # (F, 3)
        distal = keypoint_data[distal_name]  # (F, 3)

        segment_com = proximal + (distal - proximal) * seg_params.com_length_ratio
        segment_com_positions[seg_name] = segment_com
        segment_names_ordered.append(seg_name)

    # Total body CoM: weighted sum of segment CoMs
    total_body = np.zeros((num_frames, 3), dtype=np.float64)
    for seg_name, seg_params in com_definition.segments.items():
        total_body += segment_com_positions[seg_name] * seg_params.mass_fraction

    # Build output trajectories
    total_body_trajectory = SpatialTrajectory(
        name="total_body_center_of_mass",
        keypoint_names=("total_body_center_of_mass",),
        array=total_body[:, np.newaxis, :],  # (F, 1, 3)
    )

    segment_com_array = np.stack(
        [segment_com_positions[name] for name in segment_names_ordered],
        axis=1,
    )  # (F, num_segments, 3)

    segment_com_trajectory = SpatialTrajectory(
        name="segment_center_of_mass",
        keypoint_names=tuple(segment_names_ordered),
        array=segment_com_array,
    )

    return total_body_trajectory, segment_com_trajectory


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
