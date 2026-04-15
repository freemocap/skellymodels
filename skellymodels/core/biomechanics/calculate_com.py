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
