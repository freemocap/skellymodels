from pathlib import Path

import numpy as np

if __name__ == "__main__":
    from skellymodels.core.skeleton import load_skeleton_from_yaml
    from skellymodels.core.mapping import load_mapping_from_yaml
    from skellymodels.core.models.tracking_model_info import load_tracker_info_from_yaml
    from skellymodels.core.trajectory import SpatialTrajectory

    tracker_data = np.load(Path().home() /"freemocap_data/recordings/freemocap_test_data/output_data/mediapipe_skeleton_3d.npy")
    # Load definitions
    current_file_path = Path(__file__).parent
    human_body_skeleton_path = current_file_path / "skellymodels/configs/skeletons/human_body.yaml"
    mapping_path = current_file_path / "skellymodels/configs/mappings/mediapipe_human_body.yaml"
    tracker_info_path = current_file_path / "configs/skeletons/mediapipe_human_body.yaml"

    assert human_body_skeleton_path.exists(), f"Skeleton definition file not found at {human_body_skeleton_path}"
    assert mapping_path.exists(), f"Mapping file not found at {mapping_path}"
    assert tracker_info_path.exists(), f"Tracker info file not found at {tracker_info_path}"

    skeleton = load_skeleton_from_yaml(human_body_skeleton_path)
    mapping = load_mapping_from_yaml(mapping_path)
    tracker_info = load_tracker_info_from_yaml(tracker_info_path)

    # Apply mapping — tracker_info handles the point names, no raw strings needed
    mapped_array = mapping.apply_from_tracker(
        tracked_array=tracker_data,   # (num_frames, 33, 3) from mediapipe
        tracker_info=tracker_info,
        aspect_name="body",
    )

    # Create a typed trajectory
    trajectory = SpatialTrajectory(
        name="3d_xyz",
        keypoint_names=tuple(mapping.skeleton_keypoint_names),
        array=mapped_array,
    )

    # Access data via dot notation — no raw strings
    right_shoulder = trajectory.keypoints.right_shoulder   # (num_frames, 3)
    skull_origin = trajectory.keypoints.skull_origin_foramen_magnum

    # Or by string indexing when you need it
    right_shoulder = trajectory["right_shoulder"]

    # Skeleton properties — all derived from the definition
    skull_def = skeleton.rb.skull             # RigidBodyDefinition via dot access
    hierarchy = skeleton.joint_hierarchy      # parent → children keypoint tree
    junctions = skeleton.junction_keypoints   # FABRIK reconciliation points

    # Biomechanics
    from skellymodels.core.biomechanics import load_com_from_yaml
    from skellymodels.core.biomechanics.calculate_com import calculate_center_of_mass

    com_def = load_com_from_yaml("skellymodels/configs/center_of_mass/human_body_de_leva.yaml")
    total_com, segment_com = calculate_center_of_mass(trajectory, skeleton, com_def)
