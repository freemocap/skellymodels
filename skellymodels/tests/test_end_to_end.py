"""Integration tests: human body skeleton + mediapipe mapping working together."""

from pathlib import Path

import numpy as np
import pytest

from skellymodels.skeleton.skeleton_loader import load_skeleton_from_yaml
from skellymodels.mapping.mapping_loader import load_mapping_from_yaml
from skellymodels.biomechanics.com_loader import load_com_from_yaml
from skellymodels.core.trajectory.typed_trajectories import SpatialTrajectory

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
HUMAN_BODY_PATH = CONFIGS_DIR / "skeletons" / "human_body.yaml"
MEDIAPIPE_MAPPING_PATH = CONFIGS_DIR / "mappings" / "mediapipe_human_body.yaml"
DE_LEVA_PATH = CONFIGS_DIR / "center_of_mass" / "human_body_de_leva.yaml"

MEDIAPIPE_TRACKER_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


class TestHumanBodySkeleton:
    def test_loads_successfully(self) -> None:
        skeleton = load_skeleton_from_yaml(HUMAN_BODY_PATH)
        assert skeleton.name == "human_body"

    def test_rigid_body_count(self) -> None:
        skeleton = load_skeleton_from_yaml(HUMAN_BODY_PATH)
        assert len(skeleton.rigid_bodies) == 19

    def test_linkage_count(self) -> None:
        skeleton = load_skeleton_from_yaml(HUMAN_BODY_PATH)
        assert len(skeleton.linkages) == 16

    def test_chain_count(self) -> None:
        skeleton = load_skeleton_from_yaml(HUMAN_BODY_PATH)
        assert len(skeleton.chains) == 5

    def test_fully_constrained_bodies(self) -> None:
        skeleton = load_skeleton_from_yaml(HUMAN_BODY_PATH)
        fc = [name for name, rb in skeleton.rigid_bodies.items() if rb.is_fully_constrained]
        assert set(fc) == {
            "skull",
            "right_hand", "left_hand",
        }

    def test_under_constrained_bodies(self) -> None:
        skeleton = load_skeleton_from_yaml(HUMAN_BODY_PATH)
        uc = [name for name, rb in skeleton.rigid_bodies.items() if not rb.is_fully_constrained]
        assert len(uc) == 16  # 19 total - 3 fully constrained

    def test_all_keypoint_names_no_duplicates(self) -> None:
        skeleton = load_skeleton_from_yaml(HUMAN_BODY_PATH)
        kps = skeleton.all_keypoint_names
        assert len(kps) == len(set(kps))

    def test_segment_connections(self) -> None:
        skeleton = load_skeleton_from_yaml(HUMAN_BODY_PATH)
        sc = skeleton.segment_connections
        assert sc["right_upper_arm"] == {
            "proximal": "right_shoulder",
            "distal": "right_elbow",
        }
        assert sc["right_thigh"] == {
            "proximal": "pelvis_right_hip_acetabulum",
            "distal": "right_knee",
        }
        # Fully-constrained bodies should not appear
        assert "skull" not in sc
        assert "pelvis" not in sc

    def test_axial_chain_joint_sequence(self) -> None:
        skeleton = load_skeleton_from_yaml(HUMAN_BODY_PATH)
        seq = skeleton.get_chain_joint_sequence("axial")
        assert seq == [
            "pelvis_spine_sacrum_origin",
            "pelvis_spine_sacrum_origin",  # sacrum_lumbar shared
            "spine_lumbar_l1",
            "spine_thoracic_top_t1",
            "skull_origin_foramen_magnum",
        ]

    def test_right_arm_chain_joint_sequence(self) -> None:
        skeleton = load_skeleton_from_yaml(HUMAN_BODY_PATH)
        seq = skeleton.get_chain_joint_sequence("right_arm")
        assert seq == [
            "spine_thoracic_top_t1",  # right_clavicle origin
            "right_shoulder",
            "right_elbow",
            "right_wrist",
        ]

    def test_junction_keypoints(self) -> None:
        skeleton = load_skeleton_from_yaml(HUMAN_BODY_PATH)
        junctions = skeleton.junction_keypoints
        # T1 appears in axial + right_arm + left_arm chains
        assert "spine_thoracic_top_t1" in junctions


class TestMappingSkeletonCompatibility:
    """Verify the mediapipe mapping produces keypoints the skeleton needs."""

    def test_mapping_covers_skeleton_keypoints(self) -> None:
        skeleton = load_skeleton_from_yaml(HUMAN_BODY_PATH)
        mapping = load_mapping_from_yaml(MEDIAPIPE_MAPPING_PATH)

        skeleton_kps = set(skeleton.all_keypoint_names)
        mapped_kps = set(mapping.skeleton_keypoint_names)

        missing = skeleton_kps - mapped_kps
        assert missing == set(), (
            f"Skeleton requires keypoints not produced by mapping: {sorted(missing)}"
        )

    def test_end_to_end_tracker_to_trajectory(self) -> None:
        """Full pipeline: tracker data → mapping → SpatialTrajectory."""
        mapping = load_mapping_from_yaml(MEDIAPIPE_MAPPING_PATH)

        rng = np.random.default_rng(seed=42)
        tracker_data = rng.standard_normal((100, 33, 3))

        mapped_array = mapping.apply(
            tracked_array=tracker_data,
            tracked_point_names=MEDIAPIPE_TRACKER_NAMES,
        )

        trajectory = SpatialTrajectory(
            name="3d_xyz",
            keypoint_names=tuple(mapping.skeleton_keypoint_names),
            array=mapped_array,
        )

        assert trajectory.num_frames == 100
        assert trajectory.num_keypoints == len(mapping.skeleton_keypoint_names)
        assert "skull_origin_foramen_magnum" in trajectory
        assert "right_shoulder" in trajectory

        # Verify a direct mapping preserved data exactly
        nose_idx_tracker = MEDIAPIPE_TRACKER_NAMES.index("nose")
        np.testing.assert_array_equal(
            trajectory["nose_tip"],
            tracker_data[:, nose_idx_tracker, :],
        )


class TestCoMSkeletonCompatibility:
    """Verify the CoM definition references valid skeleton rigid bodies."""

    def test_all_com_rigid_bodies_exist_in_skeleton(self) -> None:
        skeleton = load_skeleton_from_yaml(HUMAN_BODY_PATH)
        com = load_com_from_yaml(DE_LEVA_PATH)

        rb_names = set(skeleton.rigid_bodies.keys())
        for seg_name, seg in com.segments.items():
            assert seg.rigid_body in rb_names, (
                f"CoM segment '{seg_name}' references rigid body "
                f"'{seg.rigid_body}' not in skeleton"
            )
