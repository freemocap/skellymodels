"""Tests for the refactored biomechanics pipeline."""

from pathlib import Path

import numpy as np
import pytest

from skellymodels.biomechanics.com_definition import CoMDefinition, SegmentCoMParameters
from skellymodels.biomechanics.pipeline import calculate_center_of_mass, enforce_rigid_bones
from skellymodels.core.trajectory.typed_trajectories import SpatialTrajectory
from skellymodels.skeleton.skeleton_loader import load_skeleton_from_yaml

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
TEST_SKELETON_PATH = CONFIGS_DIR / "skeletons" / "test_skeleton.yaml"


def _make_test_trajectory(num_frames: int = 50) -> SpatialTrajectory:
    """Build a synthetic trajectory matching the test skeleton's keypoints."""
    marker_names = (
        "pelvis_origin", "spine_base", "neck_base",
        "nose", "left_eye",
        "right_elbow", "right_wrist",
    )
    rng = np.random.default_rng(seed=42)
    data = rng.standard_normal((num_frames, len(marker_names), 3))
    return SpatialTrajectory(name="3d_xyz", keypoint_names=marker_names, array=data)


def _make_test_com() -> CoMDefinition:
    """CoM definition matching the test skeleton's under-constrained RBs."""
    return CoMDefinition(
        skeleton_name="test_skeleton",
        source="test",
        segments={
            "pelvis_seg": SegmentCoMParameters(
                rigid_body="pelvis", com_length_ratio=0.5, mass_fraction=0.3
            ),
            "spine_seg": SegmentCoMParameters(
                rigid_body="spine", com_length_ratio=0.5, mass_fraction=0.4
            ),
            "upper_arm_seg": SegmentCoMParameters(
                rigid_body="right_upper_arm", com_length_ratio=0.4, mass_fraction=0.15
            ),
            "forearm_seg": SegmentCoMParameters(
                rigid_body="right_forearm", com_length_ratio=0.4, mass_fraction=0.15
            ),
        },
    )


class TestCenterOfMassCalculation:
    def test_output_shapes(self) -> None:
        skeleton = load_skeleton_from_yaml(TEST_SKELETON_PATH)
        trajectory = _make_test_trajectory()
        com_def = _make_test_com()

        total_com, segment_com = calculate_center_of_mass(
            trajectory=trajectory,
            skeleton=skeleton,
            com_definition=com_def,
        )

        assert total_com.num_frames == 50
        assert total_com.num_keypoints == 1
        assert total_com.keypoint_names == ("total_body_center_of_mass",)

        assert segment_com.num_frames == 50
        assert segment_com.num_keypoints == 4

    def test_segment_com_is_between_proximal_and_distal(self) -> None:
        """Each segment CoM should lie on the line between proximal and distal."""
        skeleton = load_skeleton_from_yaml(TEST_SKELETON_PATH)

        # Use deterministic data: proximal at [0,0,0], distal at [10,0,0]
        marker_names = (
            "pelvis_origin", "spine_base", "neck_base",
            "nose", "left_eye", "right_elbow", "right_wrist",
        )
        data = np.zeros((1, 7, 3))
        data[0, 0, :] = [0, 0, 0]    # pelvis_origin (proximal of pelvis)
        data[0, 1, :] = [10, 0, 0]   # spine_base (distal of pelvis)
        data[0, 2, :] = [20, 0, 0]   # neck_base
        data[0, 5, :] = [30, 0, 0]   # right_elbow
        data[0, 6, :] = [40, 0, 0]   # right_wrist

        trajectory = SpatialTrajectory(name="test", keypoint_names=marker_names, array=data)
        com_def = _make_test_com()

        _, segment_com = calculate_center_of_mass(
            trajectory=trajectory, skeleton=skeleton, com_definition=com_def
        )

        # pelvis_seg: proximal=[0,0,0], distal=[10,0,0], ratio=0.5 → [5,0,0]
        pelvis_idx = list(com_def.segments.keys()).index("pelvis_seg")
        np.testing.assert_allclose(segment_com.array[0, pelvis_idx, :], [5, 0, 0])

    def test_total_com_is_weighted_average(self) -> None:
        """Total body CoM = sum of (segment_com * mass_fraction)."""
        skeleton = load_skeleton_from_yaml(TEST_SKELETON_PATH)
        trajectory = _make_test_trajectory(num_frames=10)
        com_def = _make_test_com()

        total_com, segment_com = calculate_center_of_mass(
            trajectory=trajectory, skeleton=skeleton, com_definition=com_def
        )

        # Manual weighted sum
        expected = np.zeros((10, 3))
        for i, (seg_name, seg_params) in enumerate(com_def.segments.items()):
            expected += segment_com.array[:, i, :] * seg_params.mass_fraction

        np.testing.assert_allclose(total_com.array[:, 0, :], expected, atol=1e-10)


class TestRigidBoneEnforcement:
    def test_output_shape_matches_input(self) -> None:
        skeleton = load_skeleton_from_yaml(TEST_SKELETON_PATH)
        trajectory = _make_test_trajectory()

        rigid = enforce_rigid_bones(trajectory=trajectory, skeleton=skeleton)

        assert rigid.num_frames == trajectory.num_frames
        assert rigid.num_keypoints == trajectory.num_keypoints
        assert rigid.keypoint_names == trajectory.keypoint_names

    def test_bone_lengths_are_constant(self) -> None:
        """After enforcement, each bone should have constant length across frames."""
        skeleton = load_skeleton_from_yaml(TEST_SKELETON_PATH)
        trajectory = _make_test_trajectory(num_frames=100)

        rigid = enforce_rigid_bones(trajectory=trajectory, skeleton=skeleton)
        rigid_data = rigid.as_dict

        # Check spine_base → neck_base bone length is constant
        if "spine_base" in rigid_data and "neck_base" in rigid_data:
            bone = rigid_data["neck_base"] - rigid_data["spine_base"]
            lengths = np.linalg.norm(bone, axis=1)
            np.testing.assert_allclose(
                lengths, lengths[0], atol=1e-10,
                err_msg="Bone length spine_base→neck_base should be constant"
            )

    def test_returns_spatial_trajectory(self) -> None:
        skeleton = load_skeleton_from_yaml(TEST_SKELETON_PATH)
        trajectory = _make_test_trajectory()

        rigid = enforce_rigid_bones(trajectory=trajectory, skeleton=skeleton)
        assert isinstance(rigid, SpatialTrajectory)
        assert rigid.name == "rigid_3d_xyz"
