"""Tests for skeleton definition types and YAML loading."""

from pathlib import Path

import pytest

from skellymodels.core.rigid_body.rigid_body_definition import (
    AxisDefinition,
    AxisType,
    CoordinateFrameDefinition,
    RigidBodyDefinition,
)
from skellymodels.skeleton.skeleton_definition import (
    ChainDefinition,
    LinkageDefinition,
    SkeletonDefinition,
)
from skellymodels.skeleton.skeleton_loader import (
    load_rigid_body_from_yaml,
    load_skeleton_from_yaml,
)

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
TEST_SKELETON_PATH = CONFIGS_DIR / "skeletons" / "test_skeleton.yaml"
CHARUCO_5X3_PATH = CONFIGS_DIR / "rigid_bodies" / "charuco_board_5x3.yaml"


def _auto_frame(origin: str, distal: str) -> CoordinateFrameDefinition:
    """Helper: auto-generated 1-axis frame for 2-keypoint body."""
    return CoordinateFrameDefinition(
        origin_keypoints=[origin],
        x_axis=AxisDefinition(keypoints=[distal], type=AxisType.EXACT),
    )


def _make_test_skeleton() -> SkeletonDefinition:
    """Build the test_skeleton programmatically (mirrors the YAML)."""
    skull_frame = CoordinateFrameDefinition(
        origin_keypoints=["neck_base"],
        x_axis=AxisDefinition(keypoints=["nose"], type=AxisType.EXACT),
        y_axis=AxisDefinition(keypoints=["left_eye"], type=AxisType.APPROXIMATE),
    )

    rigid_bodies = {
        "pelvis": RigidBodyDefinition(
            name="pelvis", keypoints=["pelvis_origin", "spine_base"], origin="pelvis_origin",
            coordinate_frame=_auto_frame("pelvis_origin", "spine_base"),
        ),
        "spine": RigidBodyDefinition(
            name="spine", keypoints=["spine_base", "neck_base"], origin="spine_base",
            coordinate_frame=_auto_frame("spine_base", "neck_base"),
        ),
        "skull": RigidBodyDefinition(
            name="skull",
            keypoints=["neck_base", "nose", "left_eye"],
            origin="neck_base",
            coordinate_frame=skull_frame,
        ),
        "right_upper_arm": RigidBodyDefinition(
            name="right_upper_arm",
            keypoints=["neck_base", "right_elbow"],
            origin="neck_base",
            coordinate_frame=_auto_frame("neck_base", "right_elbow"),
        ),
        "right_forearm": RigidBodyDefinition(
            name="right_forearm",
            keypoints=["right_elbow", "right_wrist"],
            origin="right_elbow",
            coordinate_frame=_auto_frame("right_elbow", "right_wrist"),
        ),
    }

    linkages = {
        "pelvis_spine": LinkageDefinition(
            name="pelvis_spine", parent_rigid_body="pelvis",
            child_rigid_bodies=["spine"], shared_keypoint="spine_base",
        ),
        "neck_branch": LinkageDefinition(
            name="neck_branch", parent_rigid_body="spine",
            child_rigid_bodies=["skull", "right_upper_arm"], shared_keypoint="neck_base",
        ),
        "right_elbow_joint": LinkageDefinition(
            name="right_elbow_joint", parent_rigid_body="right_upper_arm",
            child_rigid_bodies=["right_forearm"], shared_keypoint="right_elbow",
        ),
    }

    chains = {
        "axial": ChainDefinition(
            name="axial", root_rigid_body="pelvis",
            linkages=["pelvis_spine", "neck_branch"],
        ),
        "right_arm": ChainDefinition(
            name="right_arm", root_rigid_body="right_upper_arm",
            linkages=["right_elbow_joint"],
        ),
    }

    return SkeletonDefinition(
        name="test_skeleton", rigid_bodies=rigid_bodies,
        linkages=linkages, chains=chains,
    )


class TestLinkageDefinition:
    def test_basic(self) -> None:
        ld = LinkageDefinition(name="t", parent_rigid_body="a", child_rigid_bodies=["b"], shared_keypoint="kp")
        assert ld.is_branching is False

    def test_branching(self) -> None:
        ld = LinkageDefinition(name="t", parent_rigid_body="a", child_rigid_bodies=["b", "c"], shared_keypoint="kp")
        assert ld.is_branching is True

    def test_rejects_empty_children(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            LinkageDefinition(name="bad", parent_rigid_body="a", child_rigid_bodies=[], shared_keypoint="kp")


class TestChainDefinition:
    def test_rejects_empty_linkages(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            ChainDefinition(name="bad", root_rigid_body="root", linkages=[])


class TestSkeletonConstruction:
    def test_from_code(self) -> None:
        skeleton = _make_test_skeleton()
        assert skeleton.name == "test_skeleton"
        assert len(skeleton.rigid_bodies) == 5

    def test_from_yaml(self) -> None:
        skeleton = load_skeleton_from_yaml(TEST_SKELETON_PATH)
        assert skeleton.name == "test_skeleton"
        assert len(skeleton.rigid_bodies) == 5

    def test_yaml_matches_code(self) -> None:
        from_yaml = load_skeleton_from_yaml(TEST_SKELETON_PATH)
        from_code = _make_test_skeleton()
        assert set(from_yaml.rigid_bodies.keys()) == set(from_code.rigid_bodies.keys())
        assert set(from_yaml.linkages.keys()) == set(from_code.linkages.keys())
        assert set(from_yaml.chains.keys()) == set(from_code.chains.keys())

    def test_auto_generated_frames_for_2kp_bodies(self) -> None:
        """2-keypoint bodies loaded from YAML should have auto-generated 1-axis frames."""
        skeleton = load_skeleton_from_yaml(TEST_SKELETON_PATH)
        pelvis = skeleton.rigid_bodies["pelvis"]
        assert pelvis.coordinate_frame.num_defined_axes == 1
        assert pelvis.is_fully_constrained is False


class TestSkeletonValidation:
    def test_rejects_linkage_with_nonexistent_parent(self) -> None:
        skeleton = _make_test_skeleton()
        bad_linkages = dict(skeleton.linkages)
        bad_linkages["bad"] = LinkageDefinition(
            name="bad", parent_rigid_body="nonexistent",
            child_rigid_bodies=["spine"], shared_keypoint="spine_base",
        )
        with pytest.raises(ValueError, match="nonexistent"):
            SkeletonDefinition(name="bad", rigid_bodies=skeleton.rigid_bodies,
                             linkages=bad_linkages, chains={})

    def test_rejects_shared_keypoint_not_on_parent(self) -> None:
        skeleton = _make_test_skeleton()
        bad_linkages = dict(skeleton.linkages)
        bad_linkages["bad"] = LinkageDefinition(
            name="bad", parent_rigid_body="pelvis",
            child_rigid_bodies=["spine"], shared_keypoint="right_wrist",
        )
        with pytest.raises(ValueError, match="not found on parent"):
            SkeletonDefinition(name="bad", rigid_bodies=skeleton.rigid_bodies,
                             linkages=bad_linkages, chains={})

    def test_rejects_disconnected_chain(self) -> None:
        skeleton = _make_test_skeleton()
        bad_chains = {
            "bad": ChainDefinition(name="bad", root_rigid_body="pelvis",
                                   linkages=["pelvis_spine", "right_elbow_joint"]),
        }
        with pytest.raises(ValueError, match="disconnected"):
            SkeletonDefinition(name="bad", rigid_bodies=skeleton.rigid_bodies,
                             linkages=skeleton.linkages, chains=bad_chains)


class TestSkeletonDerivedProperties:
    def test_all_keypoint_names(self) -> None:
        skeleton = _make_test_skeleton()
        expected = {"pelvis_origin", "spine_base", "neck_base", "nose", "left_eye",
                    "right_elbow", "right_wrist"}
        assert set(skeleton.all_keypoint_names) == expected

    def test_segment_connections(self) -> None:
        skeleton = _make_test_skeleton()
        sc = skeleton.segment_connections
        assert "skull" not in sc  # fully constrained
        assert sc["pelvis"] == {"proximal": "pelvis_origin", "distal": "spine_base"}
        assert sc["right_forearm"] == {"proximal": "right_elbow", "distal": "right_wrist"}

    def test_joint_hierarchy(self) -> None:
        skeleton = _make_test_skeleton()
        jh = skeleton.joint_hierarchy
        assert "neck_base" in jh["spine_base"]
        assert set(jh["neck_base"]) == {"nose", "left_eye", "right_elbow"}
        assert jh["right_elbow"] == ["right_wrist"]

    def test_chain_joint_sequence(self) -> None:
        skeleton = _make_test_skeleton()
        assert skeleton.get_chain_joint_sequence("axial") == ["pelvis_origin", "spine_base", "neck_base"]
        assert skeleton.get_chain_joint_sequence("right_arm") == ["neck_base", "right_elbow"]

    def test_junction_keypoints(self) -> None:
        skeleton = _make_test_skeleton()
        assert "neck_base" in skeleton.junction_keypoints


class TestCharucoLoading:
    def test_load_charuco(self) -> None:
        rb = load_rigid_body_from_yaml(CHARUCO_5X3_PATH)
        assert rb.name == "charuco_board_5x3"
        assert rb.is_fully_constrained is True
        assert len(rb.keypoints) == 8
