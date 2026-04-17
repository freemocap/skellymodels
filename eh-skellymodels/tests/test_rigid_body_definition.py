"""Tests for rigid body definition types."""

import pytest

from skellymodels.core.rigid_body.rigid_body_definition import (
    AxisDefinition,
    AxisType,
    CoordinateFrameDefinition,
    RigidBodyDefinition,
)


# ============================================================
# AxisDefinition
# ============================================================


class TestAxisDefinition:
    def test_basic_construction(self) -> None:
        ad = AxisDefinition(keypoints=["nose_tip"], type=AxisType.EXACT)
        assert ad.keypoints == ["nose_tip"]
        assert ad.type == AxisType.EXACT

    def test_rejects_empty_keypoints(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            AxisDefinition(keypoints=[], type=AxisType.EXACT)

    def test_multiple_keypoints(self) -> None:
        ad = AxisDefinition(keypoints=["left_ear", "right_ear"], type=AxisType.APPROXIMATE)
        assert len(ad.keypoints) == 2


# ============================================================
# CoordinateFrameDefinition
# ============================================================


class TestCoordinateFrameDefinition:
    def test_two_axes_fully_constrained(self) -> None:
        cfd = CoordinateFrameDefinition(
            origin_keypoints=["skull_origin"],
            x_axis=AxisDefinition(keypoints=["nose_tip"], type=AxisType.EXACT),
            y_axis=AxisDefinition(keypoints=["left_eye"], type=AxisType.APPROXIMATE),
        )
        assert cfd.num_defined_axes == 2
        assert cfd.computed_axis_names == ["z_axis"]

    def test_one_axis_under_constrained(self) -> None:
        cfd = CoordinateFrameDefinition(
            origin_keypoints=["shoulder"],
            x_axis=AxisDefinition(keypoints=["elbow"], type=AxisType.EXACT),
        )
        assert cfd.num_defined_axes == 1
        assert sorted(cfd.computed_axis_names) == ["y_axis", "z_axis"]
        assert cfd.approximate_axis is None

    def test_exact_and_approximate_accessors(self) -> None:
        cfd = CoordinateFrameDefinition(
            origin_keypoints=["origin"],
            y_axis=AxisDefinition(keypoints=["a"], type=AxisType.APPROXIMATE),
            z_axis=AxisDefinition(keypoints=["b"], type=AxisType.EXACT),
        )
        exact_name, exact_def = cfd.exact_axis
        assert exact_name == "z_axis"
        approx_result = cfd.approximate_axis
        assert approx_result is not None
        approx_name, _ = approx_result
        assert approx_name == "y_axis"

    def test_rejects_zero_axes(self) -> None:
        with pytest.raises(ValueError, match="At least 1"):
            CoordinateFrameDefinition(origin_keypoints=["origin"])

    def test_rejects_three_axes(self) -> None:
        with pytest.raises(ValueError, match="At most 2"):
            CoordinateFrameDefinition(
                origin_keypoints=["origin"],
                x_axis=AxisDefinition(keypoints=["a"], type=AxisType.EXACT),
                y_axis=AxisDefinition(keypoints=["b"], type=AxisType.APPROXIMATE),
                z_axis=AxisDefinition(keypoints=["c"], type=AxisType.EXACT),
            )

    def test_rejects_single_approximate_axis(self) -> None:
        """A single defined axis must be EXACT (the observable primary axis)."""
        with pytest.raises(ValueError, match="EXACT"):
            CoordinateFrameDefinition(
                origin_keypoints=["origin"],
                x_axis=AxisDefinition(keypoints=["a"], type=AxisType.APPROXIMATE),
            )

    def test_rejects_two_exact(self) -> None:
        with pytest.raises(ValueError, match="approximate"):
            CoordinateFrameDefinition(
                origin_keypoints=["origin"],
                x_axis=AxisDefinition(keypoints=["a"], type=AxisType.EXACT),
                y_axis=AxisDefinition(keypoints=["b"], type=AxisType.EXACT),
            )

    def test_rejects_two_approximate(self) -> None:
        with pytest.raises(ValueError, match="exact"):
            CoordinateFrameDefinition(
                origin_keypoints=["origin"],
                x_axis=AxisDefinition(keypoints=["a"], type=AxisType.APPROXIMATE),
                z_axis=AxisDefinition(keypoints=["b"], type=AxisType.APPROXIMATE),
            )

    def test_all_referenced_keypoints(self) -> None:
        cfd = CoordinateFrameDefinition(
            origin_keypoints=["left_ear", "right_ear"],
            x_axis=AxisDefinition(keypoints=["nose_tip"], type=AxisType.EXACT),
            y_axis=AxisDefinition(keypoints=["left_eye"], type=AxisType.APPROXIMATE),
        )
        refs = cfd.all_referenced_keypoints()
        assert refs == {"left_ear", "right_ear", "nose_tip", "left_eye"}

    def test_all_axis_pair_combinations(self) -> None:
        """All three valid axis pair combinations work."""
        for pair in [("x_axis", "y_axis"), ("x_axis", "z_axis"), ("y_axis", "z_axis")]:
            kwargs = {
                "origin_keypoints": ["o"],
                pair[0]: AxisDefinition(keypoints=["a"], type=AxisType.EXACT),
                pair[1]: AxisDefinition(keypoints=["b"], type=AxisType.APPROXIMATE),
            }
            cfd = CoordinateFrameDefinition(**kwargs)
            expected_computed = ({"x_axis", "y_axis", "z_axis"} - set(pair)).pop()
            assert cfd.computed_axis_names == [expected_computed]


# ============================================================
# RigidBodyDefinition — Under-constrained (2 keypoints)
# ============================================================


class TestUnderconstrainedRigidBody:
    def test_basic_construction_with_auto_frame(self) -> None:
        """2-keypoint body with an explicit 1-axis frame."""
        rb = RigidBodyDefinition(
            name="right_upper_arm",
            keypoints=["right_shoulder", "right_elbow"],
            origin="right_shoulder",
            coordinate_frame=CoordinateFrameDefinition(
                origin_keypoints=["right_shoulder"],
                x_axis=AxisDefinition(keypoints=["right_elbow"], type=AxisType.EXACT),
            ),
        )
        assert rb.is_fully_constrained is False
        assert rb.coordinate_frame.num_defined_axes == 1

    def test_default_distal_is_exact_axis_keypoint(self) -> None:
        rb = RigidBodyDefinition(
            name="upper_arm",
            keypoints=["shoulder", "elbow"],
            origin="shoulder",
            coordinate_frame=CoordinateFrameDefinition(
                origin_keypoints=["shoulder"],
                x_axis=AxisDefinition(keypoints=["elbow"], type=AxisType.EXACT),
            ),
        )
        assert rb.default_distal_keypoint == "elbow"


# ============================================================
# RigidBodyDefinition — Fully constrained (3+ keypoints)
# ============================================================


def _make_skull_frame() -> CoordinateFrameDefinition:
    return CoordinateFrameDefinition(
        origin_keypoints=["skull_origin"],
        x_axis=AxisDefinition(keypoints=["nose_tip"], type=AxisType.EXACT),
        y_axis=AxisDefinition(keypoints=["left_eye"], type=AxisType.APPROXIMATE),
    )


class TestFullyConstrainedRigidBody:
    def test_basic_construction(self) -> None:
        rb = RigidBodyDefinition(
            name="skull",
            keypoints=["skull_origin", "nose_tip", "left_eye"],
            origin="skull_origin",
            coordinate_frame=_make_skull_frame(),
        )
        assert rb.is_fully_constrained is True

    def test_default_distal_is_exact_axis_keypoint(self) -> None:
        rb = RigidBodyDefinition(
            name="skull",
            keypoints=["skull_origin", "nose_tip", "left_eye"],
            origin="skull_origin",
            coordinate_frame=_make_skull_frame(),
        )
        assert rb.default_distal_keypoint == "nose_tip"

    def test_3_keypoints_with_1_axis_is_under_constrained(self) -> None:
        """3+ keypoints with only 1 axis → under-constrained (extra tracking keypoints)."""
        rb = RigidBodyDefinition(
            name="pelvis",
            keypoints=["sacrum", "left_hip", "right_hip"],
            origin="sacrum",
            coordinate_frame=CoordinateFrameDefinition(
                origin_keypoints=["sacrum"],
                y_axis=AxisDefinition(keypoints=["left_hip"], type=AxisType.EXACT),
            ),
        )
        assert rb.is_fully_constrained is False
        assert len(rb.keypoints) == 3

    def test_rejects_frame_referencing_unknown_keypoint(self) -> None:
        frame = CoordinateFrameDefinition(
            origin_keypoints=["a"],
            x_axis=AxisDefinition(keypoints=["unknown_point"], type=AxisType.EXACT),
            y_axis=AxisDefinition(keypoints=["c"], type=AxisType.APPROXIMATE),
        )
        with pytest.raises(ValueError, match="not in this rigid body"):
            RigidBodyDefinition(
                name="bad",
                keypoints=["a", "b", "c"],
                origin="a",
                coordinate_frame=frame,
            )


# ============================================================
# RigidBodyDefinition — General validation
# ============================================================


class TestRigidBodyValidation:
    def test_rejects_origin_not_in_keypoints(self) -> None:
        with pytest.raises(ValueError, match="origin"):
            RigidBodyDefinition(
                name="bad",
                keypoints=["a", "b"],
                origin="c",
                coordinate_frame=CoordinateFrameDefinition(
                    origin_keypoints=["a"],
                    x_axis=AxisDefinition(keypoints=["b"], type=AxisType.EXACT),
                ),
            )

    def test_rejects_duplicate_keypoints(self) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            RigidBodyDefinition(
                name="bad",
                keypoints=["a", "a"],
                origin="a",
                coordinate_frame=CoordinateFrameDefinition(
                    origin_keypoints=["a"],
                    x_axis=AxisDefinition(keypoints=["a"], type=AxisType.EXACT),
                ),
            )

    def test_rejects_single_keypoint(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            RigidBodyDefinition(
                name="bad",
                keypoints=["only_one"],
                origin="only_one",
                coordinate_frame=CoordinateFrameDefinition(
                    origin_keypoints=["only_one"],
                    x_axis=AxisDefinition(keypoints=["only_one"], type=AxisType.EXACT),
                ),
            )

    def test_frozen(self) -> None:
        rb = RigidBodyDefinition(
            name="test",
            keypoints=["a", "b"],
            origin="a",
            coordinate_frame=CoordinateFrameDefinition(
                origin_keypoints=["a"],
                x_axis=AxisDefinition(keypoints=["b"], type=AxisType.EXACT),
            ),
        )
        with pytest.raises(Exception):
            rb.name = "mutated"  # type: ignore[misc]
