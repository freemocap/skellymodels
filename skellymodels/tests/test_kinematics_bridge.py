"""Tests for the kinematics bridge module."""

import numpy as np
import pytest

from skellymodels.core.rigid_body.rigid_body_definition import (
    AxisDefinition,
    AxisType,
    CoordinateFrameDefinition,
    RigidBodyDefinition,
)
from skellymodels.core.kinematics import (
    build_reference_geometry_data,
    compute_basis_from_definition,
    compute_swing_quaternion,
    compute_swing_quaternion_trajectory,
)


def _make_skull_rb() -> RigidBodyDefinition:
    """Fully-constrained skull: +X forward (nose), +Y left (left_eye), +Z up."""
    return RigidBodyDefinition(
        name="skull",
        keypoints=["origin", "nose", "left_eye"],
        origin="origin",
        coordinate_frame=CoordinateFrameDefinition(
            origin_keypoints=["origin"],
            x_axis=AxisDefinition(keypoints=["nose"], type=AxisType.EXACT),
            y_axis=AxisDefinition(keypoints=["left_eye"], type=AxisType.APPROXIMATE),
        ),
    )


def _make_upper_arm_rb() -> RigidBodyDefinition:
    """Under-constrained 2-keypoint body with 1-axis frame."""
    return RigidBodyDefinition(
        name="upper_arm",
        keypoints=["shoulder", "elbow"],
        origin="shoulder",
        coordinate_frame=CoordinateFrameDefinition(
            origin_keypoints=["shoulder"],
            x_axis=AxisDefinition(keypoints=["elbow"], type=AxisType.EXACT),
        ),
    )


class TestBuildReferenceGeometryData:
    def test_fully_constrained_produces_valid_dict(self) -> None:
        rb = _make_skull_rb()
        positions = {
            "origin": np.array([0.0, 0.0, 0.0]),
            "nose": np.array([1.0, 0.0, 0.0]),
            "left_eye": np.array([0.0, 1.0, 0.0]),
        }
        data = build_reference_geometry_data(rb, positions)
        assert data["units"] == "mm"
        assert len(data["keypoints"]) == 3

    def test_under_constrained_also_works(self) -> None:
        """Under-constrained bodies produce valid reference geometry data too."""
        rb = _make_upper_arm_rb()
        positions = {
            "shoulder": np.array([0.0, 0.0, 0.0]),
            "elbow": np.array([1.0, 0.0, 0.0]),
        }
        data = build_reference_geometry_data(rb, positions)
        assert len(data["keypoints"]) == 2

    def test_rejects_missing_keypoint(self) -> None:
        rb = _make_skull_rb()
        positions = {
            "origin": np.array([0.0, 0.0, 0.0]),
            "nose": np.array([1.0, 0.0, 0.0]),
        }
        with pytest.raises(KeyError, match="left_eye"):
            build_reference_geometry_data(rb, positions)


class TestComputeBasisFullyConstrained:
    def test_axis_aligned_body(self) -> None:
        rb = _make_skull_rb()
        positions = {
            "origin": np.array([0.0, 0.0, 0.0]),
            "nose": np.array([1.0, 0.0, 0.0]),      # +X forward
            "left_eye": np.array([0.0, 1.0, 0.0]),   # +Y left
        }
        basis, origin = compute_basis_from_definition(rb, positions)

        np.testing.assert_allclose(origin, [0.0, 0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(basis[0], [1.0, 0.0, 0.0], atol=1e-10)  # X
        np.testing.assert_allclose(basis[1], [0.0, 1.0, 0.0], atol=1e-10)  # Y
        np.testing.assert_allclose(basis[2], [0.0, 0.0, 1.0], atol=1e-10)  # Z = X×Y = up ✓

    def test_basis_is_orthonormal(self) -> None:
        rb = _make_skull_rb()
        positions = {
            "origin": np.array([1.0, 2.0, 3.0]),
            "nose": np.array([3.0, 2.5, 3.0]),
            "left_eye": np.array([1.5, 4.0, 3.2]),
        }
        basis, _ = compute_basis_from_definition(rb, positions)
        for i in range(3):
            np.testing.assert_allclose(np.linalg.norm(basis[i]), 1.0, atol=1e-10)
        np.testing.assert_allclose(np.dot(basis[0], basis[1]), 0.0, atol=1e-10)
        np.testing.assert_allclose(np.dot(basis[0], basis[2]), 0.0, atol=1e-10)
        np.testing.assert_allclose(np.dot(basis[1], basis[2]), 0.0, atol=1e-10)

    def test_right_handed(self) -> None:
        rb = _make_skull_rb()
        positions = {
            "origin": np.array([0.0, 0.0, 0.0]),
            "nose": np.array([2.0, 0.3, 0.1]),
            "left_eye": np.array([0.1, 3.0, 0.2]),
        }
        basis, _ = compute_basis_from_definition(rb, positions)
        computed_z = np.cross(basis[0], basis[1])
        np.testing.assert_allclose(computed_z, basis[2], atol=1e-10)


class TestComputeBasisUnderConstrained:
    def test_returns_valid_basis(self) -> None:
        """Under-constrained bodies return a valid orthonormal basis."""
        rb = _make_upper_arm_rb()
        positions = {
            "shoulder": np.array([0.0, 0.0, 0.0]),
            "elbow": np.array([1.0, 0.0, 0.0]),
        }
        basis, origin = compute_basis_from_definition(rb, positions)

        # Should be orthonormal
        for i in range(3):
            np.testing.assert_allclose(np.linalg.norm(basis[i]), 1.0, atol=1e-10)
        np.testing.assert_allclose(np.dot(basis[0], basis[1]), 0.0, atol=1e-10)
        np.testing.assert_allclose(np.dot(basis[0], basis[2]), 0.0, atol=1e-10)

        # Primary axis (x_axis) should point from shoulder toward elbow
        np.testing.assert_allclose(basis[0], [1.0, 0.0, 0.0], atol=1e-10)

    def test_right_handed(self) -> None:
        rb = _make_upper_arm_rb()
        positions = {
            "shoulder": np.array([0.0, 0.0, 0.0]),
            "elbow": np.array([0.5, 0.5, 0.0]),
        }
        basis, _ = compute_basis_from_definition(rb, positions)
        det = np.linalg.det(basis)
        np.testing.assert_allclose(det, 1.0, atol=1e-10)

    def test_vertical_bone_uses_fallback(self) -> None:
        """When bone is parallel to reference_up (+Z), fallback perpendicular works."""
        rb = _make_upper_arm_rb()
        positions = {
            "shoulder": np.array([0.0, 0.0, 0.0]),
            "elbow": np.array([0.0, 0.0, 1.0]),  # straight up
        }
        basis, _ = compute_basis_from_definition(rb, positions)
        # Should still produce valid orthonormal basis
        for i in range(3):
            np.testing.assert_allclose(np.linalg.norm(basis[i]), 1.0, atol=1e-10)


class TestSwingQuaternion:
    def test_identity_when_same_direction(self) -> None:
        q = compute_swing_quaternion(np.array([1., 0., 0.]), np.array([1., 0., 0.]))
        np.testing.assert_allclose(q, [1., 0., 0., 0.], atol=1e-6)

    def test_90_degree_rotation(self) -> None:
        rest = np.array([1.0, 0.0, 0.0])
        observed = np.array([0.0, 1.0, 0.0])
        q = compute_swing_quaternion(rest, observed)
        np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-10)
        rotated = _rotate_vector_by_quaternion(q, rest)
        np.testing.assert_allclose(rotated, observed, atol=1e-6)

    def test_180_degree_rotation(self) -> None:
        rest = np.array([1.0, 0.0, 0.0])
        observed = np.array([-1.0, 0.0, 0.0])
        q = compute_swing_quaternion(rest, observed)
        rotated = _rotate_vector_by_quaternion(q, rest)
        np.testing.assert_allclose(rotated, observed, atol=1e-6)

    def test_trajectory_matches_single_frame(self) -> None:
        rest = np.array([1.0, 0.0, 0.0])
        directions = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
        quats = compute_swing_quaternion_trajectory(rest, directions)
        assert quats.shape == (3, 4)
        for i in range(3):
            single = compute_swing_quaternion(rest, directions[i])
            np.testing.assert_allclose(quats[i], single, atol=1e-10)


def _rotate_vector_by_quaternion(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])
    return R @ v
