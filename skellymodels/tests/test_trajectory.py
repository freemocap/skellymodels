"""Tests for core trajectory and timeseries models."""

import numpy as np
import pytest

from skellymodels.core.trajectory.trajectory import Trajectory
from skellymodels.core.trajectory.typed_trajectories import (
    AngularAccelerationTrajectory,
    AngularVelocityTrajectory,
    QuaternionTrajectory,
    SpatialTrajectory,
)
from skellymodels.core.timeseries import Timeseries
from skellymodels.tests.conftest import (
    BODY_MARKER_NAMES,
    QUATERNION_MARKER_NAMES,
    SMALL_MARKER_NAMES,
)


# ============================================================
# Base Trajectory
# ============================================================


class TestTrajectoryConstruction:
    def test_basic_construction(self, spatial_array_small: np.ndarray) -> None:
        t = Trajectory(
            name="test",
            keypoint_names=SMALL_MARKER_NAMES,
            array=spatial_array_small,
        )
        assert t.num_frames == 100
        assert t.num_keypoints == 3
        assert t.num_dimensions == 3
        assert t.name == "test"

    def test_marker_names_are_immutable_tuple(self, spatial_array_small: np.ndarray) -> None:
        t = Trajectory(
            name="test",
            keypoint_names=SMALL_MARKER_NAMES,
            array=spatial_array_small,
        )
        assert isinstance(t.keypoint_names, tuple)


class TestTrajectoryValidation:
    def test_rejects_2d_array(self) -> None:
        with pytest.raises(ValueError, match="3-dimensional"):
            Trajectory(
                name="bad",
                keypoint_names=("a",),
                array=np.zeros((10, 3)),
            )

    def test_rejects_4d_array(self) -> None:
        with pytest.raises(ValueError, match="3-dimensional"):
            Trajectory(
                name="bad",
                keypoint_names=("a",),
                array=np.zeros((10, 1, 3, 1)),
            )

    def test_rejects_marker_count_mismatch(self) -> None:
        with pytest.raises(ValueError, match="marker names"):
            Trajectory(
                name="bad",
                keypoint_names=("a", "b"),
                array=np.zeros((10, 3, 3)),
            )

    def test_rejects_zero_frames(self) -> None:
        with pytest.raises(ValueError, match="at least 1 frame"):
            Trajectory(
                name="bad",
                keypoint_names=("a",),
                array=np.zeros((0, 1, 3)),
            )

    def test_rejects_duplicate_marker_names(self) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            Trajectory(
                name="bad",
                keypoint_names=("a", "a"),
                array=np.zeros((10, 2, 3)),
            )


class TestTrajectoryAccess:
    def test_getitem_returns_correct_shape(self, spatial_array_small: np.ndarray) -> None:
        t = Trajectory(
            name="test",
            keypoint_names=SMALL_MARKER_NAMES,
            array=spatial_array_small,
        )
        result = t["marker_b"]
        assert result.shape == (100, 3)

    def test_getitem_returns_correct_data(self) -> None:
        data = np.arange(30, dtype=np.float64).reshape(2, 5, 3)
        names = ("a", "b", "c", "d", "e")
        t = Trajectory(name="test", keypoint_names=names, array=data)
        np.testing.assert_array_equal(t["c"], data[:, 2, :])

    def test_getitem_raises_on_unknown_name(self, spatial_array_small: np.ndarray) -> None:
        t = Trajectory(
            name="test",
            keypoint_names=SMALL_MARKER_NAMES,
            array=spatial_array_small,
        )
        with pytest.raises(KeyError, match="nonexistent"):
            t["nonexistent"]

    def test_contains(self, spatial_array_small: np.ndarray) -> None:
        t = Trajectory(
            name="test",
            keypoint_names=SMALL_MARKER_NAMES,
            array=spatial_array_small,
        )
        assert "marker_a" in t
        assert "nonexistent" not in t

    def test_len(self, spatial_array_small: np.ndarray) -> None:
        t = Trajectory(
            name="test",
            keypoint_names=SMALL_MARKER_NAMES,
            array=spatial_array_small,
        )
        assert len(t) == 3

    def test_as_dict(self) -> None:
        data = np.arange(18, dtype=np.float64).reshape(2, 3, 3)
        names = ("x", "y", "z")
        t = Trajectory(name="test", keypoint_names=names, array=data)
        d = t.as_dict
        assert set(d.keys()) == {"x", "y", "z"}
        np.testing.assert_array_equal(d["y"], data[:, 1, :])

    def test_as_dataframe_shape(self, spatial_array_body: np.ndarray) -> None:
        t = Trajectory(
            name="test",
            keypoint_names=BODY_MARKER_NAMES,
            array=spatial_array_body,
        )
        df = t.as_dataframe
        assert df.shape == (100 * 33, 3 + 2)  # 3 dims + frame + marker
        assert list(df.columns[:2]) == ["frame", "keypoint"]

    def test_as_array_is_same_object(self, spatial_array_small: np.ndarray) -> None:
        t = Trajectory(
            name="test",
            keypoint_names=SMALL_MARKER_NAMES,
            array=spatial_array_small,
        )
        assert t.as_array is t.array


# ============================================================
# SpatialTrajectory
# ============================================================


class TestSpatialTrajectory:
    def test_accepts_3d(self, spatial_array_small: np.ndarray) -> None:
        t = SpatialTrajectory(
            name="spatial",
            keypoint_names=SMALL_MARKER_NAMES,
            array=spatial_array_small,
        )
        assert t.num_dimensions == 3

    def test_rejects_non_3d(self) -> None:
        with pytest.raises(ValueError, match="3 dimensions"):
            SpatialTrajectory(
                name="bad",
                keypoint_names=("a",),
                array=np.zeros((10, 1, 4)),
            )

    def test_dataframe_has_xyz_columns(self, spatial_array_small: np.ndarray) -> None:
        t = SpatialTrajectory(
            name="spatial",
            keypoint_names=SMALL_MARKER_NAMES,
            array=spatial_array_small,
        )
        df = t.as_dataframe
        assert list(df.columns) == ["frame", "keypoint", "x", "y", "z"]


# ============================================================
# QuaternionTrajectory
# ============================================================


class TestQuaternionTrajectory:
    def test_accepts_unit_quaternions(self, unit_quaternion_array: np.ndarray) -> None:
        t = QuaternionTrajectory(
            name="quat",
            keypoint_names=QUATERNION_MARKER_NAMES,
            array=unit_quaternion_array,
        )
        assert t.num_dimensions == 4

    def test_rejects_non_unit_quaternions(self, non_unit_quaternion_array: np.ndarray) -> None:
        with pytest.raises(ValueError, match="unit quaternions"):
            QuaternionTrajectory(
                name="bad",
                keypoint_names=QUATERNION_MARKER_NAMES,
                array=non_unit_quaternion_array,
            )

    def test_rejects_non_4d(self) -> None:
        with pytest.raises(ValueError, match="4 dimensions"):
            QuaternionTrajectory(
                name="bad",
                keypoint_names=("a",),
                array=np.zeros((10, 1, 3)),
            )

    def test_dataframe_has_wxyz_columns(self, unit_quaternion_array: np.ndarray) -> None:
        t = QuaternionTrajectory(
            name="quat",
            keypoint_names=QUATERNION_MARKER_NAMES,
            array=unit_quaternion_array,
        )
        df = t.as_dataframe
        assert list(df.columns) == ["frame", "keypoint", "w", "x", "y", "z"]

    def test_to_rotation_matrices_shape(self, unit_quaternion_array: np.ndarray) -> None:
        t = QuaternionTrajectory(
            name="quat",
            keypoint_names=QUATERNION_MARKER_NAMES,
            array=unit_quaternion_array,
        )
        R = t.to_rotation_matrices()
        assert R.shape == (100, 5, 3, 3)

    def test_identity_quaternion_gives_identity_matrix(self) -> None:
        identity = np.array([[[1.0, 0.0, 0.0, 0.0]]])  # (1, 1, 4) wxyz
        t = QuaternionTrajectory(
            name="identity",
            keypoint_names=("test",),
            array=identity,
        )
        R = t.to_rotation_matrices()
        np.testing.assert_allclose(R[0, 0], np.eye(3), atol=1e-10)


# ============================================================
# AngularVelocity / AngularAcceleration
# ============================================================


class TestAngularVelocityTrajectory:
    def test_magnitude(self) -> None:
        data = np.array([[[3.0, 4.0, 0.0]]])  # (1, 1, 3)
        t = AngularVelocityTrajectory(
            name="omega",
            keypoint_names=("joint",),
            array=data,
        )
        mag = t.magnitude()
        assert mag.shape == (1, 1)
        np.testing.assert_allclose(mag[0, 0], 5.0)

    def test_rejects_non_3d(self) -> None:
        with pytest.raises(ValueError, match="3 dimensions"):
            AngularVelocityTrajectory(
                name="bad",
                keypoint_names=("a",),
                array=np.zeros((10, 1, 4)),
            )


class TestAngularAccelerationTrajectory:
    def test_rejects_non_3d(self) -> None:
        with pytest.raises(ValueError, match="3 dimensions"):
            AngularAccelerationTrajectory(
                name="bad",
                keypoint_names=("a",),
                array=np.zeros((10, 1, 2)),
            )


# ============================================================
# Timeseries
# ============================================================


class TestTimeseries:
    def test_basic_construction(self) -> None:
        ts = Timeseries(
            name="x_pos",
            timestamps=np.arange(10, dtype=np.float64),
            values=np.sin(np.arange(10, dtype=np.float64)),
        )
        assert ts.n_frames == 10
        assert ts.name == "x_pos"

    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="length"):
            Timeseries(
                name="bad",
                timestamps=np.arange(10, dtype=np.float64),
                values=np.arange(5, dtype=np.float64),
            )

    def test_rejects_2d_values(self) -> None:
        with pytest.raises(ValueError, match="1-dimensional"):
            Timeseries(
                name="bad",
                timestamps=np.arange(10, dtype=np.float64),
                values=np.zeros((10, 2)),
            )

    def test_duration_and_mean_dt(self) -> None:
        ts = Timeseries(
            name="test",
            timestamps=np.array([0.0, 0.1, 0.2, 0.3, 0.4]),
            values=np.zeros(5),
        )
        np.testing.assert_allclose(ts.duration, 0.4)
        np.testing.assert_allclose(ts.mean_dt, 0.1)

    def test_single_sample_duration(self) -> None:
        ts = Timeseries(
            name="test",
            timestamps=np.array([1.0]),
            values=np.array([42.0]),
        )
        assert ts.duration == 0.0
        assert ts.mean_dt == 0.0

    def test_differentiate_linear(self) -> None:
        """Derivative of a linear function y = 2t should be constant 2."""
        t = np.linspace(0, 1, 100)
        y = 2.0 * t
        ts = Timeseries(name="linear", timestamps=t, values=y)
        dts = ts.differentiate()
        np.testing.assert_allclose(dts.values, 2.0, atol=1e-10)

    def test_differentiate_too_short(self) -> None:
        ts = Timeseries(
            name="short",
            timestamps=np.array([0.0]),
            values=np.array([1.0]),
        )
        with pytest.raises(ValueError, match="fewer than 2"):
            ts.differentiate()

    def test_interpolate(self) -> None:
        ts = Timeseries(
            name="test",
            timestamps=np.array([0.0, 1.0, 2.0]),
            values=np.array([0.0, 10.0, 20.0]),
        )
        new_ts = ts.interpolate(target_timestamps=np.array([0.5, 1.5]))
        np.testing.assert_allclose(new_ts.values, [5.0, 15.0])

    def test_getitem(self) -> None:
        ts = Timeseries(
            name="test",
            timestamps=np.arange(3, dtype=np.float64),
            values=np.array([10.0, 20.0, 30.0]),
        )
        assert ts[1] == 20.0
