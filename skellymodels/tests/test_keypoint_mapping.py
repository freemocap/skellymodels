"""Tests for keypoint mapping types and YAML loading."""

from pathlib import Path

import numpy as np
import pytest

from skellymodels.mapping.keypoint_mapping import KeypointMapping
from skellymodels.mapping.mapping_loader import load_mapping_from_yaml

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
MEDIAPIPE_MAPPING_PATH = CONFIGS_DIR / "mappings" / "mediapipe_human_body.yaml"


# ============================================================
# Construction + Validation
# ============================================================


class TestKeypointMappingValidation:
    def test_basic_construction(self) -> None:
        km = KeypointMapping(
            tracker_name="test",
            skeleton_name="test_skel",
            mappings={"kp_a": "tp_a", "kp_b": ["tp_a", "tp_b"]},
        )
        assert km.tracker_name == "test"
        assert km.skeleton_keypoint_names == ["kp_a", "kp_b"]

    def test_rejects_empty_mappings(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            KeypointMapping(
                tracker_name="t", skeleton_name="s", mappings={}
            )

    def test_rejects_empty_list_source(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            KeypointMapping(
                tracker_name="t", skeleton_name="s",
                mappings={"bad": []},
            )

    def test_rejects_empty_dict_source(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            KeypointMapping(
                tracker_name="t", skeleton_name="s",
                mappings={"bad": {}},
            )

    def test_accepts_weights_not_summing_to_one(self) -> None:
        """Weights can sum to any value (supports extrapolation)."""
        km = KeypointMapping(
            tracker_name="t", skeleton_name="s",
            mappings={"extrapolated": {"a": 1.5, "b": -0.5}},
        )
        assert km.skeleton_keypoint_names == ["extrapolated"]

    def test_required_tracker_points(self) -> None:
        km = KeypointMapping(
            tracker_name="t", skeleton_name="s",
            mappings={
                "direct": "tp_a",
                "averaged": ["tp_b", "tp_c"],
                "weighted": {"tp_a": 0.5, "tp_d": 0.5},
            },
        )
        assert km.required_tracker_points == {"tp_a", "tp_b", "tp_c", "tp_d"}


# ============================================================
# apply() — correctness
# ============================================================


class TestKeypointMappingApply:
    def _make_mapping(self) -> KeypointMapping:
        return KeypointMapping(
            tracker_name="test", skeleton_name="test_skel",
            mappings={
                "direct": "tp_a",
                "averaged": ["tp_a", "tp_b"],
                "weighted": {"tp_a": 0.3, "tp_b": 0.7},
            },
        )

    def test_output_shape(self) -> None:
        km = self._make_mapping()
        data = np.zeros((50, 2, 3))
        result = km.apply(
            tracked_array=data,
            tracked_point_names=["tp_a", "tp_b"],
        )
        assert result.shape == (50, 3, 3)  # 3 skeleton keypoints

    def test_direct_mapping_copies_exactly(self) -> None:
        km = self._make_mapping()
        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 2, 3))
        result = km.apply(data, ["tp_a", "tp_b"])
        # "direct" maps to tp_a (index 0)
        np.testing.assert_array_equal(result[:, 0, :], data[:, 0, :])

    def test_list_mapping_computes_mean(self) -> None:
        km = self._make_mapping()
        # tp_a = [1,2,3], tp_b = [3,4,5] → mean = [2,3,4]
        data = np.array([
            [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]
        ])  # (1, 2, 3)
        result = km.apply(data, ["tp_a", "tp_b"])
        np.testing.assert_allclose(result[0, 1, :], [2.0, 3.0, 4.0])

    def test_dict_mapping_computes_weighted_sum(self) -> None:
        km = self._make_mapping()
        # tp_a = [10,0,0], tp_b = [0,10,0]
        # weighted = 0.3*[10,0,0] + 0.7*[0,10,0] = [3,7,0]
        data = np.array([
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]
        ])  # (1, 2, 3)
        result = km.apply(data, ["tp_a", "tp_b"])
        np.testing.assert_allclose(result[0, 2, :], [3.0, 7.0, 0.0])

    def test_multi_frame(self) -> None:
        km = KeypointMapping(
            tracker_name="t", skeleton_name="s",
            mappings={"out": ["a", "b"]},
        )
        data = np.array([
            [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[5.0, 0.0, 0.0], [7.0, 0.0, 0.0]],
        ])  # (2, 2, 3)
        result = km.apply(data, ["a", "b"])
        np.testing.assert_allclose(result[0, 0, 0], 2.0)  # mean(1,3)
        np.testing.assert_allclose(result[1, 0, 0], 6.0)  # mean(5,7)


# ============================================================
# apply() — error cases
# ============================================================


class TestKeypointMappingApplyErrors:
    def test_rejects_2d_array(self) -> None:
        km = KeypointMapping(
            tracker_name="t", skeleton_name="s",
            mappings={"out": "a"},
        )
        with pytest.raises(ValueError, match="3-dimensional"):
            km.apply(np.zeros((10, 3)), ["a"])

    def test_rejects_name_count_mismatch(self) -> None:
        km = KeypointMapping(
            tracker_name="t", skeleton_name="s",
            mappings={"out": "a"},
        )
        with pytest.raises(ValueError, match="names provided"):
            km.apply(np.zeros((10, 2, 3)), ["a"])

    def test_rejects_missing_tracker_point(self) -> None:
        km = KeypointMapping(
            tracker_name="t", skeleton_name="s",
            mappings={"out": "missing_point"},
        )
        with pytest.raises(KeyError, match="missing_point"):
            km.apply(np.zeros((10, 1, 3)), ["a"])


# ============================================================
# YAML loading — mediapipe
# ============================================================


class TestMediapipeMappingYAML:
    def test_loads_successfully(self) -> None:
        km = load_mapping_from_yaml(MEDIAPIPE_MAPPING_PATH)
        assert km.tracker_name == "mediapipe"
        assert km.skeleton_name == "human_body"

    def test_has_expected_skeleton_keypoints(self) -> None:
        km = load_mapping_from_yaml(MEDIAPIPE_MAPPING_PATH)
        kp_names = set(km.skeleton_keypoint_names)
        # Spot-check critical keypoints
        assert "skull_origin_foramen_magnum" in kp_names
        assert "nose_tip" in kp_names
        assert "right_shoulder" in kp_names
        assert "pelvis_spine_sacrum_origin" in kp_names
        assert "left_hallux_tip" in kp_names

    def test_required_tracker_points_are_mediapipe(self) -> None:
        km = load_mapping_from_yaml(MEDIAPIPE_MAPPING_PATH)
        required = km.required_tracker_points
        # All 33 mediapipe body points should be referenced
        assert "nose" in required
        assert "left_ear" in required
        assert "right_foot_index" in required

    def test_apply_produces_correct_shape(self) -> None:
        km = load_mapping_from_yaml(MEDIAPIPE_MAPPING_PATH)
        # Build tracker point name list (order from mediapipe)
        tracker_names = [
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
        rng = np.random.default_rng(42)
        data = rng.standard_normal((50, len(tracker_names), 3))
        result = km.apply(data, tracker_names)
        num_skeleton_kps = len(km.skeleton_keypoint_names)
        assert result.shape == (50, num_skeleton_kps, 3)

    def test_direct_mapping_exact_copy(self) -> None:
        """nose_tip maps directly to nose — data should be identical."""
        km = load_mapping_from_yaml(MEDIAPIPE_MAPPING_PATH)
        tracker_names = [
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
        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, len(tracker_names), 3))
        result = km.apply(data, tracker_names)

        nose_idx_tracker = tracker_names.index("nose")
        nose_tip_idx_skeleton = km.skeleton_keypoint_names.index("nose_tip")
        np.testing.assert_array_equal(
            result[:, nose_tip_idx_skeleton, :],
            data[:, nose_idx_tracker, :],
        )

    def test_averaged_mapping_is_mean(self) -> None:
        """skull_origin_foramen_magnum = mean(left_ear, right_ear)."""
        km = load_mapping_from_yaml(MEDIAPIPE_MAPPING_PATH)
        tracker_names = [
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
        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, len(tracker_names), 3))
        result = km.apply(data, tracker_names)

        left_ear_idx = tracker_names.index("left_ear")
        right_ear_idx = tracker_names.index("right_ear")
        skull_idx = km.skeleton_keypoint_names.index("skull_origin_foramen_magnum")

        expected = (data[:, left_ear_idx, :] + data[:, right_ear_idx, :]) / 2.0
        np.testing.assert_allclose(result[:, skull_idx, :], expected)
