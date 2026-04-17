"""
Shared test fixtures for skellymodels tests.

Provides synthetic data arrays and sample definitions that multiple
test modules depend on.
"""

import numpy as np
import pytest


# --- Marker name fixtures ---

BODY_MARKER_NAMES = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
)

SMALL_MARKER_NAMES = ("marker_a", "marker_b", "marker_c")

QUATERNION_MARKER_NAMES = ("skull", "pelvis", "right_upper_arm", "left_upper_arm", "thorax")


@pytest.fixture
def num_frames() -> int:
    return 100


@pytest.fixture
def body_marker_names() -> tuple[str, ...]:
    return BODY_MARKER_NAMES


@pytest.fixture
def small_marker_names() -> tuple[str, ...]:
    return SMALL_MARKER_NAMES


@pytest.fixture
def spatial_array_body(num_frames: int) -> np.ndarray:
    """Synthetic (100, 33, 3) spatial trajectory mimicking mediapipe body output."""
    rng = np.random.default_rng(seed=42)
    return rng.standard_normal((num_frames, len(BODY_MARKER_NAMES), 3))


@pytest.fixture
def spatial_array_small(num_frames: int) -> np.ndarray:
    """Synthetic (100, 3, 3) spatial trajectory for quick tests."""
    rng = np.random.default_rng(seed=42)
    return rng.standard_normal((num_frames, len(SMALL_MARKER_NAMES), 3))


@pytest.fixture
def unit_quaternion_array(num_frames: int) -> np.ndarray:
    """Synthetic (100, 5, 4) unit quaternion trajectory."""
    rng = np.random.default_rng(seed=42)
    raw = rng.standard_normal((num_frames, len(QUATERNION_MARKER_NAMES), 4))
    norms = np.linalg.norm(raw, axis=2, keepdims=True)
    return raw / norms


@pytest.fixture
def non_unit_quaternion_array(num_frames: int) -> np.ndarray:
    """Synthetic (100, 5, 4) array that is NOT unit-norm quaternions."""
    rng = np.random.default_rng(seed=42)
    return rng.standard_normal((num_frames, len(QUATERNION_MARKER_NAMES), 4)) * 5.0
