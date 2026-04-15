"""
Kinematics bridge — connects the definition layer to the runtime kinematics layer.

Provides the Type-to-Token bridge:
  RigidBodyDefinition + measured keypoint positions → basis vectors + origin

Handles both fully-constrained and under-constrained bodies:
  Fully constrained: Gram-Schmidt orthogonalization from 2 defined axes
  Under-constrained: primary axis from data, secondary/tertiary via swing-only
    (zero twist relative to a reference up vector, equivalent to Blender's Damped Track)

Coordinate convention: +X forward, +Y left, +Z up. Right-handed only.
"""

import numpy as np
from numpy.typing import NDArray

from skellymodels.core.rigid_body.rigid_body_definition import (
    RigidBodyDefinition,
)
from skellymodels.type_aliases import KeypointName

# Default reference "up" vector for swing-only computation on under-constrained bodies.
# +Z up per the global coordinate convention.
REFERENCE_UP: NDArray[np.float64] = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def compute_basis_from_definition(
    rigid_body: RigidBodyDefinition,
    keypoint_positions: dict[KeypointName, NDArray[np.float64]],
    reference_up: NDArray[np.float64] = REFERENCE_UP,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute orthonormal basis vectors and origin for any rigid body.

    For fully-constrained bodies (2 defined axes): Gram-Schmidt orthogonalization.
    For under-constrained bodies (1 defined axis): primary axis from data,
    secondary/tertiary computed via swing-only relative to reference_up.

    Args:
        rigid_body: the topological definition
        keypoint_positions: measured 3D positions for each keypoint
        reference_up: reference "up" vector for swing-only computation
            on under-constrained bodies (default: +Z = [0,0,1])

    Returns:
        basis_vectors: (3, 3) array where rows are [x_axis, y_axis, z_axis]
        origin_point: (3,) origin position
    """
    # Validate all keypoints present
    missing = set(rigid_body.keypoints) - set(keypoint_positions.keys())
    if missing:
        raise KeyError(
            f"Missing measured positions for keypoints: {sorted(missing)}. "
            f"Rigid body '{rigid_body.name}' requires: {rigid_body.keypoints}"
        )

    frame = rigid_body.coordinate_frame

    # Compute origin
    origin_points = [keypoint_positions[kp] for kp in frame.origin_keypoints]
    origin = np.mean(origin_points, axis=0)

    # Get exact axis direction
    exact_name, exact_def = frame.exact_axis
    exact_points = [keypoint_positions[kp] for kp in exact_def.keypoints]
    exact_target = np.mean(exact_points, axis=0)
    exact_vec = exact_target - origin
    exact_norm = np.linalg.norm(exact_vec)
    if exact_norm < 1e-10:
        raise ValueError(
            f"Exact axis '{exact_name}' has near-zero length for body "
            f"'{rigid_body.name}'. Keypoints may be coincident with origin."
        )
    exact_vec = exact_vec / exact_norm

    if rigid_body.is_fully_constrained:
        # Gram-Schmidt: orthogonalize approximate axis against exact axis
        approx_result = frame.approximate_axis
        assert approx_result is not None
        approx_name, approx_def = approx_result

        approx_points = [keypoint_positions[kp] for kp in approx_def.keypoints]
        approx_target = np.mean(approx_points, axis=0)
        approx_vec = approx_target - origin
        approx_vec = approx_vec - np.dot(approx_vec, exact_vec) * exact_vec
        approx_norm = np.linalg.norm(approx_vec)
        if approx_norm < 1e-10:
            raise ValueError(
                f"Approximate axis '{approx_name}' is parallel to exact axis "
                f"for body '{rigid_body.name}'. Choose different keypoints."
            )
        approx_vec = approx_vec / approx_norm

        # Third axis via cross product
        computed_name = frame.computed_axis_names[0]
        computed_vec = _compute_third_axis(
            exact_name=exact_name,
            approx_name=approx_name,
            exact_vec=exact_vec,
            approx_vec=approx_vec,
        )

        # Assemble basis
        axis_index = {"x_axis": 0, "y_axis": 1, "z_axis": 2}
        basis = np.zeros((3, 3), dtype=np.float64)
        basis[axis_index[exact_name]] = exact_vec
        basis[axis_index[approx_name]] = approx_vec
        basis[axis_index[computed_name]] = computed_vec

    else:
        # Under-constrained: swing-only basis from exact axis + reference up
        # Primary axis is exact_vec. Build perpendicular axes using reference_up.
        axis_index = {"x_axis": 0, "y_axis": 1, "z_axis": 2}
        primary_idx = axis_index[exact_name]

        # Compute secondary axis perpendicular to primary and reference_up
        secondary_vec = np.cross(exact_vec, reference_up)
        secondary_norm = np.linalg.norm(secondary_vec)
        if secondary_norm < 1e-10:
            # Primary axis is parallel to reference_up — use an arbitrary perpendicular
            fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            secondary_vec = np.cross(exact_vec, fallback)
            secondary_norm = np.linalg.norm(secondary_vec)
            if secondary_norm < 1e-10:
                fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                secondary_vec = np.cross(exact_vec, fallback)
                secondary_norm = np.linalg.norm(secondary_vec)
        secondary_vec = secondary_vec / secondary_norm

        # Tertiary axis completes the right-handed system
        tertiary_vec = np.cross(exact_vec, secondary_vec)
        tertiary_norm = np.linalg.norm(tertiary_vec)
        if tertiary_norm > 1e-10:
            tertiary_vec = tertiary_vec / tertiary_norm

        # Assign to the two computed axis slots
        computed_names = frame.computed_axis_names  # 2 names for under-constrained
        basis = np.zeros((3, 3), dtype=np.float64)
        basis[primary_idx] = exact_vec
        # Assign secondary and tertiary in axis order
        remaining = sorted(
            [axis_index[n] for n in computed_names]
        )
        basis[remaining[0]] = secondary_vec
        basis[remaining[1]] = tertiary_vec

        # Ensure right-handedness: det(basis) should be +1
        if np.linalg.det(basis) < 0:
            basis[remaining[1]] = -basis[remaining[1]]

    return basis, origin


def build_reference_geometry_data(
    rigid_body: RigidBodyDefinition,
    keypoint_positions: dict[KeypointName, NDArray[np.float64]],
) -> dict:
    """
    Bridge from RigidBodyDefinition to a dict compatible with ReferenceGeometry.

    Works for both fully-constrained and under-constrained bodies.

    Args:
        rigid_body: the topological definition
        keypoint_positions: measured 3D positions for each keypoint

    Returns:
        dict suitable for ReferenceGeometry(**result)
    """
    missing = set(rigid_body.keypoints) - set(keypoint_positions.keys())
    if missing:
        raise KeyError(
            f"Missing measured positions for keypoints: {sorted(missing)}. "
            f"Rigid body '{rigid_body.name}' requires: {rigid_body.keypoints}"
        )

    keypoints_dict = {}
    for kp_name in rigid_body.keypoints:
        pos = keypoint_positions[kp_name]
        if pos.shape != (3,):
            raise ValueError(
                f"Keypoint '{kp_name}' position must have shape (3,), got {pos.shape}"
            )
        keypoints_dict[kp_name] = {
            "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])
        }

    frame = rigid_body.coordinate_frame
    coordinate_frame_dict: dict = {
        "origin_keypoints": frame.origin_keypoints,
    }
    for axis_name in ("x_axis", "y_axis", "z_axis"):
        axis_def = getattr(frame, axis_name)
        if axis_def is not None:
            coordinate_frame_dict[axis_name] = {
                "keypoints": axis_def.keypoints,
                "type": axis_def.type.value,
            }

    return {
        "units": "mm",
        "coordinate_frame": coordinate_frame_dict,
        "keypoints": keypoints_dict,
    }


def compute_swing_quaternion(
    rest_direction: NDArray[np.float64],
    observed_direction: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute the swing-only quaternion rotating rest_direction to observed_direction.

    Zero twist by construction — equivalent to Blender's Damped Track constraint.

    Args:
        rest_direction: (3,) unit vector of the bone's rest-pose direction
        observed_direction: (3,) unit vector of the bone's observed direction

    Returns:
        (4,) quaternion in wxyz order
    """
    rest_dir = rest_direction / np.linalg.norm(rest_direction)
    obs_dir = observed_direction / np.linalg.norm(observed_direction)

    dot = np.clip(np.dot(rest_dir, obs_dir), -1.0, 1.0)

    if dot > 0.9999:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    if dot < -0.9999:
        perp = np.cross(rest_dir, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(rest_dir, np.array([0.0, 1.0, 0.0]))
        perp = perp / np.linalg.norm(perp)
        return np.array([0.0, perp[0], perp[1], perp[2]], dtype=np.float64)

    axis = np.cross(rest_dir, obs_dir)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = axis / axis_norm

    half_cos = np.sqrt(0.5 * (1.0 + dot))
    half_sin = np.sqrt(0.5 * (1.0 - dot))

    return np.array(
        [half_cos, axis[0] * half_sin, axis[1] * half_sin, axis[2] * half_sin],
        dtype=np.float64,
    )


def compute_swing_quaternion_trajectory(
    rest_direction: NDArray[np.float64],
    observed_directions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Vectorized swing-only quaternion computation over multiple frames.

    Args:
        rest_direction: (3,) unit vector of the bone's rest-pose direction
        observed_directions: (F, 3) array of observed directions per frame

    Returns:
        (F, 4) quaternion array in wxyz order
    """
    num_frames = observed_directions.shape[0]
    quaternions = np.empty((num_frames, 4), dtype=np.float64)
    for i in range(num_frames):
        quaternions[i] = compute_swing_quaternion(rest_direction, observed_directions[i])
    return quaternions


def _compute_third_axis(
    exact_name: str,
    approx_name: str,
    exact_vec: NDArray[np.float64],
    approx_vec: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute the third basis axis via cross product for a right-handed system.
    Right-hand rule: X × Y = Z, Y × Z = X, Z × X = Y.
    """
    axis_order = {"x_axis": 0, "y_axis": 1, "z_axis": 2}
    exact_idx = axis_order[exact_name]
    approx_idx = axis_order[approx_name]

    if (exact_idx + 1) % 3 == approx_idx:
        computed_vec = np.cross(exact_vec, approx_vec)
    elif (approx_idx + 1) % 3 == exact_idx:
        computed_vec = np.cross(approx_vec, exact_vec)
    else:
        if (exact_idx + 2) % 3 == approx_idx:
            computed_vec = np.cross(approx_vec, exact_vec)
        else:
            computed_vec = np.cross(exact_vec, approx_vec)

    norm = np.linalg.norm(computed_vec)
    if norm < 1e-10:
        raise ValueError("Cross product resulted in zero vector — axes may be parallel")
    return computed_vec / norm
