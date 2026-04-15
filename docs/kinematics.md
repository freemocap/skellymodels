# Kinematics

**Module:** [`kinematics/bridge.py`](../skellymodels/kinematics/bridge.py)

## Quaternion orientation

All orientations in skellymodels use unit quaternions in `[w, x, y, z]` order (scalar-first). Quaternions are the canonical representation because they are the only safe way to define 3D orientation without gimbal lock or polar singularities.

The `QuaternionTrajectory` class (see [Ontology — Trajectory](ontology.md#trajectory)) stores orientation data and provides conversion methods: `to_rotation_matrices()` for direction cosine matrices, plus helpers for Euler angles and axis-angle representations. These conversions are for output convenience — the internal representation is always quaternion.

SLERP (Spherical Linear Interpolation) is used for temporal interpolation between quaternion keyframes in trajectory computation.

## Coordinate frame construction

For fully-constrained bodies (2 defined axes, 3+ non-colinear keypoints), the body-fixed coordinate frame is computed via Gram-Schmidt orthogonalization.

**Algorithm:**

1. **Compute origin** as the mean position of `origin_keypoints`.
2. **Exact axis**: vector from origin toward the mean of the exact axis keypoints, normalized. This direction is preserved exactly.
3. **Approximate axis**: vector from origin toward the approximate axis keypoints. Orthogonalize via Gram-Schmidt (remove the component parallel to the exact axis). Normalize. This axis is perpendicular to exact while staying as close as possible to the specified direction.
4. **Third axis**: cross product of the first two, ordered for a right-handed system (X × Y = Z, Y × Z = X, Z × X = Y).

The result is a `(3, 3)` orthonormal basis matrix where rows correspond to X, Y, Z axes.

**Why exact vs approximate?** Keypoints come from noisy motion capture data. The exact axis is the direction you care about most (e.g. "which way is the nose pointing"). It's preserved exactly. The approximate axis provides the secondary reference but is adjusted to be orthogonal. Frame-to-frame noise in the approximate axis direction cannot corrupt the exact axis.

## Swing-only quaternions

For under-constrained bodies (1 defined axis, 2 keypoints), the twist (roll around the primary axis) has no data to constrain it. The kinematics bridge computes a full basis using **swing-only decomposition** — the rotation that changes the bone's direction with zero twist.

This is the same algorithm behind Blender's "Damped Track" constraint, which uses a pure swing rotation to minimize rolling around the tracking axis.

**Algorithm for single-frame basis:**

1. Compute the exact (primary) axis direction from the keypoint data.
2. Cross the primary axis with a `reference_up` vector (default: +Z = `[0, 0, 1]`) to get the secondary axis.
3. Cross the primary and secondary to get the tertiary axis.
4. Enforce right-handedness (det(basis) = +1).

If the primary axis is parallel to reference_up, an arbitrary perpendicular fallback is used.

**Algorithm for quaternion trajectory:**

Given `rest_direction` (bone direction in reference pose) and `observed_direction` (current frame):

1. Rotation axis = `cross(rest, observed)`, normalized.
2. Rotation angle from `dot(rest, observed)`.
3. Half-angle quaternion construction → wxyz.

The resulting quaternion has zero twist by construction. The secondary and tertiary axes are implicit (determined by the reference_up vector at basis computation time).

## The Type→Token bridge

The bridge between abstract definitions and runtime computation:

```python
from skellymodels.kinematics.bridge import compute_basis_from_definition

# Works for BOTH fully-constrained and under-constrained bodies
basis, origin = compute_basis_from_definition(
    rigid_body=skull_definition,
    keypoint_positions={"skull_origin": np.array([...]), ...},
)
# basis: (3, 3) orthonormal — always valid, always right-handed
# origin: (3,) position

# For under-constrained, you can optionally specify a reference_up:
basis, origin = compute_basis_from_definition(
    rigid_body=upper_arm_rb,
    keypoint_positions={...},
    reference_up=np.array([0., 0., 1.]),  # default: +Z
)
```

For trajectory-level quaternion computation on under-constrained bodies:

```python
from skellymodels.kinematics.bridge import compute_swing_quaternion_trajectory

rest_direction = np.array([0, 0, -1])  # bone points downward at rest
observed = trajectory.keypoints.right_elbow - trajectory.keypoints.right_shoulder
observed /= np.linalg.norm(observed, axis=1, keepdims=True)

quaternions = compute_swing_quaternion_trajectory(rest_direction, observed)
# (num_frames, 4) wxyz, unit norm, zero twist
```

## Coordinate convention

Enforced globally: **+X forward, +Y left, +Z up. Right-handed.**

This is the robotics/aerospace convention (Z-up). All coordinate frame definitions, basis computations, and cross products in skellymodels enforce right-handedness. The `reference_up` default of `[0, 0, 1]` assumes +Z up.

**Quaternion ordering:** `[w, x, y, z]` (scalar-first). This matches SciPy's `Rotation.as_quat(scalar_first=True)`. Note that some libraries use `[x, y, z, w]` (scalar-last) — be careful at boundaries.
