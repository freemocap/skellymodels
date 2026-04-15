# Ontology — Type definitions and configuration

This document formally defines every structural type in skellymodels and the YAML configuration format for each. The mathematical definitions drive the ontology which mirrors the configuration — they are one system.

Type aliases referenced below are defined in [`type_aliases.py`](../skellymodels/type_aliases.py).

---

## Trajectory

**Module:** [`core/trajectory/trajectory.py`](../skellymodels/core/trajectory/trajectory.py)

A Trajectory is a named, validated tensor of shape `(F, M, D)` where F is frames, M is named keypoints, and D is dimensionality per keypoint.

**Fields:**
- `name: str`
- `keypoint_names: tuple[str, ...]` — immutable, ordered, unique
- `array: NDArray[np.float64]` — the `(F, M, D)` data tensor

**Access patterns:**

```python
trajectory.keypoints.right_shoulder  # dot notation (preferred)
trajectory["right_shoulder"]         # string indexing
trajectory.as_dict                   # {name: (F, D)} for all keypoints
trajectory.as_dataframe              # long-form [frame, keypoint, ...]
```

**Properties:** `num_frames`, `num_keypoints`, `num_dimensions`

**Validation:** array must be 3D, F ≥ 1, M ≥ 1, M matches keypoint count, no duplicate names.

### Subclasses

| Class | D | Columns | Additional methods |
|---|---|---|---|
| `SpatialTrajectory` | 3 | x, y, z | — |
| `QuaternionTrajectory` | 4 | w, x, y, z | `to_rotation_matrices()` → (F,M,3,3). Validates unit norm. |
| `AngularVelocityTrajectory` | 3 | — | `magnitude()` → (F,M). Units: rad/s |
| `AngularAccelerationTrajectory` | 3 | — | `magnitude()`. Units: rad/s² |

### Timeseries

**Module:** [`core/timeseries.py`](../skellymodels/core/timeseries.py)

Scalar quantity over time: `timestamps: (N,)`, `values: (N,)`. Methods: `differentiate()`, `interpolate()`.

---

## RigidBodyDefinition

**Module:** [`core/rigid_body/rigid_body_definition.py`](../skellymodels/core/rigid_body/rigid_body_definition.py)

A rigid body is a set of keypoints whose mutual distances are constant. It has a distinguished origin (the proximal end) and a coordinate frame. The amount of the frame directly observable from data depends on the available keypoints:

- **Fully constrained** (2 defined axes from 3+ non-colinear keypoints): 6 DoF. Swing and twist both observed.
- **Under-constrained** (1 defined axis from 2 keypoints): 5 DoF. Swing observed, twist computed as zero. See [Kinematics — Swing-only quaternions](kinematics.md#swing-only-quaternions).

**Fields:**
- `name: str`
- `keypoints: list[KeypointName]` (≥ 2)
- `origin: KeypointName` — proximal end, must be in `keypoints`
- `coordinate_frame: CoordinateFrameDefinition`

**Computed:** `is_fully_constrained: bool`, `default_distal_keypoint`

### CoordinateFrameDefinition

1 or 2 of `x_axis`, `y_axis`, `z_axis` defined. 1 axis = under-constrained (must be EXACT). 2 axes = fully constrained (one EXACT, one APPROXIMATE). Remaining axes computed at runtime via cross product (Gram-Schmidt). See [Kinematics — Coordinate frame construction](kinematics.md#coordinate-frame-construction).

`origin_keypoints` = frame center at runtime (may differ from `RigidBodyDefinition.origin`).

Convention: **+X forward, +Y left, +Z up. Right-handed only.**

### YAML

See [`configs/skeletons/human_body.yaml`](../skellymodels/configs/skeletons/human_body.yaml), [`configs/rigid_bodies/charuco_board_5x3.yaml`](../skellymodels/configs/rigid_bodies/charuco_board_5x3.yaml).

2-keypoint bodies omit `coordinate_frame` — the loader auto-generates a 1-axis frame:

```yaml
right_upper_arm:
  keypoints: [right_shoulder, right_elbow]
  origin: right_shoulder
  # coordinate_frame auto-generated: x_axis=[right_elbow], type=exact

skull:
  keypoints: [skull_origin_foramen_magnum, nose_tip, left_eye_inner, ...]
  origin: skull_origin_foramen_magnum
  coordinate_frame:
    origin_keypoints: [skull_origin_foramen_magnum]
    x_axis: { keypoints: [nose_tip], type: exact }        # +X forward
    y_axis: { keypoints: [left_eye_inner], type: approximate }  # +Y left
    # z_axis = X × Y = up ✓
```

---

## SkeletonDefinition

**Module:** [`skeleton/skeleton_definition.py`](../skellymodels/skeleton/skeleton_definition.py)

A complete articulated structure: rigid bodies + linkages + chains.

### LinkageDefinition

A joint connecting a parent RB to child RBs at a shared keypoint. `shared_keypoint` must exist on parent AND every child.

**Computed:** `is_branching` — multiple children.

### ChainDefinition

A linear kinematic path. At branching linkages, follows one child. Aligned with FABRIK conventions. See [Pipeline — IK solvers](pipeline.md#for-ik-solvers-fabrik).

**Validation:** root = parent of first linkage; consecutive linkages connected.

### SkeletonDefinition

**Dot-access:** `skeleton.rb.skull`, `skeleton.link.right_elbow_joint`, `skeleton.chain.axial`

**Derived properties:**
- `all_keypoint_names` — deduplicated union
- `segment_connections` — proximal/distal for under-constrained RBs
- `joint_hierarchy` — parent→children tree from linkage traversal
- `junction_keypoints` — keypoints shared between chains (FABRIK reconciliation)
- `get_chain_joint_sequence(name)` — ordered keypoints for IK solver

**Validation:** all cross-references resolve; shared keypoints on parent AND children; chains connected.

### YAML

See [`configs/skeletons/human_body.yaml`](../skellymodels/configs/skeletons/human_body.yaml):

```yaml
name: human_body
rigid_bodies:
  skull: { ... }
  right_upper_arm: { ... }
linkages:
  thoracic_t1:                                  # branching linkage
    parent_rigid_body: spine_thoracic
    child_rigid_bodies: [spine_cervical, right_clavicle, left_clavicle]
    shared_keypoint: spine_thoracic_top_t1
chains:
  axial:
    root_rigid_body: pelvis
    linkages: [sacrum_lumbar, lumbar_thoracic, thoracic_t1, cervical_skull]
```

---

## KeypointMapping

**Module:** [`mapping/keypoint_mapping.py`](../skellymodels/mapping/keypoint_mapping.py)

Translates tracker namespace → skeleton namespace. Three forms, inferred from YAML shape:

- `str` → 1:1 direct
- `list[str]` → equal-weight average
- `dict[str, float]` → weighted sum (weights can be any values including negative for extrapolation)

**Methods:**
- `apply(tracked_array, tracked_point_names)` — low-level, takes raw name list
- `apply_from_tracker(tracked_array, tracker_info, aspect_name)` — preferred, resolves names from TrackerModelInfo

### YAML

See [`configs/mappings/mediapipe_human_body.yaml`](../skellymodels/configs/mappings/mediapipe_human_body.yaml):

```yaml
tracker_name: mediapipe
skeleton_name: human_body
mappings:
  nose_tip: nose                                        # str → direct
  skull_origin_foramen_magnum: [left_ear, right_ear]    # list → average
  spine_cervical_top_c1_axis:                           # dict → weighted
    left_ear: 0.45
    right_ear: 0.45
    left_shoulder: 0.05
    right_shoulder: 0.05
```

---

## CoMDefinition

**Module:** [`biomechanics/com_definition.py`](../skellymodels/biomechanics/com_definition.py)

Anthropometric table: skeleton segments → mass distribution. Multiple configs can target the same skeleton.

**SegmentCoMParameters:** `rigid_body`, `com_length_ratio` (0=proximal, 1=distal), `mass_fraction`, `distal` (optional override, same polymorphism as KeypointMapping; defaults to `RigidBodyDefinition.default_distal_keypoint`).

**Validation:** mass fractions sum ≈ 1.0.

### YAML

See [`configs/center_of_mass/human_body_de_leva.yaml`](../skellymodels/configs/center_of_mass/human_body_de_leva.yaml).

---

## TrackerModelInfo

**Module:** [`models/tracking_model_info.py`](../skellymodels/models/tracking_model_info.py)

Metadata about a tracker: name, aspect names, tracked point names per aspect, order. Used by `KeypointMapping.apply_from_tracker()` to resolve point names without the user constructing raw string lists.

### YAML

See [`tracker_info/mediapipe_model_info.yaml`](../skellymodels/tracker_info/mediapipe_model_info.yaml).

---

## Naming conventions

- `lowercase_snake_case` throughout (PEP 8, COCO-compatible)
- **Keypoint names** (skeleton): `skull_origin_foramen_magnum`, `right_shoulder`
- **Tracked point names** (tracker): `left_ear`, `right_eye_inner`
- Conversion utilities (Blender's `shoulder.L` etc.) at the interface layer only
- Type aliases enforce namespace separation: `KeypointName` vs `TrackedPointName`
