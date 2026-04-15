# Configuration reference

All structural definitions in skellymodels are stored as YAML files and validated by Pydantic at load time. This document specifies the format for each config type.

## Skeleton definition

**Location:** `configs/skeletons/`
**Loader:** `load_skeleton_from_yaml(path) → SkeletonDefinition`

```yaml
name: human_body  # skeleton identifier

rigid_bodies:
  # Under-constrained (2 keypoints, no coordinate frame)
  right_upper_arm:
    keypoints: [right_shoulder, right_elbow]
    origin: right_shoulder
    # No coordinate_frame field → inferred as under-constrained

  # Fully constrained (3+ keypoints with coordinate frame)
  skull:
    keypoints:
      - skull_origin_foramen_magnum
      - nose_tip
      - right_eye_inner
    origin: skull_origin_foramen_magnum
    coordinate_frame:
      origin_keypoints: [skull_origin_foramen_magnum]
      x_axis:
        keypoints: [nose_tip]
        type: exact           # one axis must be exact
      y_axis:
        keypoints: [right_eye_inner]
        type: approximate     # one axis must be approximate
      # z_axis is omitted → computed via cross product

linkages:
  skull_c1:
    parent_rigid_body: spine_cervical
    child_rigid_bodies: [skull]           # list, can have multiple (branching)
    shared_keypoint: skull_origin_foramen_magnum  # must be on parent AND all children

chains:
  axial:
    root_rigid_body: pelvis
    linkages:                             # ordered proximal-to-distal
      - sacrum_lumbar
      - lumbar_thoracic
      - thoracic_t1
      - cervical_skull
```

### Rigid body rules

- Minimum 2 keypoints
- `origin` must be one of the listed keypoints
- If `coordinate_frame` is omitted: exactly 2 keypoints required
- If `coordinate_frame` is present: 3+ keypoints required, exactly 2 of 3 axes defined (one exact, one approximate), all referenced keypoints must exist in the body's keypoint list

### Coordinate frame rules

- `origin_keypoints`: 1+ keypoints whose mean defines the frame center at runtime
- Exactly 2 of `x_axis`, `y_axis`, `z_axis` must be defined
- One must have `type: exact`, the other `type: approximate`
- The third axis is computed via cross product for a right-handed system
- The exact axis direction is preserved exactly; the approximate axis is orthogonalized via Gram-Schmidt

### Linkage rules

- `shared_keypoint` must exist in the keypoint lists of both the parent RB and every child RB
- At least 1 child rigid body

### Chain rules

- `root_rigid_body` must be the parent of the first linkage
- Each consecutive pair of linkages must be connected: one of linkage_i's children must be linkage_(i+1)'s parent
- Branching linkages are allowed — the chain follows one specific branch

## Keypoint mapping

**Location:** `configs/mappings/`
**Loader:** `load_mapping_from_yaml(path) → KeypointMapping`

```yaml
tracker_name: mediapipe
skeleton_name: human_body

mappings:
  # Direct 1:1 — skeleton keypoint name: tracker point name
  nose_tip: nose
  right_shoulder: right_shoulder

  # Equal-weight average — skeleton keypoint: [tracker points]
  skull_origin_foramen_magnum:
    - left_ear
    - right_ear

  # Weighted sum — skeleton keypoint: {tracker point: weight, ...}
  # Weights MUST sum to 1.0
  spine_cervical_top_c1_axis:
    left_ear: 0.45
    right_ear: 0.45
    left_shoulder: 0.05
    right_shoulder: 0.05
```

The mapping type (direct, averaged, weighted) is inferred from the YAML shape — no explicit type field needed.

YAML automatically distinguishes these: a bare string is `str`, a YAML sequence is `list`, a YAML mapping with numeric values is `dict[str, float]`.

## Center of mass definition

**Location:** `configs/center_of_mass/`
**Loader:** `load_com_from_yaml(path) → CoMDefinition`

```yaml
skeleton_name: human_body
source: "De Leva 1996"    # citation for the anthropometric data

segments:
  head:
    rigid_body: skull                # must exist in the target skeleton
    com_length_ratio: 0.5            # 0.0 = at origin, 1.0 = at distal
    mass_fraction: 0.081             # fraction of total body mass
    # distal: omitted → uses RigidBodyDefinition.default_distal_keypoint

  right_upper_arm:
    rigid_body: right_upper_arm
    com_length_ratio: 0.436
    mass_fraction: 0.028

  # For bodies where the default distal isn't right:
  thorax_custom_example:
    rigid_body: thorax
    distal:                          # override with weighted point
      left_shoulder: 0.5
      right_shoulder: 0.5
    com_length_ratio: 0.5
    mass_fraction: 0.3
```

### Distal resolution

The `distal` field uses the same polymorphism as KeypointMapping:
- Omitted or `null` → use `RigidBodyDefinition.default_distal_keypoint`
- `str` → a single keypoint name
- `list[str]` → equal-weight average of keypoints
- `dict[str, float]` → weighted sum (must sum to 1.0)

### Validation

- All `mass_fraction` values must sum to approximately 1.0 (tolerance: 0.02)
- `com_length_ratio` must be in [0, 1]
- `mass_fraction` must be non-negative

## Standalone rigid body

**Location:** `configs/rigid_bodies/`
**Loader:** `load_rigid_body_from_yaml(path) → RigidBodyDefinition`

For objects that aren't part of a skeleton (e.g. calibration boards):

```yaml
name: charuco_board_5x3
keypoints:
  - corner_0
  - corner_1
  - corner_2
  - corner_3
  - corner_4
  - corner_5
  - corner_6
  - corner_7
origin: corner_0
coordinate_frame:
  origin_keypoints: [corner_0]
  x_axis:
    keypoints: [corner_3]
    type: exact
  y_axis:
    keypoints: [corner_4]
    type: approximate
```

## Tracker info (trimmed)

**Location:** `tracker_info/`
**Loader:** `load_tracker_info_from_yaml(path) → TrackerModelInfo`

Stripped to essentials — no virtual markers, no CoM, no segments, no hierarchy:

```yaml
name: mediapipe
tracker_name: MediapipeHolisticTracker

order:
  - body
  - right_hand
  - left_hand
  - face

aspects:
  body:
    tracked_points:
      type: list
      names:
        - nose
        - left_eye_inner
        # ... all 33 mediapipe body points

  face:
    tracked_points:
      type: generated
      names:
        convention: "face_{:04d}"
        count: 478
```
