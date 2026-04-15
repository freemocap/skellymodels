# Pipeline — End-to-end data flow

## Overview

```
Raw tracker data (F, 33, 3)
       │
       ▼
KeypointMapping.apply_from_tracker()
       │
       ▼
SpatialTrajectory (F, 43, 3)
       │
       ├───────────────────────────┐
       ▼                           ▼
calculate_center_of_mass()   enforce_rigid_bones()
       │                           │
       ▼                           ▼
CoM trajectories          Rigidified trajectory
```

Every step is an explicit function call with named arguments. No hidden state.

## Step 1: Load definitions

```python
from skellymodels.core.skeleton import load_skeleton_from_yaml
from skellymodels.core.mapping import load_mapping_from_yaml
from skellymodels.core.models.tracking_model_info import load_tracker_info_from_yaml
from skellymodels.core.biomechanics import load_com_from_yaml

skeleton = load_skeleton_from_yaml("skellymodels/configs/skeletons/human_body.yaml")
mapping = load_mapping_from_yaml("skellymodels/configs/mappings/mediapipe_human_body.yaml")
tracker_info = load_tracker_info_from_yaml("skellymodels/tracker_info/mediapipe_model_info.yaml")
com_def = load_com_from_yaml("skellymodels/configs/center_of_mass/human_body_de_leva.yaml")
```

Immutable Pydantic objects. Load once, reuse for every recording. Pydantic validates all cross-references at load time.

## Step 2: Apply mapping

```python
mapped_array = mapping.apply_from_tracker(
    tracked_array=tracker_data,     # (num_frames, 33, 3) from mediapipe
    tracker_info=tracker_info,
    aspect_name="body",
)
```

`apply_from_tracker` resolves tracked point names from the TrackerModelInfo. For each skeleton keypoint, it applies the configured combination (direct copy, mean, or weighted sum). Output shape: `(num_frames, num_skeleton_keypoints, 3)`.

For cases where you need manual control, the low-level `apply()` method accepts an explicit name list:

```python
mapped_array = mapping.apply(
    tracked_array=tracker_data,
    tracked_point_names=["nose", "left_eye_inner", ...],
)
```

## Step 3: Create trajectory

```python
from skellymodels.core.trajectory import SpatialTrajectory

trajectory = SpatialTrajectory(
    name="3d_xyz",
    keypoint_names=tuple(mapping.skeleton_keypoint_names),
    array=mapped_array,
)
```

Pydantic validates shape, dimension count, name uniqueness. Access via dot notation:

```python
trajectory.keypoints.right_shoulder       # (F, 3)
trajectory.keypoints.skull_origin_foramen_magnum
```

## Step 4: Biomechanics

### Center of mass

```python

from skellymodels.core.biomechanics.calculate_com import calculate_center_of_mass

total_com, segment_com = calculate_center_of_mass(
    trajectory=trajectory,
    skeleton=skeleton,
    com_definition=com_def,
)
```

For each CoM segment: looks up proximal/distal from `skeleton.segment_connections`, computes `proximal + (distal - proximal) * com_length_ratio`. Total body CoM = weighted sum of segment CoMs by mass fraction.

### Rigid bone enforcement

```python
from skellymodels.core.biomechanics.enforce_rigid_bones import enforce_rigid_bones

rigid_trajectory = enforce_rigid_bones(trajectory=trajectory, skeleton=skeleton)
```

Uses `skeleton.joint_hierarchy` (derived from linkages). Finds roots, recursively enforces median bone lengths. Output: `SpatialTrajectory` with constant bone lengths.

Typical order: rigid bones first, then CoM on the rigidified data.

## Step 5: Kinematics

See [Kinematics](kinematics.md) for full details.

```python
from skellymodels.core.kinematics import compute_basis_from_definition

# Works for any rigid body — fully constrained or under-constrained
basis, origin = compute_basis_from_definition(
    rigid_body=skeleton.rb.skull,
    keypoint_positions={
        name: trajectory.keypoints.__getattr__(name)[frame_idx]
        for name in skeleton.rb.skull.keypoints
    },
)
```

## For IK solvers (FABRIK)

The skeleton provides everything FABRIK needs:

```python
# Linear chain the solver operates on
joint_seq = skeleton.get_chain_joint_sequence("right_arm")
# ["spine_thoracic_top_t1", "right_shoulder", "right_elbow", "right_wrist"]

# Junction points for reconciliation between chains
junctions = skeleton.junction_keypoints
# {"spine_thoracic_top_t1", ...}
```

FABRIK solves each chain independently, then reconciles at junction points by averaging shared joint positions.

## Error handling

Every function raises exceptions on invalid input:

- Wrong array shape → `ValueError`
- Missing keypoint → `KeyError`
- Disconnected chain → `ValueError`
- Missing cross-reference → `ValueError`

No silent fallbacks. No default values that mask errors. If data makes it through the pipeline without errors, it's structurally correct.
