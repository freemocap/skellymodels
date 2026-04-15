# skellymodels

Mathematically rigorous skeleton, rigid body, and trajectory definitions for motion capture pipelines.

`skellymodels` provides a validated type system for describing articulated structures (human bodies, animals, robotic arms, calibration objects) and the motion data captured from them. Every definition is formally specified, validated at construction time, and enforced at every layer of the pipeline.

## Installation

```bash
pip install -e .           # core
pip install -e ".[dev]"    # + pytest
pip install -e ".[bvh]"   # + scipy for BVH export
```

**Dependencies:** numpy, pydantic (v2), pandas, pyyaml, beartype.

## Quick start

```python
from skellymodels.skeleton import load_skeleton_from_yaml
from skellymodels.mapping import load_mapping_from_yaml
from skellymodels.models.tracking_model_info import load_tracker_info_from_yaml
from skellymodels.core.trajectory import SpatialTrajectory

# Load definitions
skeleton = load_skeleton_from_yaml("skellymodels/configs/skeletons/human_body.yaml")
mapping = load_mapping_from_yaml("skellymodels/configs/mappings/mediapipe_human_body.yaml")
tracker_info = load_tracker_info_from_yaml("skellymodels/tracker_info/mediapipe_model_info.yaml")

# Apply mapping — tracker_info handles the point names, no raw strings needed
mapped_array = mapping.apply_from_tracker(
    tracked_array=tracker_data,   # (num_frames, 33, 3) from mediapipe
    tracker_info=tracker_info,
    aspect_name="body",
)

# Create a typed trajectory
trajectory = SpatialTrajectory(
    name="3d_xyz",
    keypoint_names=tuple(mapping.skeleton_keypoint_names),
    array=mapped_array,
)

# Access data via dot notation — no raw strings
right_shoulder = trajectory.keypoints.right_shoulder   # (num_frames, 3)
skull_origin = trajectory.keypoints.skull_origin_foramen_magnum

# Or by string indexing when you need it
right_shoulder = trajectory["right_shoulder"]

# Skeleton properties — all derived from the definition
skull_def = skeleton.rb.skull             # RigidBodyDefinition via dot access
hierarchy = skeleton.joint_hierarchy      # parent → children keypoint tree
junctions = skeleton.junction_keypoints   # FABRIK reconciliation points

# Biomechanics
from skellymodels.biomechanics.com_loader import load_com_from_yaml
from skellymodels.biomechanics.pipeline import calculate_center_of_mass

com_def = load_com_from_yaml("skellymodels/configs/center_of_mass/human_body_de_leva.yaml")
total_com, segment_com = calculate_center_of_mass(trajectory, skeleton, com_def)
```

## Key concepts

**Type vs Token** — The system separates abstract definitions (what a skeleton IS) from data-filled instances (what a skeleton is DOING right now). Definitions are immutable YAML + Pydantic. Instances are Trajectory objects filled with measured data. See [Architecture](docs/architecture.md).

**Rigid bodies** — The atomic structural unit. Every rigid body has a coordinate frame. The amount of the frame that's directly observable depends on the keypoint count: 3+ non-colinear keypoints with 2 defined axes = fully constrained (6 DoF). 2 keypoints with 1 axis = under-constrained (5 DoF, swing-only for the unobservable twist). See [Ontology](docs/ontology.md#rigidbodydefinition).

**Additive hierarchy** — RigidBody → Linkage → Chain → Skeleton. Each level is a valid stopping point. A Charuco board is a standalone RigidBody. See [Ontology](docs/ontology.md#skeletondefinition).

**Coordinate convention** — +X forward, +Y left, +Z up. Right-handed only. Left-handed coordinate systems are illegal in skellymodels.

**Quaternion orientation** — All orientations are represented as unit quaternions (wxyz order) to avoid gimbal lock and polar singularities. Conversion methods to Euler angles, rotation matrices, and axis-angle are available, but the canonical representation is always quaternion. See [Kinematics](docs/kinematics.md).

## Project structure

```
skellymodels/
├── core/                    # Foundation types
│   ├── trajectory/          # Trajectory, SpatialTrajectory, QuaternionTrajectory
│   ├── rigid_body/          # RigidBodyDefinition, CoordinateFrameDefinition
│   ├── dot_access.py        # DotAccessDict for attribute-style access
│   └── timeseries.py        # Scalar time-varying data
├── skeleton/                # SkeletonDefinition, LinkageDefinition, ChainDefinition
├── mapping/                 # KeypointMapping with apply() and apply_from_tracker()
├── biomechanics/            # CoMDefinition, center of mass, rigid bone enforcement
├── kinematics/              # Basis computation, swing-only quaternions
├── configs/                 # YAML definitions (skeletons, mappings, CoM, rigid bodies)
├── models/                  # Aspect (data container), TrackerModelInfo
├── type_aliases.py          # Strict string/numeric type aliases
└── tests/                   # ~140 tests across 8 files
```

## Running tests

```bash
pip install -e ".[dev]"
pytest skellymodels/tests/ -v
```

## Documentation

- [Architecture](docs/architecture.md) — design principles, data flow
- [Ontology](docs/ontology.md) — formal type definitions + YAML configuration reference
- [Kinematics](docs/kinematics.md) — coordinate frames, swing-only quaternions, basis computation
- [Pipeline](docs/pipeline.md) — end-to-end data flow from tracker to biomechanics
- [Migration guide](docs/migration.md) — changes from previous skellymodels structure
