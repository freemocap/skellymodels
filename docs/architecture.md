# Architecture

## Design principles

### 1. Mathematical rigor

Every structural type has a complete, formal mathematical definition. These definitions are validated at every layer of the pipeline. If a runtime check finds data inconsistent with a definition, that's a **thrown error**, not a warning. Defensive checks are everywhere; silent tolerance is nowhere.

`beartype_this_package()` in `__init__.py` provides O(1) runtime type checking on all function signatures via [beartype](https://beartype.readthedocs.io/). Combined with Pydantic validation on models and strict [type aliases](../skellymodels/type_aliases.py), type confusion is caught immediately.

### 2. Type vs Token (Definition vs Instance)

Two distinct layers run through the entire system:

**Definition (Type):** The abstract specification. "A human skeleton has a skull rigid body with these keypoints and these axis relationships." No data, no positions, no timestamps. Lives in YAML configs and Pydantic models. Immutable.

**Instance (Token):** A specific realization filled with measured data. "At frame 42, the skull is at position [1.2, 3.4, 5.6] with orientation q=[0.9, 0.1, 0.2, 0.3]." Lives in Trajectory objects and kinematics models.

The bridge between them is explicit: `RigidBodyDefinition + measured positions → basis vectors + origin` via the [kinematics bridge](docs/kinematics.md#the-typetok-bridge).

### 3. Inferred, not declared

Properties that are mathematically determined by the definition are computed, not configured:

- A rigid body with 2 defined axes IS fully constrained — not declared as such
- A linkage with one child IS non-branching — not declared as such
- A keypoint in multiple chains' joint sequences IS a junction — not labeled as such

### 4. Additive hierarchy

```
RigidBody → Linkage → Chain → Skeleton
```

Each level is a valid stopping point. A Charuco board is a standalone RigidBody. An articulated arm stops at Chain. A full human body uses Skeleton.

### 5. Coordinate convention

Enforced globally: **+X forward, +Y left, +Z up. Right-handed only.** Left-handed coordinate systems are illegal in skellymodels. All coordinate frame definitions, basis computations, and cross products enforce this.

### 6. Quaternion orientation

All orientations use unit quaternions in **wxyz** order (scalar-first). Quaternions are the canonical representation because they avoid gimbal lock and polar singularities. Conversion to Euler angles, rotation matrices, and axis-angle is available but the internal representation is always quaternion.

## Layer dependencies

```
Layer 0  │  Trajectory, Timeseries           (no internal deps)
Layer 1  │  RigidBodyDefinition              (no internal deps)
Layer 2  │  SkeletonDefinition               (depends on Layer 1)
Layer 3  │  KeypointMapping                  (no internal deps)
Layer 4  │  CoMDefinition                    (depends on Layer 3 for MappingSource type)
Layer 5  │  Aspect, Pipeline                 (depends on Layers 0-4)
Layer 6  │  Kinematics bridge                (depends on Layer 1)
```

Each layer depends only on layers below it. You can use core types without pulling in the biomechanics pipeline or kinematics bridge.

## Data flow

```
Raw tracker data (F, 33, 3)
       │
       ▼
KeypointMapping.apply_from_tracker()   ← mapping YAML + tracker info YAML
       │
       ▼
Mapped array (F, 43, 3)
       │
       ▼
SpatialTrajectory                      ← keypoint_names from mapping
       │
       ├──────────────────────────────────────────┐
       ▼                                          ▼
calculate_center_of_mass()              enforce_rigid_bones()
  ← SkeletonDefinition                   ← SkeletonDefinition
  ← CoMDefinition                          (uses joint_hierarchy)
       │                                          │
       ▼                                          ▼
total_body_com (F, 1, 3)            rigid_trajectory (F, 43, 3)
segment_com (F, 14, 3)
```

For kinematics:

```
SpatialTrajectory + RigidBodyDefinition
       │
       ├── Fully constrained (2 defined axes):
       │     compute_basis_from_definition()
       │     → (3,3) orthonormal basis per frame
       │
       └── Under-constrained (1 defined axis):
             compute_basis_from_definition()
             → (3,3) basis with swing-only secondary axes
             → or compute_swing_quaternion_trajectory()
             → (F, 4) wxyz quaternions
```
