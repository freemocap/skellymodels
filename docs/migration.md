# Migration guide

This document records architectural changes for developers migrating from the previous skellymodels structure.

## Key changes

**AnatomicalStructure removed.** Skeleton structure, keypoint mappings, and center of mass definitions are now three separate concerns with dedicated types and YAML configs.

**Virtual markers replaced by KeypointMapping.** What were called "virtual markers" (weighted combinations of tracked points) are now explicit mappings from tracker namespace to skeleton namespace, stored in their own YAML files.

**CoM separated from tracker config.** Center of mass definitions (anthropometric data) live in dedicated YAML files, not inside tracker info YAMLs. The biomechanics pipeline takes a CoMDefinition explicitly.

**joint_hierarchy and segment_connections derived.** These are computed from the skeleton's linkage structure, not hand-written in YAML.

**Trajectory decoupled from anatomy.** Trajectory is a pure data container. It does not compute virtual markers or know about anatomy. Mapping is applied externally before trajectory construction.

**Coordinate frame on every rigid body.** Under-constrained bodies (2 keypoints) have an auto-generated 1-axis coordinate frame. The secondary/tertiary axes are computed at runtime via swing-only (zero twist). There is no UnderconstrainedError — under-constrained bodies return valid orientations.

**Coordinate convention:** +X forward, +Y left, +Z up. Right-handed only.

**beartype enabled package-wide** via `beartype_this_package()` in `__init__.py`.

## File mapping

| Previous | Current |
|---|---|
| `from_blender/` | `core/rigid_body/`, `skeleton/`, `mapping/` |
| `models/anatomical_structure.py` | `skeleton/skeleton_definition.py` + `mapping/keypoint_mapping.py` + `biomechanics/com_definition.py` |
| `Trajectory.from_tracked_points_data()` | `KeypointMapping.apply()` + `SpatialTrajectory()` |
| `Trajectory.marker_names` | `Trajectory.keypoint_names` |
| `Aspect.anatomical_structure` | Removed; anatomy passed explicitly to pipeline |
