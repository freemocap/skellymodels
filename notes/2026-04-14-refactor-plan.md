skellymodels/
│
├── core/
│   ├── __init__.py
│   │
│   ├── trajectory/
│   │   ├── __init__.py
│   │   ├── trajectory.py               # Trajectory: (F, M, D) ndarray + named markers
│   │   │                                #   D=3 for spatial, D=4 for quaternion, etc.
│   │   │                                #   Methods: as_array, as_dict, as_dataframe,
│   │   │                                #   __getitem__(name), slice_frames(), num_frames, etc.
│   │   └── typed_trajectories.py        # QuaternionTrajectory (F, M, 4), validated wxyz
│   │                                    # AngularVelocityTrajectory (F, M, 3), rad/s
│   │                                    # AngularAccelerationTrajectory (F, M, 3), rad/s²
│   │                                    # Inherit from Trajectory or compose with it
│   │
│   ├── timeseries.py                    # Timeseries: scalar (F,) + timestamps
│   │                                    #   (migrated from kinematics/timeseries_model.py)
│   │
│   └── rigid_body/
│       ├── __init__.py
│       └── rigid_body_definition.py     # RigidBodyDefinition (Pydantic, frozen):
│                                        #   name, keypoints: list[str], origin: str
│                                        #   coordinate_frame: CoordinateFrameDefinition | None
│                                        #   Properties: is_fully_constrained, distal_keypoint,
│                                        #               primary_axis_keypoints
│                                        #
│                                        # CoordinateFrameDefinition (migrated from kinematics):
│                                        #   origin_keypoints, x/y/z_axis (exactly 2 of 3),
│                                        #   one exact + one approximate
│                                        #
│                                        # AxisDefinition: keypoints + AxisType(EXACT|APPROXIMATE)
│                                        #
│                                        # UnderconstrainedError: raised when 6-DoF ops
│                                        #   are called on a 2-keypoint body
│
├── skeleton/
│   ├── __init__.py
│   ├── skeleton_definition.py           # LinkageDefinition: parent_rigid_body, child_rigid_bodies,
│   │                                    #   shared_keypoint. Inferred: is_branching (len(children) > 1)
│   │                                    #
│   │                                    # ChainDefinition: ordered list of linkage names,
│   │                                    #   root_rigid_body. Validation: all linkages must be
│   │                                    #   non-branching (1:1)
│   │                                    #
│   │                                    # SkeletonDefinition: name, rigid_bodies, linkages, chains
│   │                                    #   Derived property: joint_hierarchy (computed by traversal)
│   │                                    #   Validation: all cross-references resolve, shared keypoints
│   │                                    #   exist on both parent and child RBs
│   │
│   └── skeleton_loader.py              # load_skeleton_from_yaml(path) → SkeletonDefinition
│
├── mapping/
│   ├── __init__.py
│   ├── keypoint_mapping.py              # KeypointMapping: tracker_name, skeleton_name,
│   │                                    #   mappings: dict[str, str | list[str] | dict[str, float]]
│   │                                    #   Method: apply(tracked_array, tracked_names) → dict[str, NDArray]
│   │                                    #   Pydantic infers mapping type from shape of sources
│   │
│   └── mapping_loader.py               # load_mapping_from_yaml(path) → KeypointMapping
│
├── configs/
│   ├── skeletons/
│   │   └── human_body.yaml             # RigidBodies + Linkages + Chains (NO CoM)
│   │
│   ├── center_of_mass/
│   │   └── human_body_de_leva.yaml     # CoM per segment: com_length_ratio, mass_fraction
│   │                                    # source: "De Leva 1996"
│   │                                    # Multiple configs can target the same skeleton
│   │
│   ├── rigid_bodies/
│   │   ├── charuco_board_5x3.yaml      # Standalone RB, no skeleton hierarchy
│   │   └── charuco_board_7x5.yaml
│   │
│   └── mappings/
│       ├── mediapipe_human_body.yaml    # mediapipe tracked points → human_body keypoints
│       └── rtmpose_human_body.yaml
│
├── tracker_info/                        # TRIMMED: tracked point names only
│   ├── mediapipe_model_info.yaml        #   name, tracker_name, order, aspects[].tracked_point_names
│   └── rtmpose_model_info.yaml          #   NO virtual markers, NO CoM, NO segments, NO hierarchy
│
├── kinematics/                          # KEEP, targeted refactor
│   ├── __init__.py
│   ├── reference_geometry.py            # ReferenceGeometry (runtime, with positions + units)
│   │                                    #   Takes a RigidBodyDefinition + measured positions
│   │                                    #   Computes basis vectors, handles Gram-Schmidt
│   │                                    #   Existing code is already close to correct
│   │
│   ├── rigid_body_kinematics.py         # RigidBodyKinematics: takes ReferenceGeometry + motion data
│   │                                    #   Computes quaternion trajectories, velocities, etc.
│   │                                    #   Uses core/trajectory types for output
│   │
│   ├── quaternion_model.py              # Quaternion (single frame) — keep as-is
│   ├── quaternion_trajectory.py         # QuaternionTrajectory — refactor to use core Trajectory base
│   ├── derivative_helpers.py            # Keep as-is
│   ├── kinematics_serialization.py      # Keep as-is
│   └── rigid_body_state.py              # StaticPose, etc. — keep, update imports
│
├── biomechanics/                        # REFACTOR: decouple from AnatomicalStructure
│   ├── __init__.py
│   ├── center_of_mass.py                # CoM calculation functions
│   │                                    #   Takes CoMDefinition + trajectory segment data
│   ├── rigid_bone_enforcement.py        # Takes joint_hierarchy (derived from skeleton)
│   └── pipeline.py                      # Pipeline takes Aspect + SkeletonDefinition + CoMDefinition
│                                        #   No longer reads CoM from tracker YAML
│
├── models/                              # REFACTOR: use new definition types
│   ├── __init__.py
│   ├── actor/
│   │   ├── actor_abc.py                 # Base actor — holds aspects, skeleton ref, mapping ref
│   │   ├── human_actor.py
│   │   ├── animal_actor.py
│   │   └── charuco_actor.py             # RigidBody-level only (no skeleton)
│   │
│   ├── aspect.py                        # Aspect: name, trajectories, reprojection_error, metadata
│   │                                    #   No more AnatomicalStructure field
│   │                                    #   Mapping applied externally before trajectory creation
│   │
│   └── tracking_model_info.py           # Simplified: tracker name, aspect names,
│                                        #   tracked point names per aspect, order
│
├── bvh_exporter/                        # KEEP as-is, update imports
│   ├── __init__.py
│   └── bvh_exporter.py
│
└── tests/
    ├── __init__.py
    ├── conftest.py                      # Shared fixtures: synthetic (F, M, 3) arrays,
    │                                    #   sample skeleton definitions, sample mappings
    ├── test_trajectory.py               # Trajectory construction, shape validation,
    │                                    #   as_dict, as_dataframe, __getitem__, slice
    ├── test_rigid_body_definition.py    # Under-constrained vs fully-constrained inference,
    │                                    #   is_fully_constrained, UnderconstrainedError on
    │                                    #   6-DoF ops, validation of axis definitions
    ├── test_skeleton_definition.py      # Load from YAML, cross-ref validation,
    │                                    #   derived joint_hierarchy correctness
    ├── test_keypoint_mapping.py         # str/list/dict mapping, apply() output shapes,
    │                                    #   weight validation, missing point errors
    ├── test_com_definition.py           # Load from YAML, segment name cross-ref,
    │                                    #   mass fraction sum ~= 1.0
    ├── test_biomechanics_pipeline.py    # Synthetic data through CoM + rigid bone pipeline,
    │                                    #   output shapes, trajectory naming
    └── test_end_to_end.py              # Load tracker YAML → load mapping → apply mapping →
                                        #   build trajectory → run pipeline → verify output
                                        #   One test per tracker: mediapipe, rtmpose, charuco
