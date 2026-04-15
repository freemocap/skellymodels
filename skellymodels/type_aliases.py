"""
Type aliases for strict typing throughout skellymodels.

These aliases make type signatures self-documenting and enable
beartype to catch type confusion at runtime. Use these instead
of bare `str`, `int`, `float` for domain-specific values.

With `beartype_this_package()` in __init__.py, all function signatures
using these types get O(1) runtime type checking for free.
"""

# --- String identifiers ---

KeypointName = str
"""Name of an anatomical keypoint on a rigid body (e.g. 'right_shoulder', 'nose_tip')."""

TrackedPointName = str
"""Name of a point tracked by a motion capture system (e.g. 'left_ear', 'nose')."""

SkeletonName = str
"""Identifier for a skeleton definition (e.g. 'human_body')."""

TrackerName = str
"""Identifier for a tracking system (e.g. 'mediapipe', 'rtmpose')."""

RigidBodyName = str
"""Name of a rigid body within a skeleton (e.g. 'skull', 'right_upper_arm')."""

LinkageName = str
"""Name of a linkage (joint) within a skeleton (e.g. 'right_elbow_joint')."""

ChainName = str
"""Name of a kinematic chain within a skeleton (e.g. 'right_arm', 'axial')."""

SegmentName = str
"""Name of a biomechanical segment for CoM calculation (e.g. 'right_thigh')."""

AxisName = str
"""Name of a coordinate axis ('x_axis', 'y_axis', 'z_axis')."""

# --- Numeric types ---

MassFraction = float
"""Fraction of total body mass attributed to a segment (0.0 to 1.0)."""

ComLengthRatio = float
"""Ratio along a segment from proximal (0.0) to distal (1.0) for center of mass location."""

Weight = float
"""Weight for a weighted keypoint mapping combination."""
