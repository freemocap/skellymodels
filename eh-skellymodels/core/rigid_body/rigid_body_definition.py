"""
Rigid body definition types.

A RigidBodyDefinition is the abstract topological specification of a rigid body:
which keypoints exist, which is the origin (proximal end), and how the
coordinate frame axes are defined.

Every rigid body has a coordinate frame. The distinction between
fully-constrained and under-constrained is how much of the frame is
directly observable:

  Fully constrained (2 user-defined axes from 3+ non-colinear keypoints):
    All 6 DoF observable. Gram-Schmidt orthogonalization from the two defined axes.

  Under-constrained (1 user-defined axis from 2 keypoints):
    5 DoF observable. The primary axis (bone direction) is observed.
    Secondary and tertiary axes are computed via swing-only (zero twist).

The constraint level is inferred from the definition — never declared.

Coordinate convention (enforced globally):
  +X = forward
  +Y = left
  +Z = up
  Right-handed system only.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from skellymodels.type_aliases import KeypointName


class AxisType(str, Enum):
    """Whether an axis definition is exact or approximate."""
    EXACT = "exact"
    APPROXIMATE = "approximate"


class AxisDefinition(BaseModel):
    """
    Definition of a coordinate axis direction.

    The axis direction is computed at runtime as the vector from the coordinate
    frame origin to the mean position of the listed keypoints.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    keypoints: list[KeypointName]
    type: AxisType

    @field_validator("keypoints")
    @classmethod
    def _keypoints_not_empty(cls, v: list[KeypointName]) -> list[KeypointName]:
        if len(v) == 0:
            raise ValueError("keypoints list cannot be empty")
        return v


class CoordinateFrameDefinition(BaseModel):
    """
    Definition of the body-fixed coordinate frame.

    One or two of x_axis, y_axis, z_axis may be defined:

      1 axis defined (under-constrained): Must be EXACT. Defines the primary
        (bone) axis. Secondary and tertiary axes computed at runtime via
        swing-only quaternion (zero twist, equivalent to Blender's Damped Track).

      2 axes defined (fully constrained): One EXACT, one APPROXIMATE. The third
        axis computed via cross product. Gram-Schmidt orthogonalization ensures
        the exact axis is preserved exactly.

    origin_keypoints defines the coordinate frame's spatial center (mean of
    listed keypoints at runtime). This may differ from
    RigidBodyDefinition.origin, which is the structural proximal keypoint.

    Coordinate convention: +X forward, +Y left, +Z up. Right-handed only.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    origin_keypoints: list[KeypointName]
    x_axis: AxisDefinition | None = None
    y_axis: AxisDefinition | None = None
    z_axis: AxisDefinition | None = None

    @field_validator("origin_keypoints")
    @classmethod
    def _origin_keypoints_not_empty(cls, v: list[KeypointName]) -> list[KeypointName]:
        if len(v) == 0:
            raise ValueError("origin_keypoints list cannot be empty")
        return v

    @model_validator(mode="after")
    def _validate_axis_definitions(self) -> "CoordinateFrameDefinition":
        """Validate axis count and type constraints."""
        defined = self.defined_axes
        num_defined = len(defined)

        if num_defined == 0:
            raise ValueError("At least 1 axis must be defined")

        if num_defined > 2:
            raise ValueError(
                f"At most 2 axes may be defined, got {num_defined}: "
                f"{[name for name, _ in defined]}"
            )

        types = [axis_def.type for _, axis_def in defined]

        if num_defined == 1:
            if types[0] != AxisType.EXACT:
                raise ValueError(
                    "Under-constrained frame (1 axis defined) requires the axis "
                    "to be EXACT (the observable primary axis)"
                )

        elif num_defined == 2:
            if AxisType.EXACT not in types:
                raise ValueError("One axis must be marked as 'exact'")
            if AxisType.APPROXIMATE not in types:
                raise ValueError("One axis must be marked as 'approximate'")

        return self

    @property
    def num_defined_axes(self) -> int:
        """Number of user-defined axes (1 or 2)."""
        return len(self.defined_axes)

    @property
    def defined_axes(self) -> list[tuple[str, AxisDefinition]]:
        """List of (axis_name, axis_def) for all user-defined axes."""
        result: list[tuple[str, AxisDefinition]] = []
        for axis_name in ("x_axis", "y_axis", "z_axis"):
            axis_def = getattr(self, axis_name)
            if axis_def is not None:
                result.append((axis_name, axis_def))
        return result

    @property
    def exact_axis(self) -> tuple[str, AxisDefinition]:
        """Return (axis_name, axis_def) for the exact axis."""
        for axis_name, axis_def in self.defined_axes:
            if axis_def.type == AxisType.EXACT:
                return axis_name, axis_def
        raise ValueError("No exact axis found (should be caught by validation)")

    @property
    def approximate_axis(self) -> tuple[str, AxisDefinition] | None:
        """Return (axis_name, axis_def) for the approximate axis, or None if under-constrained."""
        for axis_name, axis_def in self.defined_axes:
            if axis_def.type == AxisType.APPROXIMATE:
                return axis_name, axis_def
        return None

    @property
    def computed_axis_names(self) -> list[str]:
        """Names of axes computed at runtime (1 for fully-constrained, 2 for under-constrained)."""
        defined = {name for name, _ in self.defined_axes}
        return [name for name in ("x_axis", "y_axis", "z_axis") if name not in defined]

    def all_referenced_keypoints(self) -> set[KeypointName]:
        """All keypoint names referenced in this frame definition."""
        result = set(self.origin_keypoints)
        for _, axis_def in self.defined_axes:
            result.update(axis_def.keypoints)
        return result


class RigidBodyDefinition(BaseModel):
    """
    Abstract topological specification of a rigid body.

    A rigid body is a set of keypoints whose mutual distances are constant.
    It has a distinguished origin keypoint (the proximal end by convention)
    and a coordinate frame that defines its orientation.

    Every rigid body has a coordinate frame. For 2-keypoint bodies, the frame
    is auto-generated by the YAML loader (1 exact axis from origin toward
    the distal keypoint). For 3+ keypoint bodies, the frame is explicitly
    defined in YAML with 2 axes.

    Constraint level is inferred:
      is_fully_constrained == True:  frame has 2 defined axes (6 DoF observable)
      is_fully_constrained == False: frame has 1 defined axis (5 DoF, swing-only)
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    keypoints: list[KeypointName]
    origin: KeypointName
    coordinate_frame: CoordinateFrameDefinition

    @model_validator(mode="after")
    def _validate_definition(self) -> "RigidBodyDefinition":
        if self.origin not in self.keypoints:
            raise ValueError(
                f"origin '{self.origin}' must be in keypoints. "
                f"Available: {self.keypoints}"
            )

        if len(self.keypoints) != len(set(self.keypoints)):
            seen: set[str] = set()
            dupes = [k for k in self.keypoints if k in seen or seen.add(k)]  # type: ignore[func-returns-value]
            raise ValueError(f"keypoints contains duplicates: {dupes}")

        if len(self.keypoints) < 2:
            raise ValueError(
                f"A rigid body requires at least 2 keypoints, "
                f"got {len(self.keypoints)}"
            )

        # All frame keypoint references must resolve
        keypoint_set = set(self.keypoints)
        referenced = self.coordinate_frame.all_referenced_keypoints()
        missing = referenced - keypoint_set
        if missing:
            raise ValueError(
                f"Coordinate frame references keypoints not in this "
                f"rigid body's keypoint list: {sorted(missing)}. "
                f"Available: {sorted(keypoint_set)}"
            )

        # Consistency: 2 keypoints can only have 1 defined axis
        if len(self.keypoints) == 2 and self.coordinate_frame.num_defined_axes != 1:
            raise ValueError(
                f"2-keypoint rigid body must have exactly 1 defined axis "
                f"(the primary bone axis), got {self.coordinate_frame.num_defined_axes}"
            )
        # 3+ keypoints may have 1 axis (under-constrained with extra tracking keypoints)
        # or 2 axes (fully constrained). Both are valid.

        return self

    @property
    def is_fully_constrained(self) -> bool:
        """True if this body has 2 defined axes (6 DoF observable from data)."""
        return self.coordinate_frame.num_defined_axes == 2

    @property
    def default_distal_keypoint(self) -> KeypointName | list[KeypointName]:
        """
        The keypoint(s) defining the "far end" of the body's primary axis.

        For 2-keypoint bodies: the non-origin keypoint.
        For fully-constrained bodies: the keypoint(s) from the exact axis definition.
        """
        _, exact_def = self.coordinate_frame.exact_axis
        if len(exact_def.keypoints) == 1:
            return exact_def.keypoints[0]
        return exact_def.keypoints
