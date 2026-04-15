"""
Center of mass definition.

A CoMDefinition is an anthropometric table mapping skeleton segments to
mass distribution parameters. Multiple CoM configs can target the same
skeleton (different studies, populations, etc.).

The distal reference point for com_length_ratio defaults to the rigid body's
default_distal_keypoint but can be overridden using the same str/list/dict
polymorphism as KeypointMapping.
"""

from pydantic import BaseModel, ConfigDict, model_validator

from skellymodels.mapping.keypoint_mapping import MappingSource
from skellymodels.type_aliases import (
    ComLengthRatio,
    MassFraction,
    RigidBodyName,
    SegmentName,
    SkeletonName,
)


# Tolerance for mass fraction sum validation
_MASS_FRACTION_SUM_TOLERANCE = 0.02


class SegmentCoMParameters(BaseModel):
    """
    Center of mass parameters for a single segment.

    Fields:
        rigid_body: which RB this segment corresponds to
        distal: override for the distal reference point (None → use RB default)
        com_length_ratio: 0.0 = at origin (proximal), 1.0 = at distal
        mass_fraction: fraction of total body mass
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    rigid_body: RigidBodyName
    distal: MappingSource | None = None
    com_length_ratio: ComLengthRatio
    mass_fraction: MassFraction

    @model_validator(mode="after")
    def _validate_ranges(self) -> "SegmentCoMParameters":
        if not (0.0 <= self.com_length_ratio <= 1.0):
            raise ValueError(
                f"com_length_ratio must be in [0, 1], got {self.com_length_ratio}"
            )
        if self.mass_fraction < 0.0:
            raise ValueError(
                f"mass_fraction must be non-negative, got {self.mass_fraction}"
            )
        return self


class CoMDefinition(BaseModel):
    """
    Anthropometric table: skeleton segments → mass distribution parameters.

    Fields:
        skeleton_name: target skeleton this applies to
        source: citation (e.g. "De Leva 1996")
        segments: dict mapping segment name → CoM parameters
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    skeleton_name: SkeletonName
    source: str
    segments: dict[SegmentName, SegmentCoMParameters]

    @model_validator(mode="after")
    def _validate_mass_fractions(self) -> "CoMDefinition":
        if len(self.segments) == 0:
            raise ValueError("segments must contain at least one entry")

        total_mass = sum(seg.mass_fraction for seg in self.segments.values())
        if abs(total_mass - 1.0) > _MASS_FRACTION_SUM_TOLERANCE:
            raise ValueError(
                f"mass_fraction values must sum to approximately 1.0, "
                f"got {total_mass:.6f}"
            )
        return self
