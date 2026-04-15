"""
Center of mass definition.

A CoMDefinition is an anthropometric table mapping skeleton segments to
mass distribution parameters. Multiple CoM configs can target the same
skeleton (different studies, populations, etc.).

The distal reference point for com_length_ratio defaults to the rigid body's
default_distal_keypoint but can be overridden using the same str/list/dict
polymorphism as KeypointMapping.
"""
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from skellymodels.core.mapping.keypoint_mapping import MappingSource
from skellymodels.type_aliases import (
    ComLengthRatio,
    MassFraction,
    RigidBodyName,
    SegmentName,
    SkeletonName,
)
from skellymodels.type_overloads import ComSourceString


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
    source: ComSourceString
    segments: dict[SegmentName, SegmentCoMParameters]

    @model_validator(mode="after")
    def _validate_mass_fractions(self) -> "CoMDefinition":
        if len(self.segments) == 0:
            raise ValueError("segments must contain at least one entry")

        total_mass = sum(seg.mass_fraction for seg in self.segments.values())
        if not np.isclose(total_mass, 1.0):
            raise ValueError(
                f"mass_fraction values must sum to approximately 1.0, "
                f"got {total_mass:.6f}"
            )
        return self

def load_com_from_yaml(path: str | Path) -> CoMDefinition:
    """Load and validate a center of mass definition from a YAML file."""
    path = Path(path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Expected a YAML mapping at top level, got {type(raw).__name__}"
        )

    segments: dict[str, SegmentCoMParameters] = {}
    for seg_name, seg_data in raw.get("segments", {}).items():
        segments[seg_name] = SegmentCoMParameters(
            rigid_body=seg_data["rigid_body"],
            distal=seg_data.get("distal"),
            com_length_ratio=seg_data["com_length_ratio"],
            mass_fraction=seg_data["mass_fraction"],
        )

    return CoMDefinition(
        skeleton_name=raw["skeleton_name"],
        source=raw["source"],
        segments=segments,
    )
