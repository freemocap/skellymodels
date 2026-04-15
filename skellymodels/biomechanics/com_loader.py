"""YAML loader for center of mass definitions."""

from pathlib import Path

import yaml

from skellymodels.biomechanics.com_definition import (
    CoMDefinition,
    SegmentCoMParameters,
)


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
