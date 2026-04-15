"""
YAML loader for keypoint mappings.

The YAML format auto-detects mapping type from shape:
  skeleton_kp: tracker_kp          → str (1:1)
  skeleton_kp: [tp1, tp2]          → list (equal weight)
  skeleton_kp: {tp1: 0.5, tp2: 0.5} → dict (weighted)
"""

from pathlib import Path

import yaml

from skellymodels.core.mapping.keypoint_mapping import KeypointMapping


def load_mapping_from_yaml(path: str | Path) -> KeypointMapping:
    """Load and validate a keypoint mapping from a YAML file."""
    path = Path(path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Expected a YAML mapping at top level, got {type(raw).__name__}"
        )

    return KeypointMapping(
        tracker_name=raw["tracker_name"],
        skeleton_name=raw["skeleton_name"],
        mappings=raw["mappings"],
    )
