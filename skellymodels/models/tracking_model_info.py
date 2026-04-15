"""
Tracking model info — metadata about a motion capture tracker.

Stores the tracker's identity, the aspects it produces (body, face, hands),
and the tracked point names for each aspect. Used by
KeypointMapping.apply_from_tracker() to resolve point names without the
user constructing raw string lists.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from skellymodels.type_aliases import TrackedPointName, TrackerName


class AspectInfo(BaseModel):
    """Tracked point names for a single aspect (e.g. body, face, hand)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tracked_point_names: list[TrackedPointName]


class TrackerModelInfo(BaseModel):
    """
    Metadata about a motion capture tracker.

    Stores: tracker identity, aspect names, tracked point names per aspect, order.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    tracker_name: TrackerName
    order: list[str]
    aspects: dict[str, AspectInfo]


def load_tracker_info_from_yaml(path: str | Path) -> TrackerModelInfo:
    """Load tracker info from a YAML file."""
    path = Path(path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    aspects: dict[str, AspectInfo] = {}
    for aspect_name, aspect_data in raw.get("aspects", {}).items():
        tracked_points = aspect_data.get("tracked_points", {})
        names = tracked_points.get("names", [])

        # Handle generated names (e.g. face_{:04d} with count)
        if tracked_points.get("type") == "generated":
            convention = names.get("convention", "point_{}")
            count = names.get("count", 0)
            names = [convention.format(i) for i in range(count)]

        aspects[aspect_name] = AspectInfo(tracked_point_names=names)

    return TrackerModelInfo(
        name=raw["name"],
        tracker_name=raw["tracker_name"],
        order=raw.get("order", []),
        aspects=aspects,
    )
