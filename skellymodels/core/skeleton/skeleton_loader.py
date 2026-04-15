"""
YAML loaders for skeleton and rigid body definitions.

load_skeleton_from_yaml() reads a skeleton YAML file and returns
a validated SkeletonDefinition.

load_rigid_body_from_yaml() reads a standalone rigid body YAML file
(e.g. for a Charuco board that has no linkages or chains).
"""

from pathlib import Path

import yaml

from skellymodels.core.rigid_body.rigid_body_definition import (
    AxisDefinition,
    AxisType,
    CoordinateFrameDefinition,
    RigidBodyDefinition,
)
from skellymodels.core.skeleton.skeleton_definition import (
    ChainDefinition,
    LinkageDefinition,
    SkeletonDefinition,
)


def load_skeleton_from_yaml(path: str | Path) -> SkeletonDefinition:
    """Load and validate a skeleton definition from a YAML file."""
    path = Path(path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping at top level, got {type(raw).__name__}")

    # Parse rigid bodies
    rigid_bodies: dict[str, RigidBodyDefinition] = {}
    for rb_name, rb_data in raw.get("rigid_bodies", {}).items():
        rigid_bodies[rb_name] = _parse_rigid_body(name=rb_name, data=rb_data)

    # Parse linkages
    linkages: dict[str, LinkageDefinition] = {}
    for link_name, link_data in raw.get("linkages", {}).items():
        linkages[link_name] = LinkageDefinition(
            name=link_name,
            parent_rigid_body=link_data["parent_rigid_body"],
            child_rigid_bodies=link_data["child_rigid_bodies"],
            shared_keypoint=link_data["shared_keypoint"],
        )

    # Parse chains
    chains: dict[str, ChainDefinition] = {}
    for chain_name, chain_data in raw.get("chains", {}).items():
        chains[chain_name] = ChainDefinition(
            name=chain_name,
            root_rigid_body=chain_data["root_rigid_body"],
            linkages=chain_data["linkages"],
        )

    return SkeletonDefinition(
        name=raw["name"],
        rigid_bodies=rigid_bodies,
        linkages=linkages,
        chains=chains,
    )


def load_rigid_body_from_yaml(path: str | Path) -> RigidBodyDefinition:
    """Load and validate a standalone rigid body definition from a YAML file."""
    path = Path(path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping at top level, got {type(raw).__name__}")

    return _parse_rigid_body(name=raw["name"], data=raw)


def _parse_rigid_body(name: str, data: dict) -> RigidBodyDefinition:
    """Parse a single rigid body definition from a YAML dict.

    For 2-keypoint bodies without an explicit coordinate_frame, the frame
    is auto-generated: origin at the origin keypoint, x_axis (exact) pointing
    from origin toward the other keypoint.
    """
    keypoints: list[str] = data["keypoints"]
    origin: str = data["origin"]
    frame_data = data.get("coordinate_frame")

    if frame_data is not None:
        # Explicit coordinate frame in YAML
        axes: dict[str, AxisDefinition | None] = {}
        for axis_name in ("x_axis", "y_axis", "z_axis"):
            axis_data = frame_data.get(axis_name)
            if axis_data is not None:
                axes[axis_name] = AxisDefinition(
                    keypoints=axis_data["keypoints"],
                    type=AxisType(axis_data["type"]),
                )
            else:
                axes[axis_name] = None

        coordinate_frame = CoordinateFrameDefinition(
            origin_keypoints=frame_data["origin_keypoints"],
            x_axis=axes["x_axis"],
            y_axis=axes["y_axis"],
            z_axis=axes["z_axis"],
        )
    elif len(keypoints) == 2:
        # Auto-generate frame for 2-keypoint body:
        # x_axis (exact) points from origin toward the other keypoint
        distal = [kp for kp in keypoints if kp != origin]
        if len(distal) != 1:
            raise ValueError(
                f"Cannot auto-generate frame for '{name}': origin '{origin}' "
                f"not found in keypoints {keypoints}"
            )
        coordinate_frame = CoordinateFrameDefinition(
            origin_keypoints=[origin],
            x_axis=AxisDefinition(keypoints=distal, type=AxisType.EXACT),
        )
    else:
        raise ValueError(
            f"Rigid body '{name}' has {len(keypoints)} keypoints but no "
            f"coordinate_frame. Bodies with 3+ keypoints require an explicit "
            f"coordinate_frame definition."
        )

    return RigidBodyDefinition(
        name=name,
        keypoints=keypoints,
        origin=origin,
        coordinate_frame=coordinate_frame,
    )
