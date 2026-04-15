"""
Skeleton definition types.

A SkeletonDefinition is a complete articulated structure composed of named
rigid bodies connected by linkages, optionally organized into chains.

The hierarchy is additive: RigidBody → Linkage → Chain → Skeleton.
Each level is a valid exit point (e.g. a Charuco board is a standalone RB).

Key derived properties:
  - joint_hierarchy: parent→children keypoint tree (replaces hand-written YAML)
  - segment_connections: proximal/distal for 2-keypoint RBs
  - junction_keypoints: shared points between chains (for FABRIK reconciliation)
"""

from functools import cached_property

from pydantic import BaseModel, ConfigDict, model_validator

from skellymodels.core.dot_access import DotAccessDict
from skellymodels.core.rigid_body.rigid_body_definition import RigidBodyDefinition
from skellymodels.type_aliases import (
    ChainName,
    KeypointName,
    LinkageName,
    RigidBodyName,
    SkeletonName,
)


class LinkageDefinition(BaseModel):
    """
    A joint connecting a parent rigid body to one or more child rigid bodies
    at a shared keypoint.

    The shared keypoint must exist on both the parent and every child RB.
    Distances from the shared keypoint to other keypoints within each body
    are constant, but distances between keypoints in different bodies may vary.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: LinkageName
    parent_rigid_body: RigidBodyName
    child_rigid_bodies: list[RigidBodyName]
    shared_keypoint: KeypointName

    @model_validator(mode="after")
    def _validate_children_not_empty(self) -> "LinkageDefinition":
        if len(self.child_rigid_bodies) == 0:
            raise ValueError(
                f"Linkage '{self.name}' must have at least 1 child rigid body"
            )
        return self

    @property
    def is_branching(self) -> bool:
        """True if this linkage connects to more than one child."""
        return len(self.child_rigid_bodies) > 1


class ChainDefinition(BaseModel):
    """
    A linear kinematic path through the skeleton graph.

    At branching linkages, the chain follows exactly one child — the one
    connecting to the next linkage in the sequence. Other children are branches
    (potentially roots of other chains, resolved at the skeleton level).

    Aligned with IK solver conventions: FABRIK operates on linear chains,
    then reconciles at junction joints where chains share keypoints.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: ChainName
    root_rigid_body: RigidBodyName
    linkages: list[LinkageName]

    @model_validator(mode="after")
    def _validate_linkages_not_empty(self) -> "ChainDefinition":
        if len(self.linkages) == 0:
            raise ValueError(
                f"Chain '{self.name}' must have at least 1 linkage"
            )
        return self


class SkeletonDefinition(BaseModel):
    """
    A complete articulated structure: named rigid bodies connected by
    linkages, optionally organized into chains.

    All cross-references are validated at construction time. Derived properties
    (joint_hierarchy, segment_connections, junction_keypoints) are computed
    from the definition and cached.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: SkeletonName
    rigid_bodies: dict[RigidBodyName, RigidBodyDefinition]
    linkages: dict[LinkageName, LinkageDefinition]
    chains: dict[ChainName, ChainDefinition]

    @model_validator(mode="after")
    def _validate_cross_references(self) -> "SkeletonDefinition":
        rb_names = set(self.rigid_bodies.keys())

        # Validate linkages reference existing RBs, and shared keypoints exist
        for link_name, linkage in self.linkages.items():
            if linkage.parent_rigid_body not in rb_names:
                raise ValueError(
                    f"Linkage '{link_name}' references parent rigid body "
                    f"'{linkage.parent_rigid_body}' which does not exist. "
                    f"Available: {sorted(rb_names)}"
                )
            parent_rb = self.rigid_bodies[linkage.parent_rigid_body]
            if linkage.shared_keypoint not in parent_rb.keypoints:
                raise ValueError(
                    f"Linkage '{link_name}' shared keypoint "
                    f"'{linkage.shared_keypoint}' not found on parent RB "
                    f"'{linkage.parent_rigid_body}'. "
                    f"Available keypoints: {parent_rb.keypoints}"
                )

            for child_name in linkage.child_rigid_bodies:
                if child_name not in rb_names:
                    raise ValueError(
                        f"Linkage '{link_name}' references child rigid body "
                        f"'{child_name}' which does not exist. "
                        f"Available: {sorted(rb_names)}"
                    )
                child_rb = self.rigid_bodies[child_name]
                if linkage.shared_keypoint not in child_rb.keypoints:
                    raise ValueError(
                        f"Linkage '{link_name}' shared keypoint "
                        f"'{linkage.shared_keypoint}' not found on child RB "
                        f"'{child_name}'. "
                        f"Available keypoints: {child_rb.keypoints}"
                    )

        # Validate chains reference existing linkages and are connected
        linkage_names = set(self.linkages.keys())
        for chain_name, chain in self.chains.items():
            if chain.root_rigid_body not in rb_names:
                raise ValueError(
                    f"Chain '{chain_name}' references root rigid body "
                    f"'{chain.root_rigid_body}' which does not exist. "
                    f"Available: {sorted(rb_names)}"
                )

            for i, link_name in enumerate(chain.linkages):
                if link_name not in linkage_names:
                    raise ValueError(
                        f"Chain '{chain_name}' references linkage "
                        f"'{link_name}' which does not exist. "
                        f"Available: {sorted(linkage_names)}"
                    )

            # Validate chain connectivity
            self._validate_chain_connectivity(chain_name, chain)

        return self

    def _validate_chain_connectivity(
        self, chain_name: str, chain: ChainDefinition
    ) -> None:
        """Verify the chain forms a connected path through the linkage graph."""
        # Root RB must be the parent of the first linkage
        first_linkage = self.linkages[chain.linkages[0]]
        if first_linkage.parent_rigid_body != chain.root_rigid_body:
            raise ValueError(
                f"Chain '{chain_name}': root rigid body "
                f"'{chain.root_rigid_body}' is not the parent of "
                f"the first linkage '{chain.linkages[0]}' "
                f"(parent is '{first_linkage.parent_rigid_body}')"
            )

        # Each consecutive pair of linkages must be connected:
        # one of linkage_i's child RBs must be linkage_(i+1)'s parent RB
        for i in range(len(chain.linkages) - 1):
            current_linkage = self.linkages[chain.linkages[i]]
            next_linkage = self.linkages[chain.linkages[i + 1]]

            if next_linkage.parent_rigid_body not in current_linkage.child_rigid_bodies:
                raise ValueError(
                    f"Chain '{chain_name}': linkage '{chain.linkages[i]}' "
                    f"(children: {current_linkage.child_rigid_bodies}) "
                    f"does not connect to linkage '{chain.linkages[i + 1]}' "
                    f"(parent: '{next_linkage.parent_rigid_body}'). "
                    f"Chain is disconnected."
                )

    @cached_property
    def all_keypoint_names(self) -> list[str]:
        """
        Union of all keypoints across all rigid bodies, deduplicated.
        Order: first appearance across RBs in dict iteration order.
        """
        seen: set[str] = set()
        result: list[str] = []
        for rb in self.rigid_bodies.values():
            for kp in rb.keypoints:
                if kp not in seen:
                    seen.add(kp)
                    result.append(kp)
        return result

    # --- Dot-access namespaces ---
    # Enable: skeleton.rb.skull, skeleton.link.right_elbow_joint, skeleton.chain.axial

    @cached_property
    def rb(self) -> DotAccessDict:
        """Dot-access to rigid bodies: `skeleton.rb.skull`."""
        return DotAccessDict(self.rigid_bodies)

    @cached_property
    def link(self) -> DotAccessDict:
        """Dot-access to linkages: `skeleton.link.right_elbow_joint`."""
        return DotAccessDict(self.linkages)

    @cached_property
    def chain(self) -> DotAccessDict:
        """Dot-access to chains: `skeleton.chain.right_arm`."""
        return DotAccessDict(self.chains)

    @cached_property
    def segment_connections(self) -> dict[str, dict[str, str]]:
        """
        For each 2-keypoint (under-constrained) RB, map its name to
        {"proximal": origin, "distal": non-origin keypoint}.
        """
        result: dict[str, dict[str, str]] = {}
        for rb_name, rb in self.rigid_bodies.items():
            if not rb.is_fully_constrained:
                distal = rb.default_distal_keypoint
                if not isinstance(distal, str):
                    raise ValueError(
                        f"Under-constrained RB '{rb_name}' has non-string "
                        f"default_distal_keypoint: {distal}"
                    )
                result[rb_name] = {"proximal": rb.origin, "distal": distal}
        return result

    @cached_property
    def joint_hierarchy(self) -> dict[str, list[str]]:
        """
        Parent→children keypoint tree derived from linkage traversal.

        For each linkage, the shared_keypoint is the parent. For each child RB,
        all keypoints that are NOT the shared_keypoint are children of the
        shared_keypoint.
        """
        hierarchy: dict[str, list[str]] = {}
        for linkage in self.linkages.values():
            parent_kp = linkage.shared_keypoint
            if parent_kp not in hierarchy:
                hierarchy[parent_kp] = []

            for child_rb_name in linkage.child_rigid_bodies:
                child_rb = self.rigid_bodies[child_rb_name]
                for kp in child_rb.keypoints:
                    if kp != parent_kp and kp not in hierarchy[parent_kp]:
                        hierarchy[parent_kp].append(kp)

        return hierarchy

    @cached_property
    def junction_keypoints(self) -> set[str]:
        """
        Keypoints appearing in more than one chain's joint sequence.
        These are reconciliation points for branching IK solvers (e.g. FABRIK).
        """
        all_joint_seqs: list[list[str]] = []
        for chain in self.chains.values():
            seq = self._compute_chain_joint_sequence(chain)
            all_joint_seqs.append(seq)

        # Count occurrences across chains
        from collections import Counter
        kp_counts: Counter[str] = Counter()
        for seq in all_joint_seqs:
            kp_counts.update(set(seq))

        return {kp for kp, count in kp_counts.items() if count > 1}

    def get_chain_joint_sequence(self, chain_name: str) -> list[str]:
        """
        Ordered list of keypoints a chain passes through.
        This is the joint list an IK solver operates on.
        """
        if chain_name not in self.chains:
            raise KeyError(
                f"Chain '{chain_name}' not found. "
                f"Available: {sorted(self.chains.keys())}"
            )
        return self._compute_chain_joint_sequence(self.chains[chain_name])

    def _compute_chain_joint_sequence(self, chain: ChainDefinition) -> list[str]:
        """Compute the joint sequence for a chain."""
        root_rb = self.rigid_bodies[chain.root_rigid_body]
        sequence = [root_rb.origin]

        for link_name in chain.linkages:
            linkage = self.linkages[link_name]
            sequence.append(linkage.shared_keypoint)

        return sequence
