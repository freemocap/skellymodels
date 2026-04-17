from dataclasses import dataclass
from typing import List, Set

from skellymodels.chain import Chain
from skellymodels.keypoint import KeypointDefinition
from skellymodels.linkage import Linkage
from skellymodels.segments import Segment
from skellymodels.type_aliases import KeypointName


@dataclass
class Skeleton:
    """
    A Skeleton is composed of chains with connecting KeyPoints.
    """
    parent: Chain
    children: List[Chain]

    def get_name(self) -> str:
        return self.__class__.__name__

    @property
    def root(self) -> KeypointName:
        # Skeleton -> Chain -> Linkage -> Segment -> Keypoint
        return self.parent.root

    @classmethod
    def get_linkages(cls) -> List[Linkage]:
        linkages = []
        linkages.extend(cls.parent.get_linkages())
        for chain in cls.children:
            linkages.extend(chain.get_linkages())
        return list(set(linkages))

    @classmethod
    def get_segments(cls) -> List[Segment]:

        segments = []
        segments.extend(cls.parent.get_segments())
        for chain in cls.children:
            segments.extend(chain.get_segments())
        return list(set(segments))

    @classmethod
    def get_keypoints(cls) -> List[KeypointDefinition]:
        keypoints = []
        for chain in cls.children:
            keypoints.extend(chain.get_keypoints())
        return keypoints

    @classmethod
    def get_keypoint_children(cls, keypoint_name: str) -> List[KeypointDefinition]:
        """
        Recursively get all children keypoints for a given keypoint name.

        Parameters
        ----------
        keypoint_name : str
            The name of the keypoint to find children for.

        Returns
        -------
        Set[KeypointDefinition]
            A set of all children keypoints.
        """

        def recursive_find_children(name: str,
                                    segments: List[Segment],
                                    found_keypoint_children: Set[KeypointDefinition]) -> None:
            for segment in segments:
                if segment.origin.lower() == name:
                    children = segment.get_children()
                    for child in children:
                        if child not in found_keypoint_children:  # Avoid infinite recursion
                            found_keypoint_children.add(child)
                            recursive_find_children(name=child.lower(),
                                                    segments=segments,
                                                    found_keypoint_children=found_keypoint_children)

        found_keypoint_children = set()
        recursive_find_children(name=keypoint_name,
                                segments=cls.get_segments(),
                                found_keypoint_children=found_keypoint_children)
        return list(found_keypoint_children)
