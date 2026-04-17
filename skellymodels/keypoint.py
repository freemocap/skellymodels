from abc import ABC
from dataclasses import dataclass



@dataclass
class KeypointDefinition:
    name: str
    definition: str

    def __str__(self):
        return f"Keypoint: {self.name}"
