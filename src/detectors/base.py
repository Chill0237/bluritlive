"""
Shared types for detectors.

A detector takes a BGR frame and returns a list of Detection boxes to redact.
Keeping this interface tiny is deliberate: at Stage II the regex-based text
detector is swapped for a learned context classifier running on the UGen300
NPU, with no change to the rest of the pipeline.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    label: str = "private"      # what kind of leak this is, shown as a tag


class Detector:
    name = "detector"

    def detect(self, frame) -> List[Detection]:
        raise NotImplementedError
