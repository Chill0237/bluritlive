"""Face detector using OpenCV Haar cascades (no extra downloads).

Runs frontal + profile, and flips the frame to catch the other-side profile,
so turning your head no longer makes the box vanish. Still a classic detector:
Stage II swaps this for a CNN face detector (YuNet) on the NPU, which handles
angles and motion far better.

`fast=True` runs the frontal cascade only (1 pass instead of 3) and a coarser
scaleFactor. On a CPU-only machine (e.g. an Apple-silicon Mac, where neither
Haar nor EasyOCR can use the GPU) this roughly triples throughput for a smooth
demo recording; the full 3-pass mode stays the default for quality.
"""

import cv2
from detectors.base import Detector, Detection


class FaceDetector(Detector):
    name = "face"

    def __init__(self, min_size=50, fast=False, min_neighbors=5):
        d = cv2.data.haarcascades
        self.frontal = cv2.CascadeClassifier(d + "haarcascade_frontalface_default.xml")
        self.profile = cv2.CascadeClassifier(d + "haarcascade_profileface.xml")
        self.min_size = min_size
        self.fast = fast
        self.min_neighbors = min_neighbors   # lower = more sensitive (catches more, more false +)

    def _run(self, cascade, gray):
        return cascade.detectMultiScale(
            gray, scaleFactor=1.2 if self.fast else 1.1,
            minNeighbors=self.min_neighbors,
            minSize=(self.min_size, self.min_size),
        )

    def detect(self, frame):
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        boxes = []
        for (x, y, w, h) in self._run(self.frontal, gray):
            boxes.append((x, y, w, h))
        if not self.fast:
            for (x, y, w, h) in self._run(self.profile, gray):          # right profile
                boxes.append((x, y, w, h))
            W = gray.shape[1]
            for (x, y, w, h) in self._run(self.profile, cv2.flip(gray, 1)):   # left profile
                boxes.append((W - x - w, y, w, h))
        return [Detection(int(x), int(y), int(w), int(h), label="face")
                for (x, y, w, h) in boxes]
