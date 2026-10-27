"""
Frame sources for BlurItLive.

One tiny unified interface (`read() -> (ok, frame_bgr)`) over three inputs:
  - screen : live capture of a monitor (the real streaming case)
  - webcam : live camera (for the bystander-face demo)
  - file   : a pre-recorded clip (most reliable for a scripted demo video)
"""

import cv2
import numpy as np


class WebcamSource:
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam {index}")

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class FileSource:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file: {path}")

    def read(self):
        ok, frame = self.cap.read()
        return ok, frame

    def release(self):
        self.cap.release()


class ScreenSource:
    """Captures one monitor using mss. monitor=1 is the primary display."""

    def __init__(self, monitor=1):
        from mss import mss  # imported lazily so file/webcam modes need no mss
        self.sct = mss()
        if monitor >= len(self.sct.monitors):
            monitor = 1
        self.region = self.sct.monitors[monitor]

    def read(self):
        shot = self.sct.grab(self.region)
        frame = np.array(shot)                      # BGRA
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return True, frame

    def release(self):
        pass


def create_source(source, path=None, monitor=1, webcam_index=0):
    if source == "screen":
        return ScreenSource(monitor=monitor)
    if source == "webcam":
        return WebcamSource(index=webcam_index)
    if source == "file":
        if not path:
            raise ValueError("--path is required when --source file")
        return FileSource(path)
    raise ValueError(f"Unknown source: {source}")
