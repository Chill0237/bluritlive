"""Applies the masking. Clean blur by default (no hard black frame)."""

import cv2


class Redactor:
    def __init__(self, style="blur", show_tags=False):
        self.style = style              # "blur" | "pixelate" | "box"
        self.show_tags = show_tags

    def _mask_region(self, frame, d):
        H, W = frame.shape[:2]
        x1, y1 = max(0, d.x), max(0, d.y)
        x2, y2 = min(W, d.x + d.w), min(H, d.y + d.h)
        if x2 <= x1 or y2 <= y1:
            return
        roi = frame[y1:y2, x1:x2]
        if self.style == "box":
            frame[y1:y2, x1:x2] = (20, 20, 20)
        elif self.style == "pixelate":
            # fixed ~10 cells across the region, so even huge text is destroyed
            cells = 10
            sw = max(1, (x2 - x1) // (max(x2 - x1, y2 - y1) // cells or 1))
            sh = max(1, (y2 - y1) // (max(x2 - x1, y2 - y1) // cells or 1))
            small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_LINEAR)
            frame[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1),
                                             interpolation=cv2.INTER_NEAREST)
        else:  # blur
            # Scale the blur to the region size, not a fixed kernel: shrink the
            # ROI hard, blur the tiny copy, then grow it back. This wipes out
            # large-scale structure (big headline text) that a fixed-kernel
            # GaussianBlur only softens, while still looking like a smooth blur.
            longest = max(x2 - x1, y2 - y1)
            shrink = max(1, longest // 16)        # bigger region -> shrink more
            sw = max(1, (x2 - x1) // shrink)
            sh = max(1, (y2 - y1) // shrink)
            small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_LINEAR)
            k = max(5, (min(sw, sh) // 2) | 1)
            small = cv2.GaussianBlur(small, (k, k), 0)
            frame[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1),
                                             interpolation=cv2.INTER_LINEAR)

    def apply(self, frame, detections):
        for d in detections:
            self._mask_region(frame, d)
            if self.show_tags:
                p = (d.x, max(14, d.y - 6))
                cv2.putText(frame, d.label, p, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, d.label, p, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 1, cv2.LINE_AA)
        return frame
