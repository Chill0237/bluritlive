"""
Text + PII detector.

Reads every text region with EasyOCR, groups the regions into text *lines*
(OCR often splits one visual line into several boxes), classifies the whole
line with `rules.classify_text`, and if the line is private it returns ONE
detection covering the entire line. Judging a full line fixes the case where a
split like "no. 7" + "roosevelt rd" left the street number exposed, and it also
makes the blur boxes look clean.

This is the "any source" core: it does not care whether the text came from a
notification, an autocomplete dropdown, a DM, or a game overlay. OCR is heavy,
so main.py runs it every N frames; that batching is what the UGen300 NPU
accelerates in Stage II.
"""

from detectors.base import Detector, Detection
import rules


class TextPiiDetector(Detector):
    name = "text-pii"

    def __init__(self, languages=("en",), use_gpu=True, min_conf=0.3):
        import easyocr  # lazy: app can run face-only without easyocr/torch
        self.reader = easyocr.Reader(list(languages), gpu=use_gpu)
        self.min_conf = min_conf

    @staticmethod
    def _box(bbox):
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        return min(xs), min(ys), max(xs), max(ys)

    def _group_lines(self, results):
        """Group OCR boxes that sit on the same visual line."""
        items = []
        for bbox, text, conf in results:
            if conf < self.min_conf:
                continue
            x0, y0, x1, y1 = self._box(bbox)
            items.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1,
                          "cy": (y0 + y1) / 2, "h": max(1, y1 - y0), "text": text})
        items.sort(key=lambda it: (it["cy"], it["x0"]))

        lines = []
        for it in items:
            placed = False
            for ln in lines:
                if abs(it["cy"] - ln["cy"]) < 0.6 * max(it["h"], ln["h"]):
                    ln["parts"].append(it)
                    ln["x0"] = min(ln["x0"], it["x0"]); ln["y0"] = min(ln["y0"], it["y0"])
                    ln["x1"] = max(ln["x1"], it["x1"]); ln["y1"] = max(ln["y1"], it["y1"])
                    ln["cy"] = (ln["y0"] + ln["y1"]) / 2
                    ln["h"] = max(ln["h"], it["h"])
                    placed = True
                    break
            if not placed:
                lines.append({"parts": [it], "x0": it["x0"], "y0": it["y0"],
                              "x1": it["x1"], "y1": it["y1"], "cy": it["cy"], "h": it["h"]})
        return lines

    def detect(self, frame):
        results = self.reader.readtext(frame)
        found = []
        for ln in self._group_lines(results):
            parts = sorted(ln["parts"], key=lambda it: it["x0"])
            line_text = " ".join(p["text"] for p in parts)
            label = rules.classify_text(line_text)
            if not label:
                continue
            pad = 8
            x = int(max(0, ln["x0"] - pad))
            y = int(max(0, ln["y0"] - pad))
            w = int(ln["x1"] - ln["x0"] + 2 * pad)
            h = int(ln["y1"] - ln["y0"] + 2 * pad)
            found.append(Detection(x, y, w, h, label=label))
        return found
