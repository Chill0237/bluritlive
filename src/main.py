"""
BlurItLive - Stage I prototype
Real-time, on-device privacy net: read the outgoing frame, cover what is
private the instant it appears, show raw-vs-protected side by side.

Run from inside this folder:

    cd src
    python main.py --source screen --compare
    python main.py --source screen --compare --fast   # smooth demo on a CPU-only Mac
    python main.py --source webcam                 # bystander-face demo
    python main.py --source file --path ../demo.mp4 --compare

Keys while running:
    q = quit
    s = save a screenshot to ../captures/
    1..9 = show/hide the element with that number badge
    (or left-click any box to show/hide just that one)

Every covered region is tracked individually, so the streamer can reveal or
re-cover ONE box at a time -- the live "you decide what's private" panel.

This laptop prototype proves the capture -> detect -> mask -> display loop.
Stage II moves the detectors onto the ASUS UGen300 NPU so none of this touches
the GPU the game needs.
"""

import argparse
import os
import threading
import time

import cv2
import numpy as np

from capture import create_source
from redactor import Redactor
from detectors.face_detector import FaceDetector
from detectors.base import Detection


class AsyncOcr:
    """Run the heavy text detector on a background thread.

    EasyOCR on a CPU-only machine takes a few hundred ms per call. Doing that
    inside the main loop is what made the fps swing between 5 and 27: every OCR
    frame stalled display + face work. Here the main loop just submits the
    newest frame and reads whatever the worker last finished. OCR results lag by
    a few hundred ms, which is invisible because on-screen text sits there far
    longer than that. In Stage II the UGen300 NPU does this work in a few ms, so
    the thread (and the lag) go away.
    """

    def __init__(self, detector):
        self.detector = detector
        self._lock = threading.Lock()
        self._latest = None      # newest frame waiting to be processed
        self._result = []        # last finished detections
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame):
        with self._lock:
            self._latest = frame          # overwrite: we only care about the newest

    def results(self):
        with self._lock:
            return self._result

    def _run(self):
        while not self._stop:
            with self._lock:
                frame, self._latest = self._latest, None
            if frame is None:
                time.sleep(0.005)
                continue
            try:
                res = self.detector.detect(frame)
            except Exception:
                res = []
            with self._lock:
                self._result = res

    def stop(self):
        self._stop = True


def parse_args():
    p = argparse.ArgumentParser(description="BlurItLive Stage I prototype")
    p.add_argument("--source", choices=["screen", "webcam", "file"], default="screen")
    p.add_argument("--path", help="video file path (when --source file)")
    p.add_argument("--monitor", type=int, default=1,
                   help="monitor index for screen capture (1 = primary)")
    p.add_argument("--webcam-index", type=int, default=0)
    p.add_argument("--compare", action="store_true",
                   help="show RAW and PROTECTED side by side")
    p.add_argument("--style", choices=["blur", "pixelate", "box"], default="blur")
    p.add_argument("--tags", action="store_true",
                   help="draw a small label on each blocked region")
    p.add_argument("--ocr-interval", type=int, default=12,
                   help="(legacy) no longer used: OCR now runs on a background thread")
    p.add_argument("--face-interval", type=int, default=1,
                   help="run face detection every N frames (persistence hides the gap)")
    p.add_argument("--keep-frames", type=int, default=8,
                   help="keep a face covered this many frames after it was last seen")
    p.add_argument("--text-miss-passes", type=int, default=2,
                   help="uncover text only after it is missing from this many "
                        "consecutive OCR passes (higher = steadier, slower to clear)")
    p.add_argument("--no-ocr", action="store_true", help="faces only; skip the text model")
    p.add_argument("--no-gpu", action="store_true", help="force EasyOCR onto CPU")
    p.add_argument("--lang", nargs="+", default=["en"],
                   help="OCR languages for EasyOCR, e.g. --lang ch_tra en for "
                        "Traditional Chinese + English (needed to read Chinese text)")
    p.add_argument("--max-width", type=int, default=1280,
                   help="downscale wide frames for speed")
    p.add_argument("--fast", action="store_true",
                   help="smooth-demo mode on a CPU-only machine: frontal-only faces, "
                        "detect less often, smaller frame, less frequent OCR")
    p.add_argument("--face-min-size", type=int, default=50,
                   help="smallest face to detect in px; lower it to catch small/group faces")
    p.add_argument("--face-neighbors", type=int, default=5,
                   help="face detection threshold; lower = more sensitive (3 for group photos)")
    args = p.parse_args()
    if args.fast:
        # only override values the user did not set explicitly
        if args.face_interval == 1:
            args.face_interval = 3
        if args.ocr_interval == 12:
            args.ocr_interval = 20
        if args.max_width == 1280:
            args.max_width = 960
        if args.keep_frames == 8:
            args.keep_frames = 12   # cover longer, since we detect less often
    return args


def build_text_detector(use_gpu, languages):
    try:
        from detectors.text_pii_detector import TextPiiDetector
        print(f"Loading text model for {languages} (first run downloads it, please wait)...")
        return TextPiiDetector(languages=languages, use_gpu=use_gpu)
    except Exception as e:
        print(f"[warn] text model unavailable ({e}). Running faces only.")
        return None


def fit_width(frame, max_width):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(h * scale)))


_FONT = cv2.FONT_HERSHEY_SIMPLEX
_GREEN = (90, 210, 90)       # covered / protected
_BLUE = (240, 150, 60)       # revealed by the user (BGR -> looks orange)
_GREY = (110, 110, 110)

PRETTY = {"face": "Face", "secret": "Name / secret", "address": "Address",
          "email": "Email", "phone": "Phone", "tw_id": "ID number",
          "credit_card": "Credit card", "ip": "IP"}


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _dedupe(dets, thr=0.6):
    """Drop near-duplicate boxes (e.g. frontal + profile on one face) so each
    real element becomes exactly one tracked item."""
    kept = []
    for d in dets:
        box = (d.x, d.y, d.w, d.h)
        if all(_iou(box, (k.x, k.y, k.w, k.h)) < thr or k.label != d.label
               for k in kept):
            kept.append(d)
    return kept


class ElementTracker:
    """Give every covered region a stable identity across frames, so the streamer
    can reveal / re-cover ONE box at a time instead of a whole category. New
    detections are matched to existing elements by overlap; each element keeps its
    revealed state while it is still on screen.

    Numbers are reassigned every frame by screen position (top to bottom, then
    left to right), so they always read 1, 2, 3... with no gaps when an element
    disappears. The revealed state lives on the element, not on its number, so
    renumbering never flips what is shown or hidden.

    Faces and text expire differently, because they arrive at different speeds:
      - faces are detected every few frames, so they expire after `face_keep`
        frames without a match;
      - text comes from the background OCR thread, which can take longer than a
        second per pass on a CPU. Counting frames would drop a box before the next
        pass arrives and make it flicker, so text only expires after it is missing
        from `text_miss_passes` consecutive OCR passes."""

    ROW_PX = 40     # boxes whose tops are within this many px count as one row

    def __init__(self, face_keep=8, text_miss_passes=2, iou_thresh=0.3, max_hotkeys=9):
        self.items = []     # {num, box, label, kind, ttl, misses, revealed}
        self.face_keep = face_keep
        self.text_miss = text_miss_passes
        self.iou = iou_thresh
        self.max_hotkeys = max_hotkeys

    def _renumber(self):
        order = sorted(self.items,
                       key=lambda it: (it["box"][1] // self.ROW_PX, it["box"][0]))
        for i, it in enumerate(order):
            it["num"] = i + 1 if i < self.max_hotkeys else None

    def _match(self, dets, kind):
        """Match detections of one kind to existing items; create new items for the
        rest. Returns the set of items that were seen."""
        seen = set()
        for d in _dedupe(dets):
            box = (d.x, d.y, d.w, d.h)
            best, best_iou = None, self.iou
            for it in self.items:
                if it["kind"] != kind or id(it) in seen or it["label"] != d.label:
                    continue
                score = _iou(it["box"], box)
                if score >= best_iou:
                    best, best_iou = it, score
            if best is None:
                best = {"num": None, "box": box, "label": d.label,
                        "kind": kind, "ttl": self.face_keep, "misses": 0,
                        "revealed": False}
                self.items.append(best)
            best["box"] = box
            seen.add(id(best))
        return seen

    def update(self, faces=None, texts=None):
        """Call every frame. `faces` / `texts` are None when that detector produced
        nothing new this frame (faces skipped by --face-interval, or no OCR pass
        finished yet), and a list (possibly empty) when it did."""
        for it in self.items:
            if it["kind"] == "face":
                it["ttl"] -= 1
        if faces is not None:
            seen = self._match(faces, "face")
            for it in self.items:
                if id(it) in seen:
                    it["ttl"] = self.face_keep
        if texts is not None:                      # a fresh OCR pass arrived
            seen = self._match(texts, "text")
            for it in self.items:
                if it["kind"] == "text":
                    it["misses"] = 0 if id(it) in seen else it["misses"] + 1
        self.items = [it for it in self.items
                      if not ((it["ttl"] <= 0) if it["kind"] == "face"
                              else (it["misses"] >= self.text_miss))]
        self._renumber()

    def masked(self):
        return [Detection(*it["box"], label=it["label"])
                for it in self.items if not it["revealed"]]

    def toggle_num(self, num):
        for it in self.items:
            if it["num"] == num:
                it["revealed"] = not it["revealed"]
                return

    def toggle_at(self, x, y):
        hit = None
        for it in self.items:
            bx, by, bw, bh = it["box"]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                if hit is None or bw * bh < hit["box"][2] * hit["box"][3]:
                    hit = it
        if hit is not None:
            hit["revealed"] = not hit["revealed"]


def label_panel(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img


def draw_badges(img, tracker):
    """Number each tracked element on the frame. Green = covered, orange =
    revealed (and outlined, so the user sees exactly what is exposed)."""
    for it in tracker.items:
        x, y, w, h = it["box"]
        covered = not it["revealed"]
        col = _GREEN if covered else _BLUE
        num = str(it["num"]) if it["num"] else "."
        bx, by = x, max(0, y)
        cv2.circle(img, (bx + 12, by + 12), 11, col, -1)
        cv2.putText(img, num, (bx + (7 if len(num) == 1 else 4), by + 17),
                    _FONT, 0.5, (15, 15, 15), 2, cv2.LINE_AA)
        if not covered:
            cv2.rectangle(img, (x, y), (x + w, y + h), col, 2)
    return img


def render_dashboard(width, tracker):
    """Control bar under the video: one chip per tracked element, with its number,
    type, and HIDDEN / SHOWN state. Tap the number or click the box to toggle."""
    pad, title_h, cols, cell_h = 16, 40, 4, 52
    items = sorted(tracker.items, key=lambda it: (it["num"] is None, it["num"] or 999))
    shown = items[:8]
    rows = max(1, (len(shown) + cols - 1) // cols)
    height = title_h + rows * cell_h + pad
    panel = np.full((height, width, 3), 22, dtype=np.uint8)

    cv2.putText(panel, "BlurItLive  |  tap a number or click a box to show / hide it",
                (pad, 27), _FONT, 0.56, (240, 240, 240), 1, cv2.LINE_AA)
    hidden = sum(1 for it in tracker.items if not it["revealed"])
    cv2.putText(panel, f"{hidden} hidden", (width - 140, 27),
                _FONT, 0.55, _GREEN, 1, cv2.LINE_AA)
    cv2.line(panel, (pad, title_h - 6), (width - pad, title_h - 6), (55, 55, 55), 1)

    if not shown:
        cv2.putText(panel, "(nothing private on screen yet)", (pad, title_h + 26),
                    _FONT, 0.5, _GREY, 1, cv2.LINE_AA)
        return panel

    cell_w = (width - pad * 2) // cols
    for i, it in enumerate(shown):
        r, c = divmod(i, cols)
        x = pad + c * cell_w
        y = title_h + r * cell_h
        covered = not it["revealed"]
        col = _GREEN if covered else _BLUE
        cv2.rectangle(panel, (x + 4, y + 2), (x + cell_w - 8, y + cell_h - 6),
                      (40, 40, 40), -1)
        cv2.rectangle(panel, (x + 4, y + 2), (x + cell_w - 8, y + cell_h - 6),
                      (70, 70, 70), 1)
        cy = y + cell_h // 2
        num = str(it["num"]) if it["num"] else "."
        cv2.circle(panel, (x + 22, cy), 11, col, -1)
        cv2.putText(panel, num, (x + (18 if len(num) == 1 else 15), cy + 5),
                    _FONT, 0.5, (15, 15, 15), 2, cv2.LINE_AA)
        cv2.putText(panel, PRETTY.get(it["label"], it["label"]), (x + 42, cy - 2),
                    _FONT, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(panel, "HIDDEN" if covered else "SHOWN", (x + 42, cy + 16),
                    _FONT, 0.44, col, 1, cv2.LINE_AA)
    if len(items) > len(shown):
        cv2.putText(panel, f"+{len(items) - len(shown)} more",
                    (width - 150, height - 10), _FONT, 0.45, _GREY, 1, cv2.LINE_AA)
    return panel


def main():
    args = parse_args()
    src = create_source(args.source, path=args.path, monitor=args.monitor,
                        webcam_index=args.webcam_index)

    face_detector = FaceDetector(min_size=args.face_min_size, fast=args.fast,
                                 min_neighbors=args.face_neighbors)
    text_detector = None if args.no_ocr else build_text_detector(not args.no_gpu, args.lang)
    ocr = AsyncOcr(text_detector) if text_detector is not None else None
    redactor = Redactor(style=args.style, show_tags=args.tags)

    os.makedirs("../captures", exist_ok=True)
    tracker = ElementTracker(face_keep=args.keep_frames,
                             text_miss_passes=args.text_miss_passes)
    frame_i = 0
    t0, fps = time.time(), 0.0
    last_ocr = None
    ui = {"compare": args.compare, "frame_w": 0, "top_h": 0}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if ui["top_h"] and y >= ui["top_h"]:
            return                       # click landed on the dashboard strip, ignore
        fw = ui["frame_w"]
        fx = x - fw if (ui["compare"] and fw and x >= fw) else x   # map right panel
        tracker.toggle_at(fx, y)

    cv2.namedWindow("BlurItLive")
    cv2.setMouseCallback("BlurItLive", on_mouse)
    print("Running. q=quit, s=save. Press a number, or click a box, to show/hide it.")

    while True:
        ok, frame = src.read()
        if not ok or frame is None:
            print("Source ended.")
            break
        frame = fit_width(frame, args.max_width)
        raw = frame.copy()

        # Faces run every --face-interval frames; text arrives from the background
        # OCR thread only when a new pass finishes. Pass None when a detector has
        # nothing new, so the tracker keeps the existing boxes instead of dropping them.
        faces = face_detector.detect(frame) if frame_i % args.face_interval == 0 else None
        texts = None
        if ocr is not None:
            ocr.submit(raw)
            res = ocr.results()
            if res is not last_ocr:
                last_ocr = res
                texts = res
        tracker.update(faces, texts)

        protected = redactor.apply(frame, tracker.masked())
        draw_badges(protected, tracker)

        frame_i += 1
        if frame_i % 10 == 0:
            now = time.time()
            fps = 10.0 / (now - t0)
            t0 = now
        cv2.putText(protected, f"{fps:4.1f} fps", (12, protected.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        if args.compare:
            left = label_panel(raw.copy(), "RAW  (what would leak)")
            right = label_panel(protected.copy(), "PROTECTED  (BlurItLive)")
            top = cv2.hconcat([left, right])
        else:
            top = protected

        ui["frame_w"] = frame.shape[1]
        ui["top_h"] = top.shape[0]
        dash = render_dashboard(top.shape[1], tracker)
        view = cv2.vconcat([top, dash])

        cv2.imshow("BlurItLive", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if ord("1") <= key <= ord("9"):
            tracker.toggle_num(key - ord("0"))
        if key == ord("s"):
            fn = f"../captures/bluritlive_{int(time.time())}.png"
            cv2.imwrite(fn, view)
            print(f"saved {fn}")

    if ocr is not None:
        ocr.stop()
    src.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
