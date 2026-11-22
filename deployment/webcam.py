"""Webcam / video demo: hand localisation -> square crop -> EvalTransform -> student -> overlay.

Design rules (see documentation/PARTITION4.md):
  * The model always receives the RAW, un-mirrored camera frame. The mirrored preview is a display-only
    flip applied after inference, so mirroring can never change the model's expected input orientation.
  * MediaPipe is OPTIONAL. If it is missing or too new (no ``mp.solutions``), the app falls back to a
    fixed centre ROI and says so on screen. In ROI mode there is no hand detection at all.
  * "No hand detected" is its own UI state and is never reported as the ``nothing`` class.
  * Predictions below ``--threshold`` are shown as LOW CONFIDENCE, not as a letter.
  * Static single-frame classifier: J and Z are motion signs and cannot be recognised properly.

    python -m deployment.webcam --checkpoint student_output/student_mobilenetv2_kd_best.pth
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

import numpy as np

from .predictor import DeploymentError, Predictor, Prediction, default_checkpoint

Box = tuple[int, int, int, int]   # x1, y1, x2, y2 in RAW (un-mirrored) frame pixels


class State(str, Enum):
    NO_HAND = "NO HAND"                 # MediaPipe mode only: nothing to classify
    LOW_CONFIDENCE = "LOW CONFIDENCE"   # classified, but below the threshold
    PREDICTION = "PREDICTION"           # classified with confidence >= threshold


class CameraError(DeploymentError):
    pass


class MediaPipeUnavailable(DeploymentError):
    pass


class Locator(Protocol):
    name: str
    detects_hands: bool
    def locate(self, rgb: np.ndarray) -> Optional[Box]: ...
    def close(self) -> None: ...


# ---------------------------------------------------------------- locators
def square_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int,
               margin: float = 0.25, min_side: int = 64) -> Box:
    """Square box around a bounding box, enlarged by ``margin``, shifted/clamped to stay inside the frame."""
    side = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * margin)
    side = int(round(min(max(side, min_side), min(width, height))))
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    left = int(round(min(max(cx - side / 2.0, 0), width - side)))
    top = int(round(min(max(cy - side / 2.0, 0), height - side)))
    return left, top, left + side, top + side


class CenterRoiLocator:
    """Fallback: always returns a fixed centred square. Performs NO hand detection."""
    name = "center-ROI fallback (no hand detection)"
    detects_hands = False

    def __init__(self, fraction: float = 0.6) -> None:
        if not 0.1 <= fraction <= 1.0:
            raise ValueError("ROI fraction must be in [0.1, 1.0]")
        self.fraction = fraction

    def locate(self, rgb: np.ndarray) -> Optional[Box]:
        h, w = rgb.shape[:2]
        side = int(min(h, w) * self.fraction)
        return square_box(w / 2 - side / 2, h / 2 - side / 2, w / 2 + side / 2, h / 2 + side / 2,
                          w, h, margin=0.0, min_side=1)

    def close(self) -> None:
        pass


class MediaPipeHandLocator:
    """MediaPipe Hands (legacy ``mp.solutions`` API, as pinned in requirements.txt: mediapipe==0.10.14)."""
    detects_hands = True

    def __init__(self, margin: float = 0.25, min_detection_confidence: float = 0.5) -> None:
        try:
            import mediapipe as mp
        except Exception as exc:
            raise MediaPipeUnavailable(f"mediapipe is not installed ({exc.__class__.__name__}: {exc})") from exc
        if not hasattr(mp, "solutions"):
            raise MediaPipeUnavailable(
                f"mediapipe {getattr(mp, '__version__', '?')} has no 'solutions' API "
                "(removed in newer releases); install the pinned version: pip install mediapipe==0.10.14")
        self.name = f"MediaPipe Hands {getattr(mp, '__version__', '')}".strip()
        self.margin = margin
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=1, min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5)

    def locate(self, rgb: np.ndarray) -> Optional[Box]:
        h, w = rgb.shape[:2]
        res = self._hands.process(rgb)
        if not res.multi_hand_landmarks:
            return None
        lm = res.multi_hand_landmarks[0].landmark
        xs = [p.x * w for p in lm]
        ys = [p.y * h for p in lm]
        return square_box(min(xs), min(ys), max(xs), max(ys), w, h, self.margin)

    def close(self) -> None:
        self._hands.close()


def make_locator(mode: str, roi_fraction: float = 0.6) -> tuple[Locator, Optional[str]]:
    """``mode``: 'auto' (MediaPipe if usable, else ROI), 'mediapipe' (error if unusable) or 'roi'.
    Returns ``(locator, warning)``; ``warning`` is set when the ROI fallback replaced MediaPipe."""
    if mode == "roi":
        return CenterRoiLocator(roi_fraction), None
    try:
        return MediaPipeHandLocator(), None
    except MediaPipeUnavailable as exc:
        if mode == "mediapipe":
            raise
        return CenterRoiLocator(roi_fraction), f"MediaPipe unavailable, using centre-ROI fallback: {exc}"


# ---------------------------------------------------------------- per-frame logic (no GUI, no mirroring)
class PredictionSmoother:
    """Moving average of probability vectors over the last ``window`` frames; cleared when the hand is lost."""

    def __init__(self, window: int = 5) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self._buf: deque[np.ndarray] = deque(maxlen=window)

    def update(self, probabilities: Sequence[float]) -> np.ndarray:
        self._buf.append(np.asarray(probabilities, dtype=np.float64))
        return np.mean(self._buf, axis=0)

    def reset(self) -> None:
        self._buf.clear()


@dataclass(frozen=True)
class FrameResult:
    state: State
    box: Optional[Box]
    label: Optional[str] = None
    confidence: Optional[float] = None


def process_frame(frame_bgr: np.ndarray, predictor: Predictor, locator: Locator,
                  smoother: PredictionSmoother, threshold: float) -> FrameResult:
    """Classify one RAW (un-mirrored) BGR frame. Mirroring is deliberately not a parameter here."""
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise CameraError("Camera returned an invalid frame.")
    rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])            # BGR -> RGB
    box = locator.locate(rgb)
    if box is None:
        smoother.reset()
        return FrameResult(State.NO_HAND, None)
    x1, y1, x2, y2 = box
    pred: Prediction = predictor.predict_rgb_array(rgb[y1:y2, x1:x2], top_k=1)
    avg = smoother.update(pred.probabilities)
    idx = int(np.argmax(avg))
    conf = float(avg[idx])
    state = State.PREDICTION if conf >= threshold else State.LOW_CONFIDENCE
    return FrameResult(state, box, predictor.class_names[idx], conf)


def mirror_box(box: Box, width: int) -> Box:
    x1, y1, x2, y2 = box
    return width - x2, y1, width - x1, y2


# ---------------------------------------------------------------- display (cv2 imported lazily)
def draw_overlay(raw_bgr: np.ndarray, result: FrameResult, locator: Locator, mirror: bool, fps: float,
                 threshold: float, warning: Optional[str] = None) -> np.ndarray:
    import cv2
    h, w = raw_bgr.shape[:2]
    view = cv2.flip(raw_bgr, 1) if mirror else raw_bgr.copy()     # display-only flip; text is drawn after it
    colors = {State.PREDICTION: (60, 200, 60), State.LOW_CONFIDENCE: (0, 190, 255), State.NO_HAND: (60, 60, 230)}
    color = colors[result.state]
    if result.box is not None:
        x1, y1, x2, y2 = mirror_box(result.box, w) if mirror else result.box
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
    if result.state is State.NO_HAND:
        headline = "NO HAND DETECTED"
    elif result.state is State.LOW_CONFIDENCE:
        headline = f"low confidence ({result.confidence:.0%} < {threshold:.0%})"
    else:
        headline = f"{result.label}  {result.confidence:.0%}"
    cv2.rectangle(view, (0, 0), (w, 58), (0, 0, 0), -1)
    cv2.putText(view, headline, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    mode = locator.name + ("" if locator.detects_hands else "  -- put your hand inside the box")
    lines = [f"{mode} | {fps:.0f} FPS | mirror {'ON' if mirror else 'OFF'}",
             "q/Esc quit, m mirror | static classifier: J and Z (motion signs) not supported"]
    if warning:
        lines.insert(0, "WARNING: " + warning[:110])
    y = h - 8 - 20 * (len(lines) - 1)
    for line in lines:
        cv2.putText(view, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 20
    return view


def open_source(source: str):
    """Open a camera index ('0') or a video file path -> ``(capture, first_frame)``. Raises CameraError."""
    import cv2
    is_index = source.isdigit()
    cap = cv2.VideoCapture(int(source) if is_index else source)
    if not cap.isOpened():
        cap.release()
        hint = ("Check that a camera is connected, that no other app is using it, that the OS has granted "
                "camera access to your terminal/Python (Windows: Settings > Privacy > Camera; macOS: "
                "System Settings > Privacy & Security > Camera), or try another index (--camera 1)."
                if is_index else "Check that the video file exists and is readable.")
        raise CameraError(f"Could not open video source {source!r}. {hint}")
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise CameraError(f"Video source {source!r} opened but delivered no frame.")
    return cap, frame


def run(predictor: Predictor, locator: Locator, source: str = "0", mirror: bool = True, threshold: float = 0.6,
        smooth: int = 5, show_window: bool = True, max_frames: Optional[int] = None,
        warning: Optional[str] = None) -> dict[str, Any]:
    """Main loop. Always releases the capture device and closes windows/locators, even on errors."""
    import cv2
    cap, pending = open_source(source)      # the frame read to validate the source is processed, not dropped
    smoother = PredictionSmoother(smooth)
    frames, t_start, fps = 0, time.perf_counter(), 0.0
    counts = {s.value: 0 for s in State}
    try:
        while max_frames is None or frames < max_frames:
            if pending is not None:
                ok, frame, pending = True, pending, None
            else:
                ok, frame = cap.read()
            if not ok or frame is None:
                if source.isdigit():
                    raise CameraError("Lost the camera feed (device disconnected or taken by another app).")
                break                                              # end of a video file
            result = process_frame(frame, predictor, locator, smoother, threshold)
            counts[result.state.value] += 1
            frames += 1
            elapsed = time.perf_counter() - t_start
            fps = frames / elapsed if elapsed > 0 else 0.0
            if show_window:
                cv2.imshow("ASL alphabet (student)", draw_overlay(frame, result, locator, mirror, fps, threshold, warning))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("m"):
                    mirror = not mirror
    finally:
        cap.release()
        locator.close()
        if show_window:
            cv2.destroyAllWindows()
    return {"frames": frames, "avg_fps": fps, "state_counts": counts}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m deployment.webcam",
                                description="Real-time ASL alphabet demo with the trained student model.")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="student checkpoint (default: $ASL_STUDENT_CHECKPOINT, else "
                        "student_output/student_mobilenetv2_kd_best.pth)")
    p.add_argument("--camera", default="0", help="camera index (default 0) or a video file path")
    p.add_argument("--hand-detector", choices=["auto", "mediapipe", "roi"], default="auto",
                   help="auto: MediaPipe if usable else centre ROI; mediapipe: fail if unusable; roi: always ROI")
    p.add_argument("--roi-fraction", type=float, default=0.6, help="side of the ROI box as a fraction of frame height")
    p.add_argument("--threshold", type=float, default=0.6, help="minimum smoothed confidence to show a letter")
    p.add_argument("--smooth", type=int, default=5, help="frames averaged for prediction smoothing")
    p.add_argument("--no-mirror", action="store_true", help="start with the un-mirrored preview (display only)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--no-window", action="store_true", help="no GUI; print a summary (useful for video files)")
    p.add_argument("--max-frames", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        predictor = Predictor.from_checkpoint(args.checkpoint or default_checkpoint("student"), "student", args.device)
        locator, warning = make_locator(args.hand_detector, args.roi_fraction)
        if warning:
            print("WARNING:", warning, file=sys.stderr)
        print(f"Hand localisation: {locator.name}")
        summary = run(predictor, locator, args.camera, not args.no_mirror, args.threshold, args.smooth,
                      not args.no_window, args.max_frames, warning)
    except DeploymentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 0
    print(f"Processed {summary['frames']} frames, {summary['avg_fps']:.1f} FPS average, states: {summary['state_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
