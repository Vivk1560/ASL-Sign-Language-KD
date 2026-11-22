"""Partition 4: webcam pipeline logic. No physical camera, GPU, MediaPipe or real checkpoint is used."""

from __future__ import annotations

import inspect
import subprocess
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from deployment import webcam
from deployment.predictor import Prediction
from deployment.webcam import (CameraError, CenterRoiLocator, FrameResult, MediaPipeUnavailable, PredictionSmoother,
                               State, draw_overlay, make_locator, mirror_box, process_frame, square_box)

CLASSES = ["A", "B", "nothing"]


class FakePredictor:
    """Records the arrays it receives and returns scripted probabilities."""
    class_names = CLASSES

    def __init__(self, probs=(0.9, 0.05, 0.05)):
        self.probs = list(probs)
        self.received: list[np.ndarray] = []

    def predict_rgb_array(self, rgb, top_k=1):
        self.received.append(rgb.copy())
        i = int(np.argmax(self.probs))
        return Prediction(CLASSES[i], i, self.probs[i], ((CLASSES[i], self.probs[i]),), tuple(self.probs))


class FixedLocator:
    name, detects_hands = "fixed", True

    def __init__(self, box):
        self.box, self.closed = box, False

    def locate(self, rgb):
        return self.box

    def close(self):
        self.closed = True


def _frame(h=120, w=160):
    return np.zeros((h, w, 3), np.uint8)


# ------------------------------------------------------------------ geometry
def test_square_box_is_square_inside_frame_and_clamped():
    for args in [(10, 10, 50, 40), (0, 0, 20, 20), (140, 100, 160, 120), (0, 0, 160, 120)]:
        x1, y1, x2, y2 = square_box(*args, width=160, height=120)
        assert x2 - x1 == y2 - y1 and 0 <= x1 and 0 <= y1 and x2 <= 160 and y2 <= 120


def test_square_box_margin_and_min_side():
    x1, y1, x2, y2 = square_box(50, 50, 150, 150, 300, 300, margin=0.25)
    assert x2 - x1 == 150                                  # 100 * (1 + 2 * 0.25)
    tiny = square_box(50, 50, 52, 52, 200, 200, margin=0.0, min_side=64)
    assert tiny[2] - tiny[0] == 64


def test_center_roi_is_centred_square_and_validated():
    box = CenterRoiLocator(0.5).locate(_frame(120, 160))
    assert box[2] - box[0] == box[3] - box[1] == 60
    assert (box[0] + box[2]) / 2 == 80 and (box[1] + box[3]) / 2 == 60
    assert CenterRoiLocator.detects_hands is False
    with pytest.raises(ValueError):
        CenterRoiLocator(0.0)


# ------------------------------------------------------------------ states
def test_no_hand_is_its_own_state_and_never_calls_the_model():
    pred, smoother = FakePredictor(), PredictionSmoother(3)
    res = process_frame(_frame(), pred, FixedLocator(None), smoother, 0.6)
    assert res.state is State.NO_HAND and res.label is None and res.box is None
    assert pred.received == []                            # and certainly not reported as the 'nothing' class
    assert "nothing" not in res.state.value.lower()


def test_threshold_separates_prediction_from_low_confidence():
    box = (10, 10, 74, 74)
    high = process_frame(_frame(), FakePredictor((0.9, 0.05, 0.05)), FixedLocator(box), PredictionSmoother(1), 0.6)
    low = process_frame(_frame(), FakePredictor((0.4, 0.35, 0.25)), FixedLocator(box), PredictionSmoother(1), 0.6)
    assert (high.state, high.label) == (State.PREDICTION, "A")
    assert low.state is State.LOW_CONFIDENCE and low.label == "A" and low.confidence == pytest.approx(0.4)


def test_nothing_class_from_the_model_is_a_normal_prediction():
    res = process_frame(_frame(), FakePredictor((0.0, 0.0, 1.0)), FixedLocator((0, 0, 64, 64)), PredictionSmoother(1), 0.6)
    assert res.state is State.PREDICTION and res.label == "nothing"


def test_smoother_averages_and_resets_when_hand_is_lost():
    s = PredictionSmoother(2)
    s.update([1.0, 0.0])
    assert s.update([0.0, 1.0]).tolist() == [0.5, 0.5]
    s.update([0.0, 1.0])
    assert s.update([0.0, 1.0]).tolist() == [0.0, 1.0]    # window slid
    smoother = PredictionSmoother(5)
    process_frame(_frame(), FakePredictor(), FixedLocator((0, 0, 64, 64)), smoother, 0.6)
    process_frame(_frame(), FakePredictor(), FixedLocator(None), smoother, 0.6)
    assert len(smoother._buf) == 0
    with pytest.raises(ValueError):
        PredictionSmoother(0)


# ------------------------------------------------------------------ preprocessing / orientation
def test_bgr_is_converted_to_rgb_and_crop_matches_box():
    frame = _frame()
    frame[:, :, 0] = 255                                   # blue channel only (OpenCV BGR)
    frame[20:30, 40:50, 2] = 200                           # a red patch
    pred = FakePredictor()
    process_frame(frame, pred, FixedLocator((40, 20, 104, 84)), PredictionSmoother(1), 0.6)
    crop = pred.received[0]
    assert crop.shape == (64, 64, 3)
    assert crop[..., 2].min() == 255 and crop[..., 0].max() == 200       # RGB order: R=patch, B=255


def test_model_input_never_depends_on_mirror_setting():
    assert "mirror" not in inspect.signature(process_frame).parameters   # inference cannot see the display flag
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    keep = frame.copy()
    result = FrameResult(State.PREDICTION, (20, 10, 84, 74), "A", 0.9)
    loc = FixedLocator(None)
    draw_overlay(frame, result, loc, mirror=True, fps=10, threshold=0.6)
    draw_overlay(frame, result, loc, mirror=False, fps=10, threshold=0.6)
    assert np.array_equal(frame, keep)                                     # raw frame is never mutated/flipped


def test_mirror_is_display_only_and_box_is_mirrored_consistently():
    assert mirror_box((20, 10, 84, 74), 160) == (76, 10, 140, 74)
    frame = _frame(200, 300)
    frame[100, 0] = (0, 0, 255)                            # left-edge marker in the RAW frame
    res = FrameResult(State.NO_HAND, None)
    mirrored = draw_overlay(frame, res, FixedLocator(None), True, 0.0, 0.6)
    plain = draw_overlay(frame, res, FixedLocator(None), False, 0.0, 0.6)
    assert tuple(plain[100, 0]) == (0, 0, 255) and tuple(mirrored[100, 299]) == (0, 0, 255)


def test_overlay_states_render_without_error():
    loc = CenterRoiLocator()
    for res in (FrameResult(State.NO_HAND, None), FrameResult(State.LOW_CONFIDENCE, (0, 0, 50, 50), "A", 0.3),
                FrameResult(State.PREDICTION, (0, 0, 50, 50), "B", 0.95)):
        out = draw_overlay(_frame(), res, loc, True, 12.0, 0.6, warning="MediaPipe unavailable")
        assert out.shape == (120, 160, 3)


def test_process_frame_rejects_invalid_frames():
    for bad in (None, np.zeros((10, 10), np.uint8)):
        with pytest.raises(CameraError):
            process_frame(bad, FakePredictor(), FixedLocator(None), PredictionSmoother(1), 0.6)


# ------------------------------------------------------------------ MediaPipe is optional
def test_make_locator_roi_mode():
    loc, warning = make_locator("roi")
    assert isinstance(loc, CenterRoiLocator) and warning is None


def test_mediapipe_missing_falls_back_with_visible_warning(monkeypatch):
    monkeypatch.setitem(sys.modules, "mediapipe", None)                    # import raises ImportError
    loc, warning = make_locator("auto")
    assert isinstance(loc, CenterRoiLocator) and "fallback" in warning
    with pytest.raises(MediaPipeUnavailable):
        make_locator("mediapipe")


def test_mediapipe_without_solutions_api_is_reported_with_pinned_version(monkeypatch):
    monkeypatch.setitem(sys.modules, "mediapipe", types.SimpleNamespace(__version__="9.9.9"))
    loc, warning = make_locator("auto")
    assert isinstance(loc, CenterRoiLocator) and "0.10.14" in warning and "9.9.9" in warning


def test_deployment_modules_import_without_mediapipe_or_cv2():
    code = ("import sys, deployment.predictor, deployment.infer, deployment.compare, deployment.benchmark, "
            "deployment.webcam; assert 'mediapipe' not in sys.modules; assert 'cv2' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).resolve().parents[1])


# ------------------------------------------------------------------ capture handling
def _video(path: Path, frames: int = 6) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    for i in range(frames):
        writer.write(np.full((120, 160, 3), 30 + i, np.uint8))
    writer.release()
    return path


def test_open_source_errors_are_actionable(tmp_path):
    with pytest.raises(CameraError, match="video file"):
        webcam.open_source(str(tmp_path / "missing.avi"))


def test_run_processes_every_frame_and_cleans_up(tmp_path):
    video = _video(tmp_path / "v.avi", 6)
    loc = FixedLocator((20, 10, 84, 74))
    summary = webcam.run(FakePredictor(), loc, str(video), show_window=False, smooth=2)
    assert summary["frames"] == 6 and summary["state_counts"][State.PREDICTION.value] == 6
    assert loc.closed


def test_run_respects_max_frames_and_counts_no_hand(tmp_path):
    video = _video(tmp_path / "v.avi", 6)
    summary = webcam.run(FakePredictor(), FixedLocator(None), str(video), show_window=False, max_frames=3)
    assert summary["frames"] == 3 and summary["state_counts"][State.NO_HAND.value] == 3


def test_run_releases_resources_when_inference_fails(tmp_path):
    class Boom(FakePredictor):
        def predict_rgb_array(self, rgb, top_k=1):
            raise RuntimeError("boom")

    loc = FixedLocator((0, 0, 64, 64))
    with pytest.raises(RuntimeError):
        webcam.run(Boom(), loc, str(_video(tmp_path / "v.avi")), show_window=False)
    assert loc.closed


def test_main_reports_missing_checkpoint_before_touching_the_camera(tmp_path, capsys):
    assert webcam.main(["--checkpoint", str(tmp_path / "none.pth"), "--camera", "0"]) == 2
    assert "error:" in capsys.readouterr().err
