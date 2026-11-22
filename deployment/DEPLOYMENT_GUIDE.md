# Partition 4: Deployment, Evaluation and Benchmarking

This document describes what was implemented in `deployment/`, how to use it, and, just as important,
what has **not** been verified yet.

## Implemented vs verified

| Capability | Implemented | Verified how |
|---|---|---|
| Student/teacher `Predictor` (checkpoint loading, single-image and array inference, error handling) | yes | automated tests with synthetic checkpoints |
| `deployment.infer` CLI | yes | automated tests (synthetic checkpoint and images) |
| `deployment.compare` | yes | automated tests with mock models; CLI error paths |
| `deployment.benchmark` | yes | automated tests with a fake timer; one tiny end-to-end CLI run |
| `deployment.webcam` | yes | pure-logic tests; headless run on a synthetic video file with a random-initialised model |
| Inference with the **recovered** student checkpoint | yes | **not yet run** (checkpoint is a local artifact) |
| Teacher-vs-student numbers from the real checkpoints | yes | **not yet run** |
| Benchmark numbers on real hardware (CPU/GPU) | yes | **not yet run** |
| Webcam with a **physical camera** | yes | **not verified** |
| MediaPipe hand detection | yes (optional) | **not verified**; the installed MediaPipe 0.10.33 in the development sandbox has no `mp.solutions` API, so only the fallback path was exercised |

## Design

- **No duplication.** Models come from `student_model.model` / `teacher_model.model` via `load_student` /
  `load_teacher`; preprocessing is `teacher_model.augmentation.EvalTransform`, the transform used for validation and test.
  The checkpoint's `class_names`, `input_size` and `normalization` are read from the checkpoint; a normalization that differs from
  `EvalTransform`'s is refused.
- **Safe loading.** Checkpoints are read with `weights_only=True`. Missing files, corrupt files, raw `state_dict` files, wrong architectures and
  the legacy `*_best_aug.pth` names produce a one-line error, not a traceback.
- **Configurable paths.** Command-line flag, then the `ASL_STUDENT_CHECKPOINT` / `ASL_TEACHER_CHECKPOINT` environment variable, then the defaults
  `student_output/student_mobilenetv2_kd_best.pth` and `model_weights/teacher_resnet50_best.pth`. Checkpoints are git-ignored (`*.pth`, `student_output/`).

## 29 classes

The model predicts A-Z plus `del`, `nothing` and `space`. The labels come from the checkpoint. In the webcam app:

- `del`, `space` and `nothing` are ordinary model outputs and are displayed as such.
- **No hand detected** (MediaPipe mode) is a separate UI state and is never mapped to `nothing`.
- In centre-ROI fallback mode there is no hand detection, so the model decides, including `nothing`.

## Webcam pipeline and preprocessing

1. Read a raw BGR frame (not mirrored).
2. Locate the hand: MediaPipe Hands (one hand, square crop with 25% margin) or, as a clearly labelled fallback, a centred square ROI.
3. Crop, convert BGR to RGB, apply `EvalTransform` (resize to the checkpoint's input size, ImageNet normalization).
4. Student forward pass, softmax, moving average of the probabilities over `--smooth` frames (reset when the hand is lost).
5. Show the letter only if the smoothed confidence is at least `--threshold`; otherwise show LOW CONFIDENCE.
6. Display: the mirror is a display-only flip applied after inference (box coordinates are mirrored, text is not). The model never sees a mirrored frame.

MediaPipe: install `pip install mediapipe==0.10.14` (pinned in `requirements.txt`). Newer MediaPipe releases removed `mp.solutions`; the app detects
that, falls back to the ROI and prints a warning. `requirements.txt` pins `numpy==2.4.4` together with `mediapipe==0.10.14`; whether that combination
installs and runs on your machine has **not** been verified here. If it does not, use `--hand-detector roi`.

Webcam limitations: lighting, background, camera angle, hand distance and hand shape differ from the training data; a plain background and a hand
filling most of the box work best. Latency and FPS depend on your hardware and have not been measured here.

## Comparison methodology

`deployment.compare` loads both models with the existing loaders, builds the same manifest-driven loader as the Partition 2/3 evaluators
(`EvalTransform`, no shuffling), runs both models on identical batches in a single pass, and computes accuracy, macro precision, recall and F1 with
`teacher_model.metrics.compute_metrics`. It also reports prediction agreement (both correct / teacher-only / student-only / both wrong), parameter counts,
the parameter reduction and checkpoint file sizes. Only the validation and internal-test splits are available; the external test set is never touched.
`--max-batches` is for smoke tests only and marks the output as partial.

## Benchmark methodology

`deployment.benchmark` measures the **model forward pass only**: a fixed-seed random tensor `[batch, 3, S, S]` (S from the checkpoint), `--warmup` untimed
passes, then `--repeats` blocks of `--iterations` timed passes under `torch.inference_mode()`, with CUDA synchronised around each timed pass.
It reports mean, median, standard deviation, min, max and p95 latency, per-repeat means, and throughput (`batch / mean latency`) for each model, device and batch size,
plus parameter counts, checkpoint sizes and the environment (Python, PyTorch, CPU threads). GPU runs are skipped and reported as skipped when CUDA is unavailable.
Preprocessing, image decoding, camera I/O and hand detection are **not** included, so end-to-end webcam FPS will be lower. Results depend on the machine
and on background load; compare models only within one run on one machine, and cite the generated JSON, not numbers from memory.

## Reported training results (not independently re-verified)

From the recovered Kaggle student run, as reported by the project team: best epoch 7, validation accuracy 0.9999, internal-test accuracy 0.9995, internal-test macro-F1 0.9995
(teacher reference internal-test accuracy 0.9998), early stopping after 5 epochs without improvement, MobileNetV2 ImageNet pretrained weights **not** used,
about 2.26 M student vs 23.57 M teacher parameters (about 90.41% fewer). These are internal-split results.
They say nothing about real-world or webcam accuracy.

## Limitations

- **Static images only.** The model classifies one image at a time.
- **J and Z are dynamic signs** (they involve motion). Single-frame classification cannot capture the motion, so these two letters are not genuinely recognised,
  and no temporal gesture recognition exists.
- **Not continuous ASL translation.** Only the fingerspelling alphabet plus `del`, `nothing`, `space` is covered; no words, grammar or sentences.
- **No real-world robustness claim.** Internal-test accuracy reflects the same dataset distribution; separate testing with other people, backgrounds, lighting
  and cameras is required before any such claim.
- Prediction smoothing and the confidence threshold are heuristics for a steadier demo, not calibrated probabilities.
- Hardware-dependent behaviour (camera access, MediaPipe, GPU timing) is unverified until run on the target machine.

## Remaining tasks that need the real checkpoint / hardware

1. `python -m deployment.infer --info ...` and a few real images with the recovered student checkpoint.
2. `python -m deployment.compare ...` on the local dataset and both checkpoints.
3. `python -m deployment.benchmark ...` on the target CPU (and GPU if available).
4. Run `python -m deployment.webcam ...` with a real camera, with and without MediaPipe, and record what was observed.
