# Partition 3 — Implementation Report (Student MobileNetV2 + Knowledge Distillation)

**Base commit:** `86a6b2b` — "Integrate ResNet50 teacher model training pipeline" (`main`, cloned from GitHub).
**Scope:** new `student_model/` package + `tests/test_student_model.py`. No tracked file was modified or deleted
(`git diff` empty; only untracked additions).

## Codebase findings that shaped the design
* Teacher checkpoints are **dicts** (`format_version`, `architecture="resnet50"`, `model_state_dict`, `class_names`,
  `class_to_idx`, `input_size`, `normalization`, …), not raw state dicts → loaded via the existing `load_teacher`.
* `teacher_model.checkpoint.save_checkpoint/read_checkpoint` hard-code `resnet50` → a separate `student_model/checkpoint.py`
  was unavoidable; it keeps the same layout and adds `distillation` and `teacher` metadata.
* Reused unchanged: `build_datasets/build_dataloaders` (via `StudentConfig.to_data_config()`), `TrainTransform/EvalTransform`,
  `compute_metrics`, `save_confusion_matrix_png`, `evaluate_split`, `run_epoch`, `build_optimizer`, `EarlyStopping`,
  `last_checkpoint_path`, `check_resume_compatible`, `count_parameters`, utils.
* A pre-existing `model_weights/student_best_aug.pth` exists; the new default name `student_mobilenetv2_kd_best.pth` avoids it.

## Requirement coverage
| Requirement | Where |
|---|---|
| MobileNetV2, 29 classes, pretrained configurable, CPU/CUDA | `model.py`, `--no-pretrained`, `--device` |
| Teacher loaded via existing conventions, verified, eval, frozen, no-grad, configurable path | `teacher.py`, `--teacher-checkpoint` |
| KD loss (T², batch-mean KL, log_softmax/softmax), validated T/alpha | `distillation.py` |
| Training: optimizer, cosine LR, early stopping, best/last, resume, logs, AMP on CUDA | `trainer.py`, `train.py` |
| Validation + internal-test evaluation: accuracy, P/R/F1, confusion matrix, params, size | `evaluate.py`, `train.run()` |
| Checkpoint metadata (architecture, classes, epoch, metrics, config, optimizer/scheduler) | `checkpoint.py` |
| No external-test use, no new split, no teacher overwrite | enforced + tested (see below) |

## Safeguards (each covered by a test)
* Class list/order, `class_to_idx`, output size, image size and normalisation of teacher vs dataset are verified; mismatch → `TeacherCompatibilityError` (no fallback).
* Teacher: `requires_grad=False`, `eval()` re-asserted on every forward, BatchNorm statistics unchanged, checkpoint file SHA-256 unchanged after training.
* Student output path may not equal the teacher checkpoint; existing best checkpoint needs `--overwrite`/`--resume`.
* A deliberately corrupt `external_test.csv` does not affect the run (never read).
* Resume refuses a different teacher (SHA-256 mismatch).

## Test results
Command: `python -m pytest -q`
Result: **`131 passed in 90.36s`** = 94 original (unchanged, also 94 passed before my changes) + **37 new** in `tests/test_student_model.py`.
Mutation sanity check: changing `batchmean`→`mean`, or removing `T²`, makes `test_kd_loss_matches_reference_formula` fail (then reverted).
One real bug was found by the tests during development (`to_data_config()` omitted optimizer/monitor/epochs/lr from validation) and fixed.

Test environment: Python 3.12.3, torch 2.15.0, torchvision 0.29.2 (CPU execution), scikit-learn 1.9.1 — **not** the project's declared
Python 3.11.9 / torch 2.13.0 / torchvision 0.28.0 (those were not installable in the sandbox). Only APIs present in the declared
versions are used, but the declared versions themselves were not run.

## What was NOT done (limitations / pending)
* **No real training was performed.** All runs used synthetic 32×32 data (29 samples per split) and a randomly initialised teacher; no accuracy/F1 is claimed for the student or teacher.
* No claim is made that the student is faster, smaller-on-disk or more accurate than the teacher. `model_footprint` reports measured parameter counts and bytes only; **no latency benchmark** was implemented or run.
* ImageNet-pretrained student weights were not downloaded in the sandbox (network blocked); the failure path raises `RuntimeError` as designed, the success path is untested here.
* No real teacher checkpoint or full dataset was available; the loader was tested only on synthetic manifests. CUDA/AMP paths are mirrored from the teacher's trainer but were not executed (CPU-only).
* `T=4`, `alpha=0.5`, lr `1e-3` are untuned defaults. Hyper-parameter search is left for the Colab run.

## Integration steps
1. Copy the ZIP contents into the repo root (paths preserved).
2. Train the teacher on Colab (`python -m teacher_model.train`), then run `python -m student_model.train --teacher-checkpoint <path>`.
3. Run `python -m pytest -q` (expect 131 passed) and commit.
