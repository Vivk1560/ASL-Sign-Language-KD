# teacher_model — ResNet50 Teacher (Partition 2)

Trains the **ResNet50 teacher** that will later guide the MobileNetV2 student in Knowledge Distillation (KD).
Scope: teacher only. The student, KD loss, webcam/MediaPipe inference and deployment are **not** part of this folder.

> **Status:** implemented and unit-tested on synthetic data. A full training run on the real ~87k-image
> dataset has **not** been executed by the author of this code; no accuracy figures are claimed here.

## 1. Why a teacher?
In KD the student learns from the teacher's *soft* probabilities (which letters it confuses, e.g. M/N), not only
from hard labels. A strong, well-evaluated teacher is therefore needed first. The checkpoint saved here stores
everything the KD stage needs (class list, input size, normalisation, raw-logit model).

## 2. Architecture
Standard `torchvision.models.resnet50`; the 1000-way `fc` layer is replaced by `Linear(2048, 29)`.
Output: raw logits `[batch, 29]`. Input: `3 x 224 x 224` (configurable), ImageNet mean/std normalisation.
* `--pretrained` (default) downloads ImageNet weights on first use (**needs internet**, cached afterwards).
  If the download fails the program stops with an error; it never silently falls back.
* `--no-pretrained` trains from random weights (no download).

## 3. How the preprocessing output is consumed
`python -m preprocessing.main --class-set 29 ...` already produced:
`data/processed/manifests/{train,validation,test}.csv`, `metadata/labels.json`, `metadata/dataset_summary.json`.
`teacher_model` only **reads** these:
* class indices come from `labels.json` (every row's `class_index` is cross-checked);
* each row's `split` must match its file; no `image_path`, `sha256` or `group_id` may occur in two splits
  (otherwise training aborts with a "leakage" error);
* `image_path` is relative to the dataset class root (`class_root` in `dataset_summary.json`, or `--dataset-root`);
* `external_test.csv` is never opened. The split is never recomputed.

## 4. Dataset loading
`ASLManifestDataset` is a lazy PyTorch `Dataset`: one image is opened from disk per `__getitem__`, so RAM use
is ~one mini-batch. Missing/unreadable files raise a `DatasetError` naming the file. `DataLoader` does the batching
(shuffle for train only; `persistent_workers` when `num_workers>0`; works on Windows).

## 5. On-the-fly SciPy augmentation
Training only: `resize -> scipy.ndimage.rotate -> tensor -> normalise`. Per access, an angle is drawn uniformly
from `[-rotation_deg, +rotation_deg]` (default 15°). Details (`augmentation.py`):
* `axes=(1, 0)` rotates the spatial plane only (the colour axis of an HWC array is untouched);
* `reshape=False` keeps the image size, `order=1` bilinear, empty corners filled with each channel's mean;
* nothing is written to disk and no expanded dataset is kept — the rotated image lives for one step;
* per-worker/per-epoch RNG seeding so workers don't repeat angles.
Validation and internal test use the deterministic resize+normalise only.
Try other limits with `--rotation-deg 10|20|25`.

## 6. Training workflow
CrossEntropyLoss, **AdamW** (lr 1e-3, weight decay 1e-4; `--optimizer sgd` available), cosine LR schedule,
batch size 32, up to 20 epochs, seed 42, CUDA mixed precision when a GPU exists (CPU works too).
After each epoch the validation split gives loss/accuracy. The **best epoch by validation loss**
(`--monitor val_accuracy` optional) is saved; training stops after `--patience` (5) epochs without improvement.
Only after training finishes is the best checkpoint evaluated on the **internal test** split. The test split never
influences checkpoint selection or hyper-parameters.

## 7. Evaluation outputs
Accuracy, macro/weighted precision, recall, F1, per-class scores, classification report, confusion matrix
(JSON + `.txt` + PNG) in `model_weights/teacher_resnet50_reports/` (`validation_*`, `test_*`, `training_history.json`, `run_config.json`).
The official 28-image external test set is intentionally not used here.

## 8. Checkpoints (`.pth`, `state_dict` based)
* `model_weights/teacher_resnet50_best.pth` — best validation epoch. Contains `model_state_dict`, `architecture`,
  `num_classes`, `class_names`, `class_to_idx`, `input_size`, `normalization`, `epoch`, `best_val_metric`, `config`,
  `history`, optimizer/scheduler/scaler state.
* `model_weights/teacher_resnet50_last.pth` — latest epoch, used by `--resume`.
* The pre-existing `model_weights/teacher_best_aug.pth` is **not touched** (different provenance/format). The trainer refuses to
  overwrite an existing best checkpoint unless `--overwrite` is given.
* No `.pkl` files are produced.

## 9. Installation (Windows PowerShell)
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install PyTorch for YOUR hardware first (CPU or CUDA): https://pytorch.org/get-started/locally/
pip install -r requirements.txt
pip install -r requirements-train.txt
```
(`requirements-train.txt` supplies scikit-learn, used for metrics. Pinned `torch`/`torchvision` in `requirements.txt`
may be replaced by the CUDA build chosen on the PyTorch site.)

## 10. Train
```powershell
# 0) preprocessing must have been run once (creates data\processed\...)
python -m preprocessing.main --class-set 29 --force --require-external-test

# 1) full training
python -m teacher_model.train

# examples
python -m teacher_model.train --epochs 15 --batch-size 64 --learning-rate 5e-4 --rotation-deg 20 --num-workers 4
python -m teacher_model.train --dataset-root "D:\data\raw\asl_alphabet_train\asl_alphabet_train"
python -m teacher_model.train --no-pretrained --device cpu --epochs 1 --max-train-batches 5   # quick smoke test
python -m teacher_model.train --resume model_weights\teacher_resnet50_last.pth --epochs 25
python -m teacher_model.evaluate --checkpoint model_weights\teacher_resnet50_best.pth --split test
```
If `--num-workers` > 0 causes trouble on Windows, use `--num-workers 0`.

## 11. Hardware
GPU with >= 6 GB VRAM recommended (batch 32, AMP). On CPU, ResNet50 over ~61k images/epoch is very slow
(many hours per epoch) — use it only for smoke tests.

## 12. Loading the teacher later (KD stage)
```python
from teacher_model.checkpoint import load_teacher
teacher, ckpt = load_teacher("model_weights/teacher_resnet50_best.pth", device="cuda")  # eval mode, frozen later by you
logits = teacher(batch)            # [B, 29] raw logits -> softmax(logits / T) for distillation
class_names = ckpt["class_names"]; mean_std = ckpt["normalization"]
```
No ImageNet download is needed to load a checkpoint.

## 13. Tests
`python -m pytest tests/test_teacher_model.py` — synthetic images only, no dataset, no downloads.
