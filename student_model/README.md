# `student_model/` — MobileNetV2 student + Knowledge Distillation (Partition 3)

Distils the ResNet50 teacher (`teacher_model/`) into a lightweight torchvision **MobileNetV2** with a 29-class head.
Reuses the existing P2 dataset loader, augmentation, metrics, optimizer builder, early stopping and utilities;
P1 preprocessing and the teacher code are untouched.

## Architecture

| File | Responsibility |
|---|---|
| `model.py` | `build_student()` – torchvision MobileNetV2, `classifier[-1]` → `Linear(1280, 29)`, optional ImageNet weights (`IMAGENET1K_V2`). A failed download raises `RuntimeError` (no silent random-weights fallback). |
| `distillation.py` | `DistillationLoss`, `kd_divergence`, `validate_kd_hyperparameters` – independent of data/teacher. |
| `teacher.py` | Loads the teacher through `teacher_model.checkpoint.load_teacher`, verifies class list / `class_to_idx` / output size / image size / normalisation, freezes it, gradient-free `teacher_logits()` (re-asserts `eval()` on each call). |
| `checkpoint.py` | Student checkpoints (`architecture="mobilenet_v2"`), `load_student()` for inference. Needed because the teacher's checkpoint functions hard-code `resnet50`. |
| `trainer.py` | `train_epoch()` (KD step) and `fit()` (validation-based selection, early stopping, resume, best + last checkpoints). |
| `evaluate.py` | Validation / internal-test evaluation, parameter count and size, CLI. |
| `train.py` | CLI `python -m student_model.train` and `run()` (train → evaluate validation + internal test). |
| `config.py` | `StudentConfig` (mirrors `TeacherConfig`; `to_data_config()` feeds the existing data pipeline). |

## Loss

With raw logits `z_s` (student), `z_t` (teacher), temperature `T > 0`, `alpha ∈ [0, 1]`:

```
L_CE = CrossEntropy(z_s, y)
L_KD = T² · KL( softmax(z_t / T) ‖ softmax(z_s / T) )      (batch-mean, F.kl_div(..., reduction="batchmean"))
L    = alpha · L_CE + (1 − alpha) · L_KD
```

Student uses `log_softmax`, teacher uses `softmax`; teacher logits are detached; computed in float32 under AMP.
Defaults `T=4`, `alpha=0.5` are conventional starting points, **not tuned values**. Validation loss/accuracy use plain
cross-entropy so `val_loss` is comparable to the teacher's.

## Data & leakage rules

Only `train.csv` (fit), `validation.csv` (model selection) and `test.csv` (single final evaluation) are read, through
`teacher_model.dataset.build_datasets`, which also runs the split-leakage assertion. `external_test.csv` is never read.
Teacher and student receive the **same augmented batch** (same size, same ImageNet normalisation); a teacher trained at a
different `--image-size` or normalisation is rejected.

## Training

```bash
python -m student_model.train \
    --teacher-checkpoint model_weights/teacher_resnet50_best.pth \
    --temperature 4 --alpha 0.5 --epochs 20 --batch-size 64 --device cuda
```

Useful flags: `--no-pretrained`, `--optimizer {adamw,sgd}`, `--monitor {val_loss,val_accuracy}`, `--patience`,
`--output-dir`, `--checkpoint-name`, `--dataset-root`, `--resume <*_last.pth>`, `--overwrite`, `--max-train-batches N`
(smoke test), `--no-amp`.

Outputs (default `model_weights/`, none overwrite teacher files):

* `student_mobilenetv2_kd_best.pth` – best on validation; `student_mobilenetv2_kd_last.pth` – latest epoch (resume).
* `student_mobilenetv2_reports/`: `training_history.json`, `run_config.json`, `final_summary.json`,
  `{validation,test}_metrics.json` (accuracy, macro/weighted precision/recall/F1, per-class, confusion matrix,
  `model_footprint`), `*_classification_report.txt`, `*_confusion_matrix.png`, and
  `teacher_reference_{validation,test}_metrics.json` (the teacher measured on the same internal splits).

The run aborts (exit code 1, nothing written) if the teacher checkpoint is missing/incompatible, if the student output
path would equal the teacher checkpoint, or if the best checkpoint exists without `--overwrite`/`--resume`.
Resume additionally requires the same teacher (SHA-256 stored in the student checkpoint).

## Evaluation

```bash
python -m student_model.evaluate --checkpoint model_weights/student_mobilenetv2_kd_best.pth --split both
```

## Using a student checkpoint

```python
from student_model.checkpoint import load_student
model, ckpt = load_student("model_weights/student_mobilenetv2_kd_best.pth", device="cpu")  # eval mode
ckpt["class_names"], ckpt["input_size"], ckpt["normalization"]   # use with teacher_model.augmentation.EvalTransform
```

Checkpoint keys: `architecture`, `num_classes`, `class_names`, `class_to_idx`, `input_size`, `normalization`,
`model_state_dict`, `epoch`, `best_val_metric`, `best_metric_name`, `config`, `distillation` (T, alpha, label smoothing),
`teacher` (file name, SHA-256, epoch, metric), `history`, plus optimizer / scheduler / scaler state.

## Google Colab

```python
from google.colab import drive; drive.mount("/content/drive")
!git clone https://github.com/Vivk1560/ASL-Sign-Language-KD && cd ASL-Sign-Language-KD
# copy student_model/ and tests/test_student_model.py into the repo, then:
!pip install -r requirements-train.txt
# Runtime > Change runtime type > GPU. Put processed/manifests, dataset and the teacher checkpoint on Drive/local disk.
!python -m student_model.train \
    --processed-dir /content/drive/MyDrive/asl/processed \
    --dataset-root /content/asl_alphabet_train/asl_alphabet_train \
    --teacher-checkpoint /content/drive/MyDrive/asl/teacher_resnet50_best.pth \
    --output-dir /content/drive/MyDrive/asl/student --device cuda --batch-size 64 --num-workers 2
```

All paths are CLI arguments; nothing Colab-specific is hard-coded. Save `--output-dir` on Drive so `--resume` survives
runtime resets.

## Tests

`python -m pytest tests/test_student_model.py -q` – synthetic data only (a tiny random ResNet50 stands in for the
teacher); no dataset, no ImageNet download, no real checkpoint required.
