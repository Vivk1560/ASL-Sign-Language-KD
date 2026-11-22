# ASL Gesture Recognition Using Knowledge Distillation

**Group 21 | B.Tech Information Technology | AISSMS IOIT**

A deep learning-based American Sign Language (ASL) alphabet recognition system designed to achieve efficient, lightweight, and real-time gesture classification.

The project explores **Knowledge Distillation (KD)** to transfer learned representations from a larger ResNet50 teacher network to a lightweight MobileNetV2 student network. The student model is intended for efficient inference on commodity hardware and edge devices.

## Project Overview

The objective is to develop an ASL alphabet recognition system that can classify hand gestures accurately while keeping inference computationally efficient.

The proposed system combines:

- **MediaPipe Hands:** Hand detection and landmark-based tracking from webcam frames.
- **ResNet50:** Teacher network for learning visual representations from ASL images.
- **MobileNetV2:** Lightweight student network designed for efficient inference.
- **Knowledge Distillation:** Training approach that transfers knowledge from the teacher model to the student model.
- **OpenCV:** Webcam frame capture and real-time prediction display.

### System Workflow

1. **Dataset preparation**
   - Inspect and validate ASL images.
   - Detect exact and near-duplicate images.
   - Create group-aware training, validation, and test datasets.
   - Generate reproducible dataset manifests.

2. **Teacher model training**
   - Train a ResNet50 model on the prepared dataset.
   - Evaluate its classification performance.
   - Save the best-performing checkpoint.

3. **Knowledge distillation**
   - Train a MobileNetV2 student model.
   - Transfer knowledge from the trained ResNet50 teacher using temperature-scaled distillation loss.
   - Compare the student model against its baseline and teacher.

4. **Inference and demonstration**
   - Capture webcam frames using OpenCV.
   - Detect hand regions using MediaPipe.
   - Classify detected hand gestures using the student model.
   - Display predicted letters and confidence scores.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Computer Vision | OpenCV, MediaPipe |
| Data Processing | NumPy, Pandas |
| Image Processing | Pillow |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib |

## Datasets

### 1. ASL Alphabet Dataset â€” Kaggle

**Source:** [ASL Alphabet Dataset by grassknoted](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

The primary dataset contains 87,000 images distributed across 29 classes:

- 26 English alphabet letters (Aâ€“Z).
- `del`
- `nothing`
- `space`

It is used for dataset preparation, model training, validation, and internal testing.

### 2. American Sign Language Letters â€” Roboflow

**Source:** [American Sign Language Letters Dataset](https://public.roboflow.com/object-detection/american-sign-language-letters/1)

This dataset contains real-world images with varying backgrounds and Pascal VOC-style annotations, including hand bounding boxes.

It can support experimentation with hand localization and recognition in less controlled environments.

The two datasets serve different purposes. The Kaggle dataset is the primary classification dataset; the Roboflow dataset is intended for additional real-world experimentation.

## Model Architecture

The system follows a teacher-student architecture:

**Teacher Network â€” ResNet50**

- A deeper convolutional neural network.
- Learns visual features from ASL hand gesture images.
- Provides soft-target predictions for knowledge distillation.

**Student Network â€” MobileNetV2**

- A lightweight convolutional neural network.
- Designed for lower computational requirements.
- Learns from both ground-truth labels and the teacher's predictions.

**Knowledge Distillation**

The student is trained using a combination of classification loss and temperature-scaled distillation loss.

This approach aims to retain useful knowledge from the larger teacher network while reducing model complexity.

### Architecture Diagram

![ASL Recognition System Architecture](https://github.com/user-attachments/assets/51231f86-07ac-4cf1-9467-bd763964accd)

## Dataset Preprocessing

A modular preprocessing pipeline is implemented in the `preprocessing/` directory.

Its responsibilities include:

- Dataset discovery and structural inspection.
- Image validation and corrupt-file handling.
- Exact duplicate detection using SHA-256.
- Near-duplicate detection using perceptual difference hashing (dHash).
- Anchor-based duplicate grouping to prevent excessive transitive chaining.
- Group-aware train, validation, and internal test splitting.
- External test image isolation.
- Label generation and reproducible CSV manifests.
- Training-only image augmentation.
- Dataset validation and preprocessing reports.

### Dataset Split

The current verified preprocessing run produced the following distribution:

| Split | Images | Percentage |
|---|---:|---:|
| Training | 60,872 | 70% |
| Validation | 13,050 | 15% |
| Internal Test | 13,050 | 15% |
| **Total** | **86,972** | **100%** |

The 28 official Kaggle external test images were checked against the internal dataset using SHA-256 hashes. Their exact internal copies were excluded before duplicate grouping and splitting.

The official external folder does not contain a `del_test.jpg`; this is an expected dataset limitation.

The external test set is kept separate from training and is intended for an additional smoke test. It is not large enough to serve as a comprehensive performance benchmark.

### Important Dataset Considerations

- Duplicate groups are kept within a single split to reduce data leakage.
- Only exact copies of official external test images are excluded; near-duplicate external images may still exist in the internal dataset.
- Anchor-based grouping limits chaining but cannot guarantee that every visually similar neighbouring frame belongs to the same split.
- The letters J and Z involve motion. A single-frame classifier does not provide complete temporal sign recognition.

## Installation and Setup

### Prerequisites

- Python 3.11
- Git
- A compatible PyTorch installation
- Kaggle ASL Alphabet dataset

### 1. Clone the repository

```bash
git clone https://github.com/Vivk1560/ASL-Sign-Language-KD.git
cd ASL-Sign-Language-KD
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-train.txt
```

Install the appropriate PyTorch build for your system by following the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/).

### 4. Prepare the dataset

Place the extracted Kaggle dataset in the repository's expected raw-data directory:

```text
data/
â””â”€â”€ raw/
    â”œâ”€â”€ asl_alphabet_train/
    â”‚   â””â”€â”€ asl_alphabet_train/
    â”‚       â”œâ”€â”€ A/
    â”‚       â”œâ”€â”€ B/
    â”‚       â””â”€â”€ ...
    â””â”€â”€ asl_alphabet_test/
        â””â”€â”€ asl_alphabet_test/
```

The preprocessing pipeline can also use a custom dataset location through its `--raw-dir` argument.

### 5. Run preprocessing

```powershell
python -m preprocessing.main --class-set 29 --force --require-external-test
```

This command processes all 29 classes, regenerates preprocessing outputs, and requires the official external test folder to be available for isolation verification.

Generated manifests, metadata, and reports are written under `data/processed/`.

## Partition 4: Inference, Comparison, Benchmarking and Webcam

All deployment code lives in `deployment/` and reuses the Partition 2/3 model builders, checkpoint loaders and the
exact validation/test `EvalTransform`. Full details: [`deployment/DEPLOYMENT_GUIDE.md`](deployment/DEPLOYMENT_GUIDE.md).

> **Checkpoints are local artifacts and are not in Git.** The recovered student checkpoint is expected at
> `student_output/student_mobilenetv2_kd_best.pth`. Every command below accepts an explicit path; the default can also be
> set with the `ASL_STUDENT_CHECKPOINT` / `ASL_TEACHER_CHECKPOINT` environment variables.
> `model_weights/teacher_best_aug.pth` and `model_weights/student_best_aug.pth` are **legacy** pre-Partition-2/3 files and
> are refused by the deployment tools.

### Single-image inference (no camera, no MediaPipe needed)

```powershell
python -m deployment.infer path\to\image.jpg --checkpoint student_output\student_mobilenetv2_kd_best.pth
python -m deployment.infer a.jpg b.jpg --checkpoint student_output\student_mobilenetv2_kd_best.pth --json
python -m deployment.infer --info --checkpoint student_output\student_mobilenetv2_kd_best.pth   # stored metadata only
```

### Webcam inference

```powershell
python -m deployment.webcam --checkpoint student_output\student_mobilenetv2_kd_best.pth
```

Keys: `q` / `Esc` quit, `m` toggle the (display-only) mirror. Options: `--camera 1` (other camera or a video file path),
`--hand-detector auto|mediapipe|roi`, `--threshold 0.6`, `--smooth 5`, `--no-mirror`, `--no-window`, `--max-frames N`.

- **MediaPipe is optional.** With a usable MediaPipe (`pip install mediapipe==0.10.14`, the pinned version) the hand is
  detected and cropped. Otherwise the app uses a fixed **centre-ROI fallback** (a box you must put your hand in) and says so
  on screen; in that mode no hand detection takes place.
- "No hand detected" is a separate state and is never shown as the `nothing` class (which only appears when the model predicts it).
- The model always receives the raw, un-mirrored frame; mirroring only affects the preview.
- Camera access: grant camera permission to your terminal/Python (Windows: Settings > Privacy > Camera) and close other apps using the camera.

### Teacher vs student comparison and benchmarking

```powershell
pip install -r requirements-train.txt     # scikit-learn is needed by the comparison metrics
python -m deployment.compare --student-checkpoint student_output\student_mobilenetv2_kd_best.pth `
    --teacher-checkpoint model_weights\teacher_resnet50_best.pth --split test
python -m deployment.benchmark --student-checkpoint student_output\student_mobilenetv2_kd_best.pth `
    --teacher-checkpoint model_weights\teacher_resnet50_best.pth --devices cpu cuda
```

Both write JSON (and, for `compare`, Markdown) into `deployment_reports/`. Benchmark numbers exist only after you run it on
your machine; the repository contains none. The comparison reads the validation/internal-test split only, never the external test set.

### Results (reported training/evaluation run)

These are the figures reported by the project team for the recovered Kaggle student run. They are internal-split results
and are **not** evidence of real-world or webcam accuracy.

| Item | Value |
|---|---|
| Student best epoch | 7 (early stopping after 5 epochs without improvement) |
| Student validation accuracy | 0.9999 |
| Student internal-test accuracy / macro-F1 | 0.9995 / 0.9995 |
| Teacher internal-test accuracy (reference) | 0.9998 |
| Parameters (student / teacher) | approx. 2.26 M / 23.57 M (approx. 90.41% fewer) |
| Student ImageNet pretrained weights | **not used** |

Run `python -m deployment.compare ...` to regenerate teacher/student metrics from the actual checkpoints; no webcam,
latency or CPU/GPU numbers are reported here until they have been measured.

### Limitations (summary)

- The model classifies **single static images**. **J** and **Z** are motion signs; a single frame cannot capture them, so
  they are not genuinely recognised and no temporal gesture recognition is implemented.
- This is **not** continuous ASL translation; it is a per-frame fingerspelling-alphabet classifier (26 letters plus `del`, `nothing`, `space`).
- Internal-test accuracy comes from the same dataset distribution. Robustness to other backgrounds, lighting, skin tones,
  hands and cameras has **not** been tested.

## Project Structure

```text
ASL-Sign-Language-KD/
â”‚
â”œâ”€â”€ preprocessing/
â”‚   â”œâ”€â”€ config.py
â”‚   â”œâ”€â”€ data_cleaning.py
â”‚   â”œâ”€â”€ data_splitting.py
â”‚   â”œâ”€â”€ duplicate_detection.py
â”‚   â”œâ”€â”€ dataset_inspection.py
â”‚   â”œâ”€â”€ dataset_validation.py
â”‚   â”œâ”€â”€ external_test.py
â”‚   â””â”€â”€ main.py
â”‚
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/
â”‚   â””â”€â”€ processed/
â”‚       â”œâ”€â”€ manifests/
â”‚       â”œâ”€â”€ metadata/
â”‚       â””â”€â”€ reports/
â”‚
â”œâ”€â”€ teacher_model/              # Partition 2 (ResNet50 teacher)
â”œâ”€â”€ student_model/              # Partition 3 (MobileNetV2 student + knowledge distillation)
â”‚
â”œâ”€â”€ deployment/                 # Partition 4
â”‚   â”œâ”€â”€ predictor.py            # checkpoint-backed Predictor (shared by all tools)
â”‚   â”œâ”€â”€ infer.py                # single-image CLI
â”‚   â”œâ”€â”€ webcam.py               # webcam / video demo
â”‚   â”œâ”€â”€ compare.py              # teacher vs student evaluation
â”‚   â””â”€â”€ benchmark.py            # latency / throughput benchmark
â”‚
â”œâ”€â”€ deployment/DEPLOYMENT_GUIDE.md
â”œâ”€â”€ tests/
â”œâ”€â”€ asl_mediapipe_final.py      # LEGACY (see below)
â”‚
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ requirements-train.txt
â””â”€â”€ README.md
```

`asl_mediapipe_final.py` is a **LEGACY** prototype from before Partitions 2-3. It defines its own 26-class model wrappers, loads raw `state_dict` files (`model_weights/*_best_aug.pth`) and duplicates preprocessing. It is kept untouched for reference only; use `deployment/` instead.

## Current Implementation Status

- [x] Dataset inspection and validation.
- [x] Exact and near-duplicate detection.
- [x] Anchor-based duplicate grouping.
- [x] Group-aware dataset splitting.
- [x] External test isolation.
- [x] Reproducible manifests and preprocessing reports.
- [x] Automated preprocessing tests.
- [x] ResNet50 teacher model training (Partition 2).
- [x] MobileNetV2 student training with knowledge distillation (Partition 3; student retrained, checkpoints held locally).
- [x] Student inference, teacher-vs-student comparison and benchmarking code (Partition 4, covered by automated tests).
- [ ] Teacher-vs-student comparison executed on the recovered checkpoints (pending a local run).
- [ ] Benchmarks executed on target hardware (pending a local run).
- [ ] Webcam application verified with a real camera and the recovered checkpoint (pending; only code-level and synthetic-video checks exist).

## Future Scope

- Improve recognition under varying lighting conditions and backgrounds.
- Evaluate generalization on independent real-world ASL images.
- Optimize student-model inference for edge devices.
- Explore temporal models for dynamic signs such as J and Z.
- Integrate hand detection, classification, and prediction stabilization into a complete webcam application.

## Contributors

**Group 21 â€” B.Tech Information Technology**
AISSMS Institute of Information Technology (AISSMS IOIT)


