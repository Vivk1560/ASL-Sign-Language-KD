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

### 1. ASL Alphabet Dataset — Kaggle

**Source:** [ASL Alphabet Dataset by grassknoted](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

The primary dataset contains 87,000 images distributed across 29 classes:

- 26 English alphabet letters (A–Z).
- `del`
- `nothing`
- `space`

It is used for dataset preparation, model training, validation, and internal testing.

### 2. American Sign Language Letters — Roboflow

**Source:** [American Sign Language Letters Dataset](https://public.roboflow.com/object-detection/american-sign-language-letters/1)

This dataset contains real-world images with varying backgrounds and Pascal VOC-style annotations, including hand bounding boxes.

It can support experimentation with hand localization and recognition in less controlled environments.

The two datasets serve different purposes. The Kaggle dataset is the primary classification dataset; the Roboflow dataset is intended for additional real-world experimentation.

## Model Architecture

The system follows a teacher-student architecture:

**Teacher Network — ResNet50**

- A deeper convolutional neural network.
- Learns visual features from ASL hand gesture images.
- Provides soft-target predictions for knowledge distillation.

**Student Network — MobileNetV2**

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
└── raw/
    ├── asl_alphabet_train/
    │   └── asl_alphabet_train/
    │       ├── A/
    │       ├── B/
    │       └── ...
    └── asl_alphabet_test/
        └── asl_alphabet_test/
```

The preprocessing pipeline can also use a custom dataset location through its `--raw-dir` argument.

### 5. Run preprocessing

```powershell
python -m preprocessing.main --class-set 29 --force --require-external-test
```

This command processes all 29 classes, regenerates preprocessing outputs, and requires the official external test folder to be available for isolation verification.

Generated manifests, metadata, and reports are written under `data/processed/`.

## Project Structure

```text
ASL-Sign-Language-KD/
│
├── preprocessing/
│   ├── config.py
│   ├── data_cleaning.py
│   ├── data_splitting.py
│   ├── duplicate_detection.py
│   ├── dataset_inspection.py
│   ├── dataset_validation.py
│   ├── external_test.py
│   └── main.py
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── manifests/
│       ├── metadata/
│       └── reports/
│
├── tests/
│
├── requirements.txt
├── requirements-train.txt
└── README.md
```

Additional model-training, evaluation, and inference modules will be documented here as they are implemented and integrated.

## Current Implementation Status

- [x] Dataset inspection and validation.
- [x] Exact and near-duplicate detection.
- [x] Anchor-based duplicate grouping.
- [x] Group-aware dataset splitting.
- [x] External test isolation.
- [x] Reproducible manifests and preprocessing reports.
- [x] Automated preprocessing tests.
- [ ] ResNet50 teacher model training.
- [ ] MobileNetV2 student baseline training.
- [ ] Knowledge distillation implementation and evaluation.
- [ ] Final model comparison and performance evaluation.
- [ ] Webcam-based real-time inference integration.

## Future Scope

- Improve recognition under varying lighting conditions and backgrounds.
- Evaluate generalization on independent real-world ASL images.
- Optimize student-model inference for edge devices.
- Explore temporal models for dynamic signs such as J and Z.
- Integrate hand detection, classification, and prediction stabilization into a complete webcam application.

## Contributors

**Group 21 — B.Tech Information Technology**
AISSMS Institute of Information Technology (AISSMS IOIT)

