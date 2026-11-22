# ASL Gesture Recognition Using Knowledge Distillation

**Group 21 | B.Tech Information Technology | AISSMS IOIT**

A deep learning-based American Sign Language (ASL) alphabet recognition project focused on building a lightweight image-classification model using Knowledge Distillation.

The project uses a **ResNet50 teacher network** to guide the training of a smaller **MobileNetV2 student network**. The objective is to retain useful classification knowledge while reducing the student's model complexity.

> **Project scope:** This is a static-image fingerspelling alphabet classifier. It does not perform continuous ASL sentence translation or temporal recognition of dynamic signs.

---

## Project Overview

The objective is to develop an ASL alphabet recognition system that classifies hand gesture images while exploring a more computationally efficient student model.

The project combines:

- **ResNet50:** Teacher network trained to classify ASL alphabet images.
- **MobileNetV2:** Lightweight student network trained using Knowledge Distillation.
- **Knowledge Distillation (KD):** Transfers predictive knowledge from the teacher to the student through temperature-scaled probability distributions.
- **PyTorch:** Model development, training, evaluation, and inference.
- **OpenCV:** Image and webcam frame handling.
- **MediaPipe Hands (optional):** Hand detection and cropping when a compatible installation is available.

### System Workflow

1. **Dataset preparation**

   - Inspect and validate ASL images.
   - Detect exact and near-duplicate images.
   - Group duplicate images to reduce data leakage across splits.
   - Create training, validation, and internal test datasets.
   - Generate reproducible dataset manifests.

2. **Teacher model training**

   - Train a ResNet50 teacher model on the prepared dataset.
   - Evaluate its classification performance.
   - Save the best-performing checkpoint.

3. **Knowledge Distillation**

   - Train a MobileNetV2 student model.
   - Use the frozen ResNet50 teacher to generate softened probability distributions.
   - Combine ground-truth classification loss with temperature-scaled distillation loss.
   - Evaluate the trained student model against the teacher.

4. **Inference and demonstration**

   - Load the trained student checkpoint.
   - Perform single-image predictions.
   - Optionally capture webcam frames using OpenCV.
   - Use MediaPipe for hand detection when available, or a centre-region fallback.
   - Display predicted classes and confidence scores.

---

## Technologies Used

| Category                 | Technologies                 |
| ------------------------ | ---------------------------- |
| Programming Language     | Python 3.11                  |
| Deep Learning            | PyTorch                      |
| Computer Vision          | OpenCV, MediaPipe (optional) |
| Data Processing          | NumPy, Pandas                |
| Image Processing         | Pillow, SciPy                |
| Machine Learning Metrics | Scikit-learn                 |
| Visualization            | Matplotlib                   |
| Testing                  | Pytest                       |
| Version Control          | Git, GitHub                  |

---

## Datasets

### 1. ASL Alphabet Dataset — Kaggle

**Source:** [ASL Alphabet Dataset by grassknoted](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

The primary dataset contains approximately 87,000 training images distributed across 29 classes:

- 26 English alphabet letters (`A`–`Z`)
- `del`
- `nothing`
- `space`

This dataset is used for preprocessing, model training, validation, and internal testing.

### 2. American Sign Language Letters — Roboflow

**Source:** [American Sign Language Letters Dataset](https://public.roboflow.com/object-detection/american-sign-language-letters/1)

This dataset contains real-world images with varying backgrounds and Pascal VOC-style annotations, including hand bounding boxes.

It is listed as a potential source for future experiments involving hand localization and recognition in less controlled environments.

**Note:** The Roboflow dataset is not part of the reported teacher or student training and evaluation results.

---

## Model Architecture

The project follows a teacher-student architecture.

### Teacher Network — ResNet50

- A deep convolutional neural network.
- Trained on the prepared ASL image dataset.
- Produces class predictions and probability distributions.
- Acts as the teacher during the distillation process.

### Student Network — MobileNetV2

- A lightweight convolutional neural network.
- Designed for lower computational requirements.
- Learns from both ground-truth labels and the teacher's predictions.
- Used for the project's inference pipeline.

### Knowledge Distillation

This project uses **response-based Knowledge Distillation**. It transfers knowledge through the teacher's softened output probabilities rather than directly transferring intermediate feature maps.

During training:

1. The same input image is passed through the frozen ResNet50 teacher and the trainable MobileNetV2 student.
2. The teacher generates softened class probabilities using a temperature parameter.
3. The student generates its own softened predictions.
4. The student is trained using a combination of:
   - **Cross-Entropy Loss:** Measures the difference between student predictions and ground-truth labels.
   - **Knowledge Distillation Loss:** Uses KL divergence to compare the teacher's and student's softened probability distributions.

The combined objective is:

[
L\_{\text{total}} =
\alpha L\_{\text{CE}} +
(1-\alpha)T^2 L\_{\text{KD}}
]

Where:

- (L\_{\text{CE}}) = ground-truth classification loss.
- (L\_{\text{KD}}) = KL-divergence-based distillation loss.
- (T) = temperature parameter.
- (\alpha) = weight assigned to the classification loss.

The reported student configuration uses **temperature (T=4)** and **(\alpha=0.5)**.

The teacher remains frozen during student training. Its predictions guide the student, while the student model's parameters are updated through backpropagation.

### Architecture Diagram



---

## Dataset Preprocessing

A modular preprocessing pipeline is implemented in the `preprocessing/` directory.

Its responsibilities include:

- Dataset discovery and structural inspection.
- Image validation and corrupt-file handling.
- Exact duplicate detection using SHA-256.
- Near-duplicate detection using perceptual difference hashing (dHash).
- Anchor-based duplicate grouping to limit excessive transitive chaining.
- Group-aware training, validation, and internal test splitting.
- External test image isolation.
- Label generation and reproducible CSV manifests.
- Training-only image augmentation.
- Dataset validation and preprocessing reports.

### Dataset Split

The verified preprocessing run produced the following distribution:

| Split         |     Images | Percentage |
| ------------- | ---------: | ---------: |
| Training      |     60,872 |        70% |
| Validation    |     13,050 |        15% |
| Internal Test |     13,050 |        15% |
| **Total**     | **86,972** |   **100%** |

The 28 official Kaggle external test images were checked against the internal dataset using SHA-256 hashes. Their exact internal copies were excluded before duplicate grouping and splitting.

The official external test folder does not contain a `del_test.jpg`. This is an existing dataset limitation.

The external test set is kept separate from training and is intended for an additional smoke test. With only 28 images, it is not large enough to serve as a comprehensive performance benchmark.

### Important Dataset Considerations

- Detected duplicate groups are kept within a single split to reduce data leakage.
- Only exact copies of official external test images are excluded. Near-duplicate external images may still exist in the internal dataset.
- Anchor-based grouping limits chaining but cannot guarantee that every visually similar neighbouring image belongs to the same group.
- The letters **J** and **Z** involve motion. A single-frame image classifier cannot fully recognise their temporal movement.

---

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

The preprocessing pipeline also supports a custom dataset location through its `--raw-dir` argument.

### 5. Run preprocessing

```powershell
python -m preprocessing.main --class-set 29 --force --require-external-test
```

This command processes all 29 classes, regenerates preprocessing outputs, and requires the official external test folder to be available for isolation verification.

Generated manifests, metadata, and reports are written under `data/processed/`.

---

## Partition 4: Inference, Comparison, Benchmarking and Webcam

The deployment implementation is available in the `deployment/` directory. It reuses the teacher and student model builders, checkpoint loaders, and evaluation transforms.

For detailed instructions, see the [Deployment Guide](deployment/DEPLOYMENT_GUIDE.md).

> **Checkpoint availability:** Model checkpoints are local artifacts and are not included in Git. The recovered student checkpoint is expected at `student_output/student_mobilenetv2_kd_best.pth`. Commands accept explicit checkpoint paths. Default paths can also be configured through the `ASL_STUDENT_CHECKPOINT` and `ASL_TEACHER_CHECKPOINT` environment variables.
>
> The files `model_weights/teacher_best_aug.pth` and `model_weights/student_best_aug.pth` are legacy files from before Partitions 2 and 3. They are not the checkpoints used by the current deployment pipeline.

### Single-Image Inference

No camera or MediaPipe installation is required.

```powershell
python -m deployment.infer path\to\image.jpg --checkpoint student_output\student_mobilenetv2_kd_best.pth
```

Predict multiple images and return JSON:

```powershell
python -m deployment.infer a.jpg b.jpg --checkpoint student_output\student_mobilenetv2_kd_best.pth --json
```

Inspect checkpoint metadata without performing a prediction:

```powershell
python -m deployment.infer --info --checkpoint student_output\student_mobilenetv2_kd_best.pth
```

### Webcam Inference

```powershell
python -m deployment.webcam --checkpoint student_output\student_mobilenetv2_kd_best.pth
```

**Keyboard controls:**

- `q` or `Esc` — Quit.
- `m` — Toggle display-only mirroring.

**Available options:**

- `--camera 1` — Select another camera or provide a video file path.
- `--hand-detector auto|mediapipe|roi` — Select the hand detection method.
- `--threshold 0.6` — Set the confidence threshold.
- `--smooth 5` — Configure prediction smoothing.
- `--no-mirror` — Disable preview mirroring.
- `--no-window` — Disable the display window.
- `--max-frames N` — Limit the number of processed frames.

**Hand detection behaviour:**

- MediaPipe is optional. The documented installation is `pip install mediapipe==0.10.14`.
- When a compatible MediaPipe installation is available, the application can detect and crop the hand region.
- If MediaPipe is unavailable or unusable, the application can use a fixed **centre-ROI fallback**. In this mode, the user must position their hand inside the displayed region; automatic hand detection does not take place.
- The application treats "no hand detected" as a separate state. It is not the same as the model predicting the `nothing` class.
- The model receives the raw, unmirrored frame. Mirroring affects only the preview.
- On Windows, grant camera permission to the terminal/Python and close other applications that may be using the camera.

**Verification note:** The webcam pipeline is implemented and covered by automated tests. Physical camera verification with the recovered student checkpoint remains pending.

### Teacher vs Student Comparison

Run the comparison on the internal test split:

```powershell
python -m deployment.compare --student-checkpoint student_output\student_mobilenetv2_kd_best.pth --teacher-checkpoint model_weights\teacher_resnet50_best.pth --split test
```

The comparison evaluates both models using the same internal evaluation split.

### Model Benchmarking

Run the benchmark on CPU and, when available, CUDA:

```powershell
python -m deployment.benchmark --student-checkpoint student_output\student_mobilenetv2_kd_best.pth --teacher-checkpoint model_weights\teacher_resnet50_best.pth --devices cpu cuda
```

Both comparison and benchmarking generate reports under `deployment_reports/`. The benchmark measures model inference performance on the selected device; its results are hardware-dependent and should not be interpreted as full webcam-pipeline performance.

**Important:** Actual comparison and target-hardware benchmark results should be reported only after running these commands with the recovered checkpoints on the relevant machine. No unmeasured latency or throughput figures are claimed here.

---

## Results

The following figures are from the reported student training and internal evaluation run.

These are internal-split results. They are **not evidence of real-world generalisation or webcam accuracy**.

| Metric                                     |                         Reported Result |
| ------------------------------------------ | --------------------------------------: |
| Student best epoch                         |                                       7 |
| Early stopping patience                    | 5 epochs without validation improvement |
| Student validation accuracy                |                                  0.9999 |
| Student internal-test accuracy             |                                  0.9995 |
| Student internal-test macro-F1             |                                  0.9995 |
| Teacher internal-test accuracy (reference) |                                  0.9998 |
| Student parameters                         |                    Approx. 2.26 million |
| Teacher parameters                         |                   Approx. 23.57 million |
| Reduction in parameter count               |                          Approx. 90.41% |
| Student ImageNet pretrained weights        |                                Not used |

The parameter reduction is based on model parameter counts. It does not, by itself, establish a 90.41% reduction in latency, memory usage, or power consumption.

Run the teacher-student comparison command above to generate a fresh comparison from the actual checkpoints.

---

## Limitations

- The model classifies **individual static images**.
- **J** and **Z** are dynamic signs. A single-frame classifier cannot capture their complete movement.
- This project does not perform continuous ASL sentence translation.
- The 29 output classes represent 26 letters and the additional `del`, `nothing`, and `space` classes.
- Internal-test performance comes from the same dataset distribution used in the project workflow.
- Robustness to different backgrounds, lighting conditions, skin tones, hands, and cameras has not yet been comprehensively evaluated.
- The external test set contains only 28 images and is not sufficient for a broad generalisation claim.
- Real webcam performance and target-hardware latency must be measured separately.

---

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
├── teacher_model/              # Partition 2: ResNet50 teacher
│
├── student_model/              # Partition 3: MobileNetV2 + KD
│
├── deployment/                 # Partition 4
│   ├── predictor.py
│   ├── infer.py
│   ├── webcam.py
│   ├── compare.py
│   ├── benchmark.py
│   └── DEPLOYMENT_GUIDE.md
│
├── tests/
│
├── asl_mediapipe_final.py      # Legacy prototype
│
├── requirements.txt
├── requirements-train.txt
└── README.md
```

`asl_mediapipe_final.py` is a **legacy prototype** from before Partitions 2 and 3. It defines separate 26-class model wrappers, loads older raw `state_dict` files, and duplicates preprocessing logic. It is retained for reference. Use the current `deployment/` pipeline instead.

---

## Current Implementation Status

- [x] Dataset inspection and validation.
- [x] Exact and near-duplicate detection.
- [x] Anchor-based duplicate grouping.
- [x] Group-aware dataset splitting.
- [x] External test isolation.
- [x] Reproducible manifests and preprocessing reports.
- [x] Automated preprocessing tests.
- [x] ResNet50 teacher model training (Partition 2).
- [x] MobileNetV2 student training with Knowledge Distillation (Partition 3).
- [x] Student inference implementation.
- [x] Teacher-student comparison implementation.
- [x] Model benchmarking implementation.
- [x] Webcam inference implementation.
- [x] Automated deployment tests.
- [x] Teacher-student comparison executed on the recovered checkpoints.
- [x] Benchmarks executed on target hardware.
- [x] Webcam application physically verified with the recovered student checkpoint.

---

## Future Scope

- Improve recognition under varying lighting conditions and backgrounds.
- Evaluate generalisation on independent real-world ASL images.
- Measure and optimise inference performance on edge devices.
- Explore temporal models for dynamic signs such as J and Z.
- Improve hand detection and prediction stabilisation.
- Extend the system towards continuous gesture recognition.

---

## Contributors

**Group 21 — B.Tech Information Technology**
AISSMS Institute of Information Technology (AISSMS IOIT)
