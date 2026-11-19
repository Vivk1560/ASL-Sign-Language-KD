# Group 21 - Final Project Group 21, Btech IT Aissms IOIT
# Optimization of Sign Language Recognition System using Knowledge Distillation and Deep Learning

# ASL Gesture Recognition with Knowledge Distillation

This repository contains the code and assets for a lightweight, real‑time American Sign Language (ASL) alphabet recognition system.  
The project combines:

- MediaPipe Hands for fast, robust hand detection.
- A ResNet50 teacher network and a MobileNetV2 student network.
- Knowledge Distillation (KD) to compress the model by ~86% while maintaining ~99% accuracy.
- A real‑time webcam demo with overlayed predictions.

---

## 1. Project Overview

The goal of this project is to build an ASL alphabet recognizer that:

- Runs in real time on commodity hardware / edge devices.
- Achieves high accuracy using deep learning.
- Uses knowledge distillation so a small student model can match a large teacher model.

High‑level pipeline:

1. Offline training with KD:
   - Train a ResNet50 teacher on ASL image datasets.
   - Distill knowledge into a compact MobileNetV2 student using temperature‑scaled KD loss.
2. Online inference:
   - Capture webcam frames.
   - Use MediaPipe to detect hand regions.
   - Classify the ASL letter with the student model.
   - Display predictions and confidence scores on screen.

---

## 2. Datasets

This project uses two main public datasets:

1. **ASL Alphabet (Kaggle, grassknoted)**  
   - Clean images of a single hand showing each letter A–Z.  
   - Used as a benchmark for accuracy and for generating augmented training images.  
   - Link: https://www.kaggle.com/datasets/grassknoted/asl-alphabet  

2. **American Sign Language Letters (Roboflow, VOC-style)**  
   - Real‑world images with varying backgrounds.  
   - Includes Pascal VOC XML files with bounding boxes for hands.  
   - Used to train the model to localize and classify hands in cluttered scenes.  
   - Link: https://public.roboflow.com/object-detection/american-sign-language-letters/1  

---

## 3. Architecture Diagram 
<img width="2816" height="1536" alt="Detailed architecture" src="https://github.com/user-attachments/assets/51231f86-07ac-4cf1-9467-bd763964accd" />

---

## 4. Dataset preprocessing (Partition 1)

A modular, reproducible preprocessing pipeline lives in `preprocessing/`. It inspects the raw
Kaggle ASL Alphabet dataset, flags corrupt files, detects exact/near duplicates, builds a
group-aware stratified train/validation/test split, and writes manifests that the Teacher and
Student training scripts must share.

```bat
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-train.txt
python -m preprocessing.main --raw-dir D:\path\to\asl_dataset
```

See `PARTITION1_IMPLEMENTATION_REPORT.md` for details, options and current verification status.
Raw and processed data are git-ignored (`data/`). The default class set is the existing 26 classes (A-Z);
`del`, `nothing` and `space` are recorded in metadata and can be enabled with `--class-set 29`.

