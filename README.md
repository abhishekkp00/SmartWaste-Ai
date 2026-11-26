# 🗑️ SmartWaste AI
### Deep Learning-Based Waste Classification and Recycling Recommendation System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> An academic deep learning project demonstrating CNN and Transfer Learning using PyTorch for automated waste image classification with recycling recommendations.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Waste Categories](#-waste-categories)
- [System Architecture](#-system-architecture)
- [Dataset](#-dataset)
- [Models](#-models)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Development Phases](#-development-phases)
- [Technology Stack](#-technology-stack)
- [Academic Documentation](#-academic-documentation)

---

## 🎯 Project Overview

SmartWaste AI uses computer vision and deep learning to automatically classify waste images into 6 recyclable categories and provide appropriate disposal recommendations. 

**Key features:**
- Custom CNN architecture built from scratch using PyTorch
- ResNet18 transfer learning (frozen + fine-tuned comparison)
- Grad-CAM explainability heatmaps
- FastAPI REST backend with checkpoint-aware single and batch inference
- React + Vite responsive dashboard with live API integration
- Prediction history with SQLite storage
- Complete training pipeline with early stopping and checkpointing

---

## 🏷️ Waste Categories

| Index | Class | Display Name |
|-------|-------|-------------|
| 0 | cardboard | Cardboard |
| 1 | glass | Glass |
| 2 | metal | Metal |
| 3 | paper | Paper |
| 4 | plastic | Plastic |
| 5 | trash | Trash / General Waste |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        React Frontend                         │
│   Dashboard │ Classify │ Batch │ Performance │ History        │
└──────────────────┬───────────────────────────────────────────┘
                   │ HTTP (Axios)
┌──────────────────▼───────────────────────────────────────────┐
│                     FastAPI Backend                           │
│  POST /api/predict   GET /api/analytics   GET /api/health    │
│                                                               │
│  ┌─────────────────┐    ┌────────────────────────────────┐   │
│  │  ModelService   │    │  RecommendationEngine          │   │
│  │  (Singleton)    │    │  (Rule-based, 6-class lookup)  │   │
│  └────────┬────────┘    └────────────────────────────────┘   │
│           │                                                   │
│  ┌────────▼────────┐    ┌────────────────────────────────┐   │
│  │  WastePredictor │    │  SQLite (aiosqlite)            │   │
│  │  (PyTorch)      │    │  prediction_logs table         │   │
│  └─────────────────┘    └────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────────────┐
│                     ML Pipeline (PyTorch)                     │
│                                                               │
│   CustomCNN (from scratch)    ResNet18 (Transfer Learning)   │
│        5 conv blocks              Stage A: frozen             │
│        ~6.5M parameters          Stage B: fine-tuned          │
│                                                               │
│   Training: CrossEntropyLoss (weighted) + AdamW               │
│   Scheduler: CosineAnnealingLR                               │
│   Regularisation: EarlyStopping (patience=7)                  │
│   Explainability: Grad-CAM heatmaps (ResNet18)                │
└──────────────────────────────────────────────────────────────┘
```

---

## 📦 Dataset

**TrashNet** by Gary Thung & Mindy Yang (Stanford University)
- **License**: MIT
- **Total images**: ~2,527
- **Split**: 70% train / 15% val / 15% test (stratified, seed=42)

| Class | Total Images | % of Dataset |
|-------|-------------|-------------|
| Paper | ~594 | 23.5% |
| Glass | ~501 | 19.8% |
| Plastic | ~482 | 19.1% |
| Metal | ~410 | 16.2% |
| Cardboard | ~403 | 15.9% |
| **Trash** | **~137** | **5.4%** ⚠️ |

> ⚠️ **Class Imbalance**: The `trash` class is severely underrepresented (5.4%).
> This is handled via `CrossEntropyLoss(weight=...)` and `WeightedRandomSampler`.

---

## 🧠 Models

### Model 1: Custom CNN (from scratch)

```
Input (3, 224, 224)
  → Block 1: Conv2d(3→32)   + BN + ReLU + MaxPool  → (32, 112, 112)
  → Block 2: Conv2d(32→64)  + BN + ReLU + MaxPool  → (64, 56, 56)
  → Block 3: Conv2d(64→128) + BN + ReLU + MaxPool  → (128, 28, 28)
  → Block 4: Conv2d(128→256)+ BN + ReLU + MaxPool  → (256, 14, 14)
  → Block 5: Conv2d(256→512)+ BN + ReLU + MaxPool  → (512, 7, 7)
  → AdaptiveAvgPool2d(1×1)                          → (512)
  → FC(512→256) + ReLU + Dropout(0.5)
  → FC(256→128) + ReLU + Dropout(0.3)
  → FC(128→6)                                       → 6 logits
```

### Model 2: ResNet18 Transfer Learning

- **Stage A** (head only, backbone frozen): Replace `model.fc` with `Linear(512→256→6)`. Train with `lr=1e-3`.
- **Stage B** (fine-tuning): Unfreeze `layer3` and `layer4`. Train backbone at `lr=1e-4`, head at `lr=1e-3`.
- **Grad-CAM**: Applied on the final conv layer to generate class activation maps.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip
- For training: Google Colab (recommended) or a CUDA-capable GPU

### 1. Clone and Set Up

```bash
git clone https://github.com/YOUR
cd SmartWaste_Ai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Run scaffold (creates all directories and __init__.py files)
python setup_project.py

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # for development/testing
```

### 2. Download Dataset

```bash
# Automatic download (uses gdown for Google Drive):
python dataset/download_trashnet.py

# If you have the ZIP already:
python dataset/download_trashnet.py --zip /path/to/dataset-resized.zip
```

### 3. Verify Dataset

```bash
python dataset/verify_dataset.py
# Expected: ✅ ALL CHECKS PASSED
```

### 4. Run Phase 2 EDA and Validation

```bash
# Generate EDA charts and the class-imbalance report
python ml/eda.py

# Run the complete PyTorch Phase 2 validation suite
python tests/validate_phase2.py

# Optional lightweight local checks
python tests/validate_phase2_local.py
```

Phase 2 uses `ml/config.py` as the single source of truth for image size, ImageNet normalization values, split ratios,
class names, and batch settings.

**Actual dataset summary:**

| Class | Train | Validation | Test | Total |
|-------|------:|-----------:|-----:|------:|
| Cardboard | 282 | 60 | 61 | 403 |
| Glass | 350 | 75 | 76 | 501 |
| Metal | 287 | 61 | 62 | 410 |
| Paper | 415 | 89 | 90 | 594 |
| Plastic | 337 | 72 | 73 | 482 |
| Trash | 95 | 20 | 22 | 137 |
| **Total** | **1,766** | **377** | **384** | **2,527** |

The `trash` class is approximately 4.3× less represented than `paper`. Phase 2 computes inverse-frequency training weights,
including `trash = 3.0982`, from training labels only. EDA and validation outputs are written under `results/eda/`.

**DataLoader worker settings:**

- Local Ubuntu or debugging: `num_workers=0`
- Local Ubuntu with a performant dataset filesystem: `num_workers=2` or `4`
- Google Colab: `num_workers=2` is a practical starting point
- Set `pin_memory=True` for CUDA; the DataLoader does not gain a benefit on CPU-only runs

**Preprocessing summary:**

- Training: `RandomResizedCrop`, `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`, optional grayscale, `ToTensor`, and ImageNet normalization
- Validation/test: deterministic resize to 256, center crop to 224, `ToTensor`, and the same ImageNet normalization
- Output tensor contract: `[Batch, 3, 224, 224]` with `float32` images and `int64` labels

### 5. Run Phase 3 Custom CNN Validation

```bash
# Smoke test the Custom CNN module
python ml/custom_cnn.py

# Validate architecture, gradients, freeze/unfreeze, and real batches
python tests/validate_phase3.py
```

The Custom CNN maps `[Batch, 3, 224, 224]` inputs through five convolution blocks to 512 features and six logits:

```text
224 -> 112 -> 56 -> 28 -> 14 -> 7 -> 512 features -> 6 logits
```

The implementation is in `ml/custom_cnn.py`. It contains 4,879,206 trainable parameters and supports CPU, CUDA, feature freezing, and feature extraction for forward inspection. Model training starts in a later phase.

### 6. Run Phase 4 ResNet18 and Grad-CAM Validation

```bash
# Smoke test Stage A and Stage B
python ml/resnet_model.py

# Validate architecture freezing, real batches, and Grad-CAM
python tests/validate_phase4.py
```

Phase 4 provides two transfer-learning configurations:

- **Stage A**: pretrained ResNet18 backbone frozen; only the six-class classifier head is trainable.
- **Stage B**: `layer3`, `layer4`, and the classifier head are trainable; `layer1` and `layer2` remain frozen.

Grad-CAM is generated from `backbone.layer4[-1]` and saved to `results/eda/phase4_gradcam_heatmap.png`. Model training remains in Phase 5.

### 7. Run Phase 5 Training Engine Validation

```bash
# Run metrics and training-engine CPU smoke validation
python tests/validate_phase5.py
```

The reusable engine is implemented in `ml/train.py` and `ml/metrics.py`. It supports weighted `CrossEntropyLoss`, optional `WeightedRandomSampler`, AdamW, cosine scheduling, early stopping, best-checkpoint saving, and training history.

This phase validates the engine only; actual 30/15-epoch experiments remain separate model-training work.

### 8. Run Phase 6 Evaluation Validation

```bash
python tests/validate_phase6.py
```

Phase 6 evaluation utilities are implemented in `ml/evaluate.py`. They provide test-set inference, macro and per-class metrics, confusion matrices, JSON reports, training-history plots, and model-comparison plots.

Generated validation artifacts include:

- `results/confusion_matrix/phase6_validation_confusion_matrix.png`
- `results/confusion_matrix/phase6_validation_counts.png`
- `results/reports/phase6_validation_report.json`
- `results/graphs/training_history.png`
- `results/graphs/model_comparison.png`

### 9. Run Phase 7 Inference Validation

```bash
python tests/validate_phase7.py
```

Phase 7 introduces `ml/predict.py`, a reusable `WastePredictor` inference engine with:

- model loading (`custom_cnn`, `resnet18_stage_a`, `resnet18_stage_b`)
- top-k prediction output with calibrated confidence scores
- low-confidence warning when below `cfg.inference.CONFIDENCE_THRESHOLD`
- class-wise recycling recommendation lookup for all six TrashNet classes

### 10. Run Phase 8 Backend Validation

```bash
python tests/validate_phase8.py
```

Phase 8 implements the FastAPI backend in `backend/app/main.py` with production-ready route groups:

- `GET /api/health`
- `POST /api/predict`
- `POST /api/predict/batch`
- `GET /api/analytics`

It includes async SQLite logging (`prediction_logs`) and aggregate analytics for recent predictions.

### 11. Train on Google Colab

Open all available phase notebooks from `01_phase1_dataset_setup.ipynb` through `07_phase7_inference_engine.ipynb` in Colab.

### 12. Run Backend

```bash
# Run from the repository root (not from backend/)
python -m uvicorn backend.app.main:app --reload --port 8000
# API docs: http://localhost:8000/docs
# Health:  http://localhost:8000/api/health
```

### 13. Run Frontend

```bash
cd frontend
npm install
npm run dev
# App: http://localhost:5173
```

---

## 📁 Project Structure

```
SmartWaste_Ai/
│
├── dataset/
│   ├── download_trashnet.py    # Reproducible dataset download
│   ├── verify_dataset.py       # 5-point integrity verification
│   ├── raw/                    # Downloaded ZIP + extracted files
│   └── processed/              # train / val / test splits
│       ├── train/{class}/
│       ├── val/{class}/
│       └── test/{class}/
│
├── ml/
│   ├── config.py               # All hyperparameters and paths
│   ├── utils.py                # Seed, device, logging, checkpointing
│   ├── preprocessing.py        # Train + val/test transforms (Phase 2)
│   ├── dataset.py              # WasteDataset + DataLoaders (Phase 2)
│   ├── custom_cnn.py           # CNN from scratch (Phase 3)
│   ├── resnet_model.py         # ResNet18 transfer learning (Phase 4)
│   ├── gradcam.py              # Grad-CAM explainability (Phase 4)
│   ├── train.py                # Trainer class + training loop (Phase 5)
│   ├── metrics.py              # Accuracy, precision, recall, F1 (Phase 5)
│   ├── evaluate.py             # Evaluation + plots (Phase 6)
│   ├── predict.py              # WastePredictor inference engine (Phase 7)
│   └── recommendations.py      # Rule-based disposal guidance (Phase 7)
│
├── models/                     # Saved .pth checkpoints (gitignored)
├── results/                    # Training graphs, confusion matrices, reports
├── notebooks/                  # Jupyter/Colab notebooks per phase
│
├── backend/app/
│   ├── main.py                 # FastAPI application (Phase 8)
│   ├── routes/                 # predict, batch, analytics, health
│   ├── schemas/                # Pydantic request/response models
│   ├── services/               # ModelService (singleton), analytics service
│   ├── database/               # SQLAlchemy ORM, CRUD, session
│   └── utils/                  # Image validation, logging
│
├── frontend/                   # React + Vite + Tailwind (Phase 9)
├── tests/                      # pytest test suite (Phase 10)
│
├── setup_project.py            # One-shot scaffold creator
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 📅 Development Phases

| Phase | Description | Status |
|-------|-------------|--------|
| **1** | Architecture, dataset download, verification | ✅ Complete |
| **2** | EDA, preprocessing transforms, Dataset class | ✅ Complete |
| **3** | Custom CNN from scratch | ✅ Complete |
| **4** | ResNet18 + Grad-CAM | ✅ Complete |
| **5** | Training engine (weighted loss + sampler) | ✅ Complete |
| **6** | Evaluation, confusion matrix, comparison | ✅ Complete |
| **7** | Inference engine + recommendation engine | ✅ Complete |
| **8** | FastAPI backend + all REST endpoints | ✅ Complete |
| **9** | React frontend dashboard + API integration | ✅ Implemented — local smoke test pending |
| **10** | Tests, Docker, documentation | ⬜ Pending |

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | PyTorch 2.1+, torchvision |
| Explainability | pytorch-grad-cam |
| Data Science | NumPy, Pandas, scikit-learn |
| Visualisation | Matplotlib, Seaborn |
| Image Processing | Pillow |
| Backend | FastAPI, Uvicorn |
| Database | SQLite (async via aiosqlite + SQLAlchemy 2.0) |
| Validation | Pydantic v2 |
| Frontend | React, Vite, responsive CSS, Recharts, Axios |
| Testing | pytest
| Deployment | Docker, Docker Compose (Phase 10) |

---

## 📄 Academic Documentation

Full documentation covering abstract, literature review, methodology, results, and system architecture is generated in Phase 10.

**Key evaluation metrics reported:**
- Accuracy, Precision, Recall, F1-score (per class + macro avg)
- Confusion matrix heatmap
- Training/validation loss and accuracy curves
- Custom CNN vs ResNet18 (frozen) vs ResNet18 (fine-tuned) comparison

---

## 📜 License

This project is released under the **MIT License**.

The TrashNet dataset is also MIT licensed.  
© Gary Thung & Mindy Yang, Stanford University.

---

## 🤝 Acknowledgements

- [TrashNet](https://github.com/garythung/trashnet) — Gary Thung & Mindy Yang
- [PyTorch](https://pytorch.org) — The PyTorch team
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) — Jacob Gildenblat
