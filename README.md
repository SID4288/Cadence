# Cadence : Nepali Folk Music Classification

State-of-the-art approach for classifying six Nepali folk music genres using domain-specific audio feature engineering and a custom residual CNN.

**Genres:** `tamang_selo`, `deuda`, `bhajan`, `newari`, `tharu`, `lok_dohori`

---

## Core Concepts

### 1. Constant-Q Transform (CQT)

Mel spectrograms use linearly spaced frequency bins, which do not align with how pitch works in music. The **Constant-Q Transform** uses logarithmically spaced bins so each octave contains the same number of bins — matching musical semitones and octaves.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `N_BINS` | 84 | 7 octaves × 12 semitones |
| `BINS_PER_OCTAVE` | 12 | One bin per semitone |
| `HOP_LENGTH` | 512 | Time resolution for ~30 s clips |
| `SAMPLE_RATE` | 22050 Hz | Standard librosa rate |

For a 30-second clip, CQT produces a time–frequency map of shape **(84, ~1292)** — 84 frequency bins across ~1292 time frames.

### 2. Harmonic–Percussive Source Separation (HPSS)

Raw audio mixes melody/harmony and rhythm/percussion into a single waveform. **HPSS** (`librosa.effects.hpss`) decomposes the signal into:

- **Harmonic** — sustained tonal content (melody, vocals, drones)
- **Percussive** — transient rhythmic content (drums, plucks, beats)

Each component is transformed separately, giving the model physically meaningful channels instead of duplicating a single spectrogram into fake RGB channels (as the baseline does).

### 3. Two-Channel Input Representation

The final feature tensor has shape **(2, 84, 1292)**:

| Channel | Content |
|---------|---------|
| 0 | Normalized harmonic CQT |
| 1 | Normalized percussive CQT |

**Normalization pipeline (per channel):**
1. Compute magnitude CQT
2. Convert to decibel scale (`librosa.amplitude_to_db`)
3. Per-channel min–max normalize to [0, 1]

This log-scale + per-channel normalization produces stable neural network inputs and handles varying loudness across recordings.

### 4. Precomputed Features

CQT and HPSS are CPU-bound (librosa/numpy). To avoid recomputing them every epoch:

1. Raw `.wav` files live in `dataset/splits/{train,test}/<genre>/`
2. Preprocessing writes `.npy` tensors to `dataset/processed_cqt/{train,test}/<genre>/`
3. Training loads precomputed arrays; on-the-fly computation is used only as a fallback

This shifts the bottleneck from CPU feature extraction to GPU model training.

### 5. SpecAugment

During training, random frequency and time masks are applied to spectrograms (zeroing rectangular regions). This regularizes the model against overfitting to specific spectral patterns and improves generalization, standard practice for spectrogram-based audio classifiers.

### 6. Chunked Inference with Confidence Scoring

Long audio files are split into overlapping 30-second chunks. Each chunk is classified independently; predictions are aggregated with consistency, entropy, and confidence thresholds to flag unreliable results.

---

## Architecture

### Overview

```
Raw Audio (.wav)
      │
      ▼
┌─────────────────┐
│  HPSS Split     │  harmonic │ percussive
└────────┬────────┘
         ▼
┌─────────────────┐
│  CQT (per src)  │  → dB → normalize [0,1]
└────────┬────────┘
         ▼
   (2, 84, 1292)  ──►  FolkMusicCQTResNet  ──►  6-class logits
```

### FolkMusicCQTResNet

A custom 2D residual network (`model.py`) trained from scratch — no ImageNet transfer learning. It is shaped for the high aspect ratio of spectrograms (84 frequency bins vs. 1292 time frames).

**Input:** `(batch, 2, 84, 1292)`
**Output:** `(batch, 6)` genre logits

### ResBlock2D

Each residual block contains:

- `Conv2d(3×3)` → `BatchNorm2d` → `ReLU`
- `Dropout2d`
- `Conv2d(3×3)` → `BatchNorm2d`
- Skip connection (with 1×1 conv + BN when channels or stride change)
- Final `ReLU`

BatchNorm and dropout are integrated into every block (not only the classifier head), improving training stability and generalization.

### Layer Hierarchy & Shape Progression

| Stage | Operation | Output Shape (H × W) |
|-------|-----------|----------------------|
| Input | — | 84 × 1292 |
| `conv_init` | Conv2d 2→32, BN, ReLU | 84 × 1292 |
| `layer1` + `pool1` | ResBlock 32→64, MaxPool 2×2 | 42 × 646 |
| `layer2` + `pool2` | ResBlock 64→128, MaxPool 2×2 | 21 × 323 |
| `layer3` + `pool3` | ResBlock 128→256, MaxPool **1×2** | 21 × 161 |
| `layer4` + `pool4` | ResBlock 256→256, MaxPool 2×2 | 10 × 80 |
| `layer5` | ResBlock 256→512 | 10 × 80 |
| `global_pool` | AdaptiveAvgPool2d(1, 1) | 1 × 1 |
| `fc` | Linear 512→128→6 | 6 |

The asymmetric **1×2 pool** at `pool3` downsamples time more aggressively than frequency, reflecting that spectrograms are much wider in time than in frequency.

### Classification Head

```
AdaptiveAvgPool2d(1,1)
  → Flatten
  → Linear(512, 128) → BatchNorm1d → ReLU → Dropout
  → Linear(128, num_classes)
```

---

## Training Configuration

| Setting | Value |
|---------|-------|
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Loss | CrossEntropyLoss (label smoothing=0.1) |
| Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Augmentation | SpecAugment (freq + time masking) |
| Batch size | 16 |
| Epochs | 25 |
| Device | CUDA when available (`pin_memory`, `num_workers`) |

---

## Pipeline

```
preprocess.py  →  train_sota.py  →  evaluate_sota.py  →  predict_sota.py
     │                  │                   │                    │
  .wav → .npy      train model         metrics & plots      classify audio
```

| Script | Role |
|--------|------|
| `utils.py` | Shared config, paths, genre labels |
| `preprocess.py` | HPSS + CQT feature extraction, saves `.npy` |
| `model.py` | `FolkMusicCQTResNet` and `ResBlock2D` |
| `dataset.py` | PyTorch `Dataset` / `DataLoader` with precomputed fallback |
| `train_sota.py` | Training loop with SpecAugment |
| `evaluate_sota.py` | Accuracy, F1, confusion matrix, confidence analysis |
| `predict_sota.py` | Chunked inference with reliability scoring |
| `integrate_workflow.py` | End-to-end automation |
| `notebook.ipynb` | Standalone Kaggle notebook (full pipeline + GPU) |

---

## Project Layout

```
Cadence/
├── utils.py
├── preprocess.py
├── model.py
├── dataset.py
├── train_sota.py
├── evaluate_sota.py
├── predict_sota.py
├── integrate_workflow.py
├── models/                  # saved checkpoints
├── results/                 # plots and evaluation reports
└── dataset/
    ├── splits/              # raw .wav (train / test / genre)
    └── processed_cqt/       # precomputed .npy features
```

---

## Quick Start

```bash
# 1. Precompute CQT features (CPU-bound)
python preprocess.py

# 2. Train the model (GPU)
python train_sota.py

# 3. Evaluate
python evaluate_sota.py

# 4. Predict a file
python predict_sota.py --file path/to/audio.wav --show-chunks
```


---

## Design Rationale vs. Baseline

| Aspect | Baseline (Cadence) | SOTA (this approach) |
|--------|--------------------|----------------------|
| Representation | Mel spectrogram | CQT + HPSS |
| Channels | 3 (repeated Mel) | 2 (harmonic + percussive) |
| Model | Pretrained EfficientNet-B0 | Custom FolkMusicCQTResNet |
| Philosophy | Image transfer learning | Domain-specific audio CNN |
| Features | On-the-fly | Precomputed with fallback |
| Augmentation | None | SpecAugment |

See `comparisons.md` for a detailed side-by-side analysis.
