# Cadence : Nepali Folk Music Classification

## Approach 1 : HPSS + CQT (`cadence.ipynb`)

State-of-the-art approach for classifying six Nepali folk music genres using domain-specific audio feature engineering and a custom residual CNN.

**Genres:** `tamang_selo`, `deuda`, `bhajan`, `newari`, `tharu`, `lok_dohori`

---

## Core Concepts

### 1. Constant-Q Transform (CQT)

Mel spectrograms use linearly spaced frequency bins, which do not align with how pitch works in music. The **Constant-Q Transform** uses logarithmically spaced bins so each octave contains the same number of bins — matching musical semitones and octaves.

| Parameter         | Value    | Rationale                       |
| ----------------- | -------- | ------------------------------- |
| `N_BINS`          | 84       | 7 octaves × 12 semitones        |
| `BINS_PER_OCTAVE` | 12       | One bin per semitone            |
| `HOP_LENGTH`      | 512      | Time resolution for ~30 s clips |
| `SAMPLE_RATE`     | 22050 Hz | Standard librosa rate           |

For a 30-second clip, CQT produces a time–frequency map of shape **(84, ~1292)**, 84 frequency bins across ~1292 time frames.

### 2. Harmonic–Percussive Source Separation (HPSS)

Raw audio mixes melody/harmony and rhythm/percussion into a single waveform. **HPSS** (`librosa.effects.hpss`) decomposes the signal into:

- **Harmonic** : sustained tonal content (melody, vocals, drones)
- **Percussive** : transient rhythmic content (drums, plucks, beats)

Each component is transformed separately, giving the model physically meaningful channels instead of duplicating a single spectrogram into fake RGB channels (as the baseline does).

### 3. Two-Channel Input Representation

The final feature tensor has shape **(2, 84, 1292)**:

| Channel | Content                   |
| ------- | ------------------------- |
| 0       | Normalized harmonic CQT   |
| 1       | Normalized percussive CQT |

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
│  CQT (per src)  │  -> dB -> normalize [0,1]
└────────┬────────┘
         ▼
   (2, 84, 1292)  ──►  FolkMusicCQTResNet  ──►  6-class logits
```

### FolkMusicCQTResNet

A custom 2D residual network (`model.py`) trained from scratch, no ImageNet transfer learning. It is shaped for the high aspect ratio of spectrograms (84 frequency bins vs. 1292 time frames).

**Input:** `(batch, 2, 84, 1292)`
**Output:** `(batch, 6)` genre logits

### ResBlock2D

Each residual block contains:

- `Conv2d(3×3)` -> `BatchNorm2d` -> `ReLU`
- `Dropout2d`
- `Conv2d(3×3)` -> `BatchNorm2d`
- Skip connection (with 1×1 conv + BN when channels or stride change)
- Final `ReLU`

BatchNorm and dropout are integrated into every block (not only the classifier head), improving training stability and generalization.

### Layer Hierarchy & Shape Progression

| Stage              | Operation                          | Output Shape (H × W) |
| ------------------ | ---------------------------------- | -------------------- |
| Input              | —                                  | 84 × 1292            |
| `conv_init`        | Conv2d 2->32, BN, ReLU             | 84 × 1292            |
| `layer1` + `pool1` | ResBlock 32->64, MaxPool 2×2       | 42 × 646             |
| `layer2` + `pool2` | ResBlock 64->128, MaxPool 2×2      | 21 × 323             |
| `layer3` + `pool3` | ResBlock 128->256, MaxPool **1×2** | 21 × 161             |
| `layer4` + `pool4` | ResBlock 256->256, MaxPool 2×2     | 10 × 80              |
| `layer5`           | ResBlock 256->512                  | 10 × 80              |
| `global_pool`      | AdaptiveAvgPool2d(1, 1)            | 1 × 1                |
| `fc`               | Linear 512->128->6                 | 6                    |

The asymmetric **1×2 pool** at `pool3` downsamples time more aggressively than frequency, reflecting that spectrograms are much wider in time than in frequency.

### Classification Head

```
AdaptiveAvgPool2d(1,1)
  -> Flatten
  -> Linear(512, 128) -> BatchNorm1d -> ReLU -> Dropout
  -> Linear(128, num_classes)
```

---

## Training Configuration

| Setting      | Value                                             |
| ------------ | ------------------------------------------------- |
| Optimizer    | AdamW (lr=1e-3, weight_decay=1e-4)                |
| Loss         | CrossEntropyLoss (label smoothing=0.1)            |
| Scheduler    | ReduceLROnPlateau (patience=5, factor=0.5)        |
| Augmentation | SpecAugment (freq + time masking)                 |
| Batch size   | 16                                                |
| Epochs       | 25                                                |
| Device       | CUDA when available (`pin_memory`, `num_workers`) |

---

## Pipeline

```
preprocess.py  ->  train_sota.py  ->  evaluate_sota.py  ->  predict_sota.py
     │                  │                   │                    │
  .wav -> .npy      train model         metrics & plots      classify audio
```

| Script                  | Role                                                       |
| ----------------------- | ---------------------------------------------------------- |
| `utils.py`              | Shared config, paths, genre labels                         |
| `preprocess.py`         | HPSS + CQT feature extraction, saves `.npy`                |
| `model.py`              | `FolkMusicCQTResNet` and `ResBlock2D`                      |
| `dataset.py`            | PyTorch `Dataset` / `DataLoader` with precomputed fallback |
| `train_sota.py`         | Training loop with SpecAugment                             |
| `evaluate_sota.py`      | Accuracy, F1, confusion matrix, confidence analysis        |
| `predict_sota.py`       | Chunked inference with reliability scoring                 |
| `integrate_workflow.py` | End-to-end automation                                      |
| `notebook.ipynb`        | Standalone Kaggle notebook (full pipeline + GPU)           |

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

## Approach 2 — Improved Model (`cadence-second.ipynb`)

After training and evaluating the first model, several weaknesses became clear: severe train–test overfitting, unstable test accuracy across epochs, weak performance on **Tharu** and **Newari**, and limited feature diversity (CQT-only). The second notebook addresses these with richer features, a stronger architecture, and more aggressive regularization.

**Notebook:** `notebook/cadence-second.ipynb`
**Best checkpoint:** `models/second/best_model.pth`
**Results:** `results/second/`

### Motivation (from first-model analysis)

| Issue | First model symptom | Second-model fix |
| ----- | ------------------- | ---------------- |
| Overfitting | Train ~89%, test 55–73% oscillating | Mixup, stronger dropout, cosine LR, gradient clipping |
| Weak classes | Tharu F1=0.56, Newari F1=0.63 | Inverse-frequency class weights + mel/delta features |
| Feature gap | 2-channel CQT only | 4-channel fusion (CQT + mel + delta-mel) |
| Architecture | Plain ResNet, avg pool only | SE attention, dual global pooling |
| Augmentation | 2 small freq/time masks | 3 wider masks + mixup (first 40 epochs) |

---

### 1. Four-Channel Feature Representation

The second approach extends HPSS + CQT with mel-spectrogram and temporal dynamics:

| Channel | Content | Role |
| ------- | ------- | ---- |
| 0 | Normalized harmonic CQT | Tonal / melodic structure |
| 1 | Normalized percussive CQT | Rhythmic / transient content |
| 2 | Normalized mel-spectrogram (84 mels, 20–8000 Hz) | Perceptual frequency layout |
| 3 | Delta-mel (first-order temporal derivative) | Onset / articulation cues |

**Output shape:** `(4, 84, 1292)` — same spatial dimensions as the first model, with two additional complementary views of the signal.

Precomputed arrays are saved to `processed_features/{train,test}/<genre>/` inside the notebook working directory (Kaggle: `/kaggle/working/processed_features`).

---

### 2. ImprovedFolkNet Architecture

A custom residual CNN with **~6.47M parameters** (vs. ~6.13M in `FolkMusicCQTResNet`).

```
Raw Audio (.wav)
      │
      ▼
┌─────────────────────────────────────────┐
│  HPSS + CQT-H/P + Mel + Delta-Mel       │
└────────────────┬────────────────────────┘
                 ▼
          (4, 84, 1292)  ──►  ImprovedFolkNet  ──►  6-class logits
```

**Input:** `(batch, 4, 84, 1292)`
**Output:** `(batch, 6)` genre logits

#### ResBlock with Squeeze-and-Excitation (SE)

Each block adds channel re-weighting via `SEBlock` (reduction=8), helping the network emphasize informative feature maps per genre.

#### Layer hierarchy

| Stage | Operation | Output shape (H × W) |
| ----- | --------- | -------------------- |
| `stem` | Conv2d 4→32 (7×7), BN, ReLU | 84 × 1292 |
| `layer1` + `pool1` | ResBlock 32→64, MaxPool 2×2 | 42 × 646 |
| `layer2` + `pool2` | ResBlock 64→128, MaxPool 2×2 | 21 × 323 |
| `layer3` + `pool3` | ResBlock 128→256, MaxPool **1×2** | 21 × 161 |
| `layer4` + `pool4` | ResBlock 256→256, MaxPool 2×2 | 10 × 80 |
| `layer5` | ResBlock 256→512 | 10 × 80 |
| Global pool | AdaptiveAvgPool + AdaptiveMaxPool | 512 + 512 |
| Classifier | Linear 1024→256→128→6 | 6 |

The classifier concatenates **average** and **max** global pooling (1024-dim input), capturing both typical and salient spectral patterns.

---

### 3. Training Configuration (Approach 2)

| Setting | Value | Notes |
| ------- | ----- | ----- |
| Optimizer | AdamW (lr=3e-4, weight_decay=5e-4) | Lower LR, stronger L2 vs. approach 1 |
| Loss | CrossEntropyLoss (weighted, label_smoothing=0.1) | Inverse-frequency class weights |
| Scheduler | CosineAnnealingLR (T_max=50, eta_min=1e-6) | Smooth decay; avoids ReduceLROnPlateau oscillation |
| Augmentation | SpecAugment (3 freq × width 15, 3 time × width 60) | Stronger than baseline |
| Mixup | α=0.3 for epochs 1–40; disabled last 10 epochs | Sharper decision boundaries at end |
| Gradient clipping | max_norm=2.0 | Stabilizes training |
| Batch size | 32 | Larger batches → more stable gradients |
| Epochs | 50 | Longer schedule than approach 1 |
| Dropout | 0.35 (Conv2d blocks + classifier) | Higher than approach 1 (0.2) |

**Class weights (inverse frequency):** tamang_selo=0.90, deuda=0.91, bhajan=1.16, newari=1.14, tharu=0.89, lok_dohori=1.01

---

### 4. Evaluation Results (Approach 2)

| Metric | Value |
| ------ | ----- |
| Test accuracy | **75.06%** |
| Macro F1 | **0.7408** |
| Best epoch checkpoint | Saved to `models/second/best_model.pth` |

**Per-class metrics (test set, 810 samples):**

| Genre | Precision | Recall | F1 |
| ----- | --------- | ------ | -- |
| tamang_selo | 0.94 | 0.75 | 0.84 |
| deuda | 0.84 | 0.50 | 0.62 |
| bhajan | 0.89 | 0.87 | 0.88 |
| newari | 0.58 | 0.88 | 0.70 |
| tharu | 0.70 | 0.67 | 0.69 |
| lok_dohori | 0.63 | 0.82 | 0.71 |

Plots: `results/second/confusion_matrix.png`, `results/second/per_class_metrics.png`, `results/second/training_curves.png`

---

### 5. Inference (Approach 2)

The notebook includes a `Predictor` class that:

1. Loads `ImprovedFolkNet` from the best checkpoint
2. Splits long audio into overlapping 30-second chunks (50% overlap)
3. Extracts 4-channel features per chunk
4. Averages softmax probabilities across chunks for the final genre label

---

### Quick Start (Notebook)

Open and run `notebook/cadence-second.ipynb` on Kaggle (GPU recommended) or locally with the dataset at `dataset/splits/`. The notebook is self-contained: preprocessing, training, evaluation, and inference run in sequence.

---

## Model Comparison — Approach 1 vs. Approach 2

Both models classify the same six Nepali folk genres on an identical train/test split (2,903 train / 810 test samples). Below is a detailed side-by-side comparison.

### Summary metrics

| Metric | Approach 1 (`cadence.ipynb`) | Approach 2 (`cadence-second.ipynb`) | Δ |
| ------ | ---------------------------- | ------------------------------------- | - |
| Test accuracy | 73.21% | **75.06%** | +1.85 pp |
| Macro F1 | 0.7179 | **0.7408** | +0.023 |
| Parameters | ~6.13M | ~6.47M | +340K |
| Training epochs | 25 | 50 | +25 |
| Best checkpoint | `models/first/sota_best_model.pth` | `models/second/best_model.pth` | — |

Approach 2 improves overall accuracy and macro F1 modestly but consistently. The larger gains appear on previously weak classes rather than on headline accuracy alone.

---

### Per-class F1 comparison

| Genre | Approach 1 F1 | Approach 2 F1 | Change | Notes |
| ----- | ------------- | ------------- | ------ | ----- |
| tamang_selo | 0.786 | **0.84** | +0.05 | Higher precision (0.94); recall still strong |
| deuda | **0.659** | 0.62 | −0.04 | Approach 2 trades recall (0.50) for precision (0.84) |
| bhajan | **0.926** | 0.88 | −0.05 | Approach 1 remains best on bhajan |
| newari | 0.634 | **0.70** | +0.07 | Largest relative gain; recall jumps to 0.88 |
| tharu | 0.558 | **0.69** | +0.13 | Largest absolute gain; main target of improvements |
| lok_dohori | **0.745** | 0.71 | −0.04 | Slight regression; still competitive |

**Takeaway:** Approach 2 clearly wins on **Tharu**, **Newari**, and **Tamang Selo** — the classes most confused in the first model. **Bhajan** and **Lok Dohori** remain strong under approach 1; **Deuda** is mixed (better precision, worse recall in approach 2).

---

### Feature representation

| Aspect | Approach 1 | Approach 2 |
| ------ | ---------- | ---------- |
| Channels | 2 (harmonic CQT, percussive CQT) | 4 (+ mel, delta-mel) |
| Mel bands | None | 84 mels, 20–8000 Hz |
| Temporal dynamics | Implicit in CQT time axis | Explicit delta-mel channel |
| Preprocessed path | `dataset/processed_cqt/` | `processed_features/` (notebook-local) |
| Rationale | Music-aware CQT + HPSS separation | Multi-view fusion for timbre and rhythm |

Approach 1 is leaner and faster to preprocess. Approach 2 extracts roughly twice the feature information per clip, which helps disambiguate rhythmically similar genres (e.g. Tharu vs. Tamang Selo) but increases CPU preprocessing time (~100 minutes on the full dataset in the notebook run).

---

### Architecture

| Component | Approach 1 (`FolkMusicCQTResNet`) | Approach 2 (`ImprovedFolkNet`) |
| --------- | -------------------------------- | ------------------------------ |
| Input channels | 2 | 4 |
| Stem | 3×3 conv | 7×7 conv (wider receptive field) |
| Residual block | Conv → BN → ReLU → Dropout → Conv → BN → skip | Same + **SEBlock** after conv2 |
| Global pooling | AdaptiveAvgPool only | **Avg + Max** concatenated |
| Classifier | 512 → 128 → 6 | 1024 → 256 → 128 → 6 |
| Dropout | 0.2 | 0.35 |
| Attention | None | Squeeze-and-Excitation per block |

Both share the same asymmetric **1×2 pool** at stage 3 to preserve frequency resolution. Approach 2 adds capacity and regularization in the head and channel attention without changing the overall encoder depth.

---

### Training strategy

| Setting | Approach 1 | Approach 2 |
| ------- | ---------- | ---------- |
| Learning rate | 1e-3 | 3e-4 |
| Weight decay | 1e-4 | 5e-4 |
| Batch size | 16 | 32 |
| LR scheduler | ReduceLROnPlateau (patience=5) | CosineAnnealingLR (50 epochs) |
| Class balancing | None | Inverse-frequency weights |
| Mixup | No | Yes (α=0.3, epochs 1–40) |
| SpecAugment | 2 masks, max width 10 / 30 | 3 masks, max width 15 / 60 |
| Gradient clipping | No | Yes (max_norm=2.0) |
| Label smoothing | 0.1 | 0.1 |

Approach 1’s ReduceLROnPlateau contributed to **±15% test accuracy swings** between epochs (see `results/first/analysis.md`). Approach 2’s cosine schedule, mixup, and larger batches produce **smoother learning curves** and a smaller train–test gap, though some overfitting remains after 50 epochs.

---

### Training dynamics and overfitting

| Observation | Approach 1 | Approach 2 |
| ----------- | ---------- | ---------- |
| Final train accuracy | ~89% | Lower gap (mixup inflates train metrics) |
| Test accuracy stability | High variance epoch-to-epoch | More monotonic improvement (44% → 75% over 50 epochs) |
| Best test epoch | Within 25 epochs | Epoch with 75.06% (saved as best) |
| Primary failure mode | Memorization + class confusion | Deuda recall collapse; bhajan slightly below baseline |

---

### Confusion patterns (qualitative)

From the first model’s confusion matrix analysis:

- **Tharu → Tamang Selo** and **Deuda → Newari / Lok Dohori** were major error paths.

Approach 2 reduces cross-confusion among minority / rhythmically similar classes by combining mel timbre, delta dynamics, and class-weighted loss. Remaining weak spot: **Deuda recall (0.50)** — the model often abstains from predicting deuda in favor of higher-precision but wrong alternatives, suggesting deuda still overlaps acoustically with newari and lok_dohori.

---

### When to use which model

| Use case | Recommended model | Reason |
| -------- | ----------------- | ------ |
| Fast preprocessing / minimal compute | **Approach 1** | 2-channel CQT only; fewer features to extract |
| Best overall accuracy & macro F1 | **Approach 2** | +1.85 pp accuracy, +0.023 macro F1 |
| Bhajan-heavy deployment | **Approach 1** | F1 0.926 vs. 0.88 |
| Tharu / Newari / minority classes | **Approach 2** | +0.13 and +0.07 F1 on tharu and newari |
| Production inference latency | **Approach 1** | Smaller input tensor (2 vs. 4 channels) |
| Research / further fine-tuning | **Approach 2** | Richer features, stronger regularization stack |

---

### Artifacts by approach

```
Cadence/
├── notebook/
│   ├── cadence.ipynb           # Approach 1 — full pipeline
│   └── cadence-second.ipynb    # Approach 2 — improved pipeline
├── models/
│   ├── first/                  # sota_best_model.pth, epoch checkpoints
│   └── second/                 # best_model.pth
└── results/
    ├── first/                  # evaluation.md, analysis.md, plots
    └── second/                 # confusion_matrix, per_class_metrics, training_curves
```

---
