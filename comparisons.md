# Project Comparison Analysis: Root vs. /new

This document analyzes the differences and improvements in the project located in the `./new` directory compared to the original project in the root directory.

## 1. High-Level Summary
The project in the root directory is a baseline implementation of a music genre classifier using a pretrained EfficientNet-B0 model and Mel Spectrograms. The project in the `./new` folder represents a "SOTA" (State-of-the-Art) evolution, moving from general-purpose image models to a custom-architected CNN specifically designed for audio features (CQT).

---

## 2. Detailed Comparison

### A. Feature Engineering & Preprocessing
| Feature | Root Project (Baseline) | `./new` Project (SOTA) | Improvement |
| :--- | :--- | :--- | :--- |
| **Audio Representation** | Mel Spectrograms | Constant-Q Transform (CQT) | CQT is superior for music as it aligns better with musical octaves and pitch. |
| **Source Separation** | None (Raw audio $\rightarrow$ Mel) | Harmonic-Percussive Source Separation (HPSS) | Separates "melody/harmony" from "rhythm/percussion," providing the model with richer, disentangled features. |
| **Input Channels** | 3 Channels (Repeated Mel) | 2 Channels (Harmonic CQT, Percussive CQT) | Instead of fake RGB channels, it uses physically meaningful channels (Harmonic vs. Percussive). |
| **Normalization** | Global Min-Max | Per-channel dB scale $\rightarrow$ Min-Max | More stable inputs for the neural network by converting to log-scale (dB) first. |
| **Data Pipeline** | On-the-fly computation | Precomputed `.npy` files with fallback | Massive speed increase during training by avoiding redundant FFT/CQT calculations. |

### B. Model Architecture
| Component | Root Project (Baseline) | `./new` Project (SOTA) | Improvement |
| :--- | :--- | :--- | :--- |
| **Backbone** | Pretrained EfficientNet-B0 | Custom `FolkMusicCQTResNet` | Moving from an ImageNet-trained model to a custom Residual Network tailored for audio shapes. |
| **Design Philosophy** | Transfer Learning | Domain-Specific Architecture | Custom ResBlocks with specific pooling (1x2) to handle the high aspect ratio of spectrograms. |
| **Input Shape** | (3, 128, 1292) | (2, 84, 1292) | Optimized for CQT bins (84) and specific audio duration. |
| **Regularization** | Dropout in FC head | BatchNorm + Dropout in every ResBlock | Significantly better stability and generalization through integrated Batch Normalization. |

### C. Software Engineering & Workflow
| Aspect | Root Project (Baseline) | `./new` Project (SOTA) | Improvement |
| :--- | :--- | :--- | :--- |
| **Modularity** | Basic script structure | Specialized pipeline scripts (`train_sota.py`, `predict_sota.py`, `evaluate_sota.py`) | Better separation of concerns: training, evaluation, and inference are now distinct workflows. |
| **Robustness** | Basic error handling | Safe fallbacks (Zero-tensors on load failure, Mel-fallback for CQT) | Prevents training crashes due to a few corrupted audio files. |
| **Infrastructure** | Standard `dataset.py` | High-performance `DataLoader` with `pin_memory` and `num_workers` | Optimized for GPU utilization and faster data throughput. |

---

## 3. Why the `/new` Project is Better

1.  **Musical Intelligence**: By using **CQT** and **HPSS**, the model "understands" music fundamentally better than a model looking at a Mel Spectrogram. It can distinguish between a rhythmic beat (percussive) and a melodic line (harmonic).
2.  **Architectural Fit**: EfficientNet is designed for natural images (cats, dogs). The new **Custom ResNet** is designed for time-frequency representations, using adaptive pooling and asymmetric kernels to better capture audio patterns.
3.  **Training Efficiency**: The introduction of **precomputed features** removes the CPU bottleneck, allowing the GPU to be the limiting factor rather than the audio processing logic.
4.  **Professional Pipeline**: The transition from a set of scripts to a structured workflow (`preprocess` $\rightarrow$ `train` $\rightarrow$ `evaluate` $\rightarrow$ `predict`) makes the project reproducible and scalable.

## 4. Conclusion
The project in `./new` is a professional-grade upgrade. It replaces general-purpose "deep learning defaults" with **domain-specific audio engineering**, resulting in a system that is likely more accurate, significantly faster to train, and more robust to real-world audio data.
