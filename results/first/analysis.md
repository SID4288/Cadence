The current model's weaknesses from the training curves and results:

**Key observations:**

1. **Severe overfitting** : Train acc hits 89% but test oscillates 55–73%, never stabilizing. The gap widens continuously.
2. **High test variance** : Test accuracy swings ±15% between epochs, indicating the LR scheduler and batch size aren't stabilizing training.
3. **Weak classes** : Tharu (F1=0.558) and Newari (F1=0.634) are confused heavily with Tamang Selo and Newari respectively (confusion matrix shows 26 Tharu -> Tamang, 26 Deuda -> Newari, 29 Deuda -> Lok Dohori).
4. **Architecture bottleneck** : Simple ResNet with no attention, no temporal modeling.
5. **Feature limitation** : Only CQT (no mel, no chroma, no MFCC fusion).
6. **No class balancing** : Tharu/Newari have fewer test samples.
7. **SpecAugment too weak** : Only 2 freq + 2 time masks with small windows.
