# Linear SVM – Results Report (raw_input)

## Summary

- **Test accuracy:** **0.8691** (balanced accuracy **0.8691**, macro F1 **0.8658**, macro ROC AUC **0.9754**).
- **Improvement over raw-feature linear SVM:** accuracy improved from **0.5585 → 0.8691** (+0.3106) and macro F1 from **0.5082 → 0.8658** (+0.3577) by adding causal time-domain window statistics (moving mean, std, waveform length, min/max/range) and polynomial channel interactions.

## Setup

| Item | Value |
|---|---|
| Data files | `RawInput.txt` (raw_input) |
| Channels | 2 EMG channels |
| Classes | grip1, grip10, grip2, grip3, grip4, grip5, grip6, grip7, grip8, grip9, rest |
| Train set | blocks/repetitions < 8, 23,100 samples |
| Test set | blocks/repetitions ≥ 8, 9,900 samples (held out) |
| Features | log1p → per-session standardization → causal window stats (mean, deviation, std, waveform length, max, min, range) over (10, 25, 50, 100, 200, 400) samples + degree-2 polynomial interactions (207 features) |
| Classifier | StandardScaler + LinearSVC (C=0.1, class_weight=balanced, one-vs-rest) |
| Hyperparameter search | GridSearchCV over C, GroupKFold(4) by (session, block/repetition), scoring = balanced accuracy |
| Prediction time (test set, features precomputed) | 35.4 ms total, 3.57 µs/sample |

## Overall metrics (train vs. test vs. raw baseline)

| Metric | Raw LinearSVC (test) | Feature-engineered Train | Feature-engineered Test | Δ (Test vs. Raw) |
|---|---|---|---|---|
| Accuracy | 0.5585 | 0.9998 | 0.8691 | +0.3106 |
| Balanced accuracy | 0.5585 | 0.9998 | 0.8691 | +0.3106 |
| Macro F1 | 0.5082 | 0.9998 | 0.8658 | +0.3577 |
| Weighted F1 | 0.5082 | 0.9998 | 0.8658 | +0.3577 |
| Cohen's kappa | 0.5143 | 0.9998 | 0.8560 | +0.3417 |
| Matthews corr. coef. (MCC) | 0.5238 | 0.9998 | 0.8569 | +0.3331 |
| Macro ROC AUC (one-vs-rest) | – | – | 0.9754 | – |

## Per-class metrics (test set)

| Class | Precision | Recall (sensitivity) | Specificity | F1 | ROC AUC | Support |
|---|---|---|---|---|---|---|
| grip1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9995 | 900 |
| grip10 | 0.7416 | 0.9600 | 0.9666 | 0.8368 | 0.9573 | 900 |
| grip2 | 1.0000 | 0.9533 | 1.0000 | 0.9761 | 0.9973 | 900 |
| grip3 | 0.7484 | 0.7700 | 0.9741 | 0.7590 | 0.9595 | 900 |
| grip4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9811 | 900 |
| grip5 | 0.9667 | 1.0000 | 0.9966 | 0.9831 | 1.0000 | 900 |
| grip6 | 0.6378 | 0.4989 | 0.9717 | 0.5599 | 0.9101 | 900 |
| grip7 | 0.7725 | 0.7811 | 0.9770 | 0.7768 | 0.9701 | 900 |
| grip8 | 1.0000 | 0.8978 | 1.0000 | 0.9461 | 1.0000 | 900 |
| grip9 | 0.8724 | 0.6989 | 0.9898 | 0.7761 | 0.9750 | 900 |
| rest | 0.8357 | 1.0000 | 0.9803 | 0.9105 | 0.9792 | 900 |

## Confusion matrix (test set, counts)

Rows = true class, columns = predicted class.

| True \ Pred | grip1 | grip10 | grip2 | grip3 | grip4 | grip5 | grip6 | grip7 | grip8 | grip9 | rest |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **grip1** | 900 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **grip10** | 0 | 864 | 0 | 36 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **grip2** | 0 | 0 | 858 | 0 | 0 | 31 | 0 | 0 | 0 | 0 | 11 |
| **grip3** | 0 | 0 | 0 | 693 | 0 | 0 | 0 | 207 | 0 | 0 | 0 |
| **grip4** | 0 | 0 | 0 | 0 | 900 | 0 | 0 | 0 | 0 | 0 | 0 |
| **grip5** | 0 | 0 | 0 | 0 | 0 | 900 | 0 | 0 | 0 | 0 | 0 |
| **grip6** | 0 | 301 | 0 | 0 | 0 | 0 | 449 | 0 | 0 | 0 | 150 |
| **grip7** | 0 | 0 | 0 | 197 | 0 | 0 | 0 | 703 | 0 | 0 | 0 |
| **grip8** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 808 | 92 | 0 |
| **grip9** | 0 | 0 | 0 | 0 | 0 | 0 | 255 | 0 | 0 | 629 | 16 |
| **rest** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 900 |

## Confusion matrix (row-normalized = recall)

| True \ Pred | grip1 | grip10 | grip2 | grip3 | grip4 | grip5 | grip6 | grip7 | grip8 | grip9 | rest |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **grip1** | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **grip10** | 0.0% | 96.0% | 0.0% | 4.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **grip2** | 0.0% | 0.0% | 95.3% | 0.0% | 0.0% | 3.4% | 0.0% | 0.0% | 0.0% | 0.0% | 1.2% |
| **grip3** | 0.0% | 0.0% | 0.0% | 77.0% | 0.0% | 0.0% | 0.0% | 23.0% | 0.0% | 0.0% | 0.0% |
| **grip4** | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **grip5** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **grip6** | 0.0% | 33.4% | 0.0% | 0.0% | 0.0% | 0.0% | 49.9% | 0.0% | 0.0% | 0.0% | 16.7% |
| **grip7** | 0.0% | 0.0% | 0.0% | 21.9% | 0.0% | 0.0% | 0.0% | 78.1% | 0.0% | 0.0% | 0.0% |
| **grip8** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 89.8% | 10.2% | 0.0% |
| **grip9** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 28.3% | 0.0% | 0.0% | 69.9% | 1.8% |
| **rest** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |

## Most frequent misclassifications

| True | Predicted as | Count | % of true class |
|---|---|---|---|
| grip6 | grip10 | 301 | 33.44% |
| grip9 | grip6 | 255 | 28.33% |
| grip3 | grip7 | 207 | 23.00% |
| grip7 | grip3 | 197 | 21.89% |
| grip6 | rest | 150 | 16.67% |
| grip8 | grip9 | 92 | 10.22% |
| grip10 | grip3 | 36 | 4.00% |
| grip2 | grip5 | 31 | 3.44% |

## Cross-validation (training blocks, balanced accuracy)

| C | Mean ± SD | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|
| 0.001 | 0.8663 ± 0.0418 | 0.8420 | 0.8961 | 0.8111 | 0.9161 |
| 0.01 | 0.8855 ± 0.0716 | 0.8903 | 0.9209 | 0.7694 | 0.9612 |
| 0.1 | 0.8933 ± 0.0826 | 0.8989 | 0.9265 | 0.7612 | 0.9867 |

## Test performance by held-out block/repetition

| Session | Block/Rep | Samples | Accuracy | Balanced acc. | Macro F1 |
|---|---|---|---|---|---|
| raw_input | 8 | 3300 | 0.9176 | 0.9176 | 0.9165 |
| raw_input | 9 | 3300 | 0.7891 | 0.7891 | 0.7684 |
| raw_input | 10 | 3300 | 0.9006 | 0.9006 | 0.8985 |
