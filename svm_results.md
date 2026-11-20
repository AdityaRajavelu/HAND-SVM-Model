# Linear SVM – Results Report (new_position)

## Summary

- **Test accuracy:** **0.7266** (balanced accuracy **0.7266**, macro F1 **0.6883**, macro ROC AUC **0.9367**).
- **Calibrated confidence:** temperature-scaled softmax (**T = 0.8840**, multiclass log-loss **0.9109**). Mean confidence on test predictions is **70.3%** (**74.8%** when correct vs. **58.3%** when incorrect).
- **Improvement over raw-feature linear SVM:** accuracy improved from **0.6847 → 0.7266** (+0.0418) and macro F1 from **0.6604 → 0.6883** (+0.0279) by adding causal time-domain window statistics (moving mean, std, waveform length, min/max/range) and polynomial channel interactions.

## Setup

| Item | Value |
|---|---|
| Data files | `NewPosition.txt` (new_position) |
| Channels | 2 EMG channels |
| Classes | grip1, grip10, grip2, grip3, grip4, grip5, grip6, grip7, grip8, grip9, rest |
| Train set | blocks/repetitions < 8, 23,100 samples |
| Test set | blocks/repetitions ≥ 8, 9,900 samples (held out) |
| Features | log1p → per-session standardization → causal window stats (mean, deviation, std, waveform length, max, min, range) over (10, 25, 50, 100, 200, 400) samples + degree-2 polynomial interactions (207 features) |
| Classifier | StandardScaler + LinearSVC (C=0.1, class_weight=balanced, one-vs-rest) |
| Confidence calibration | Out-of-fold temperature-scaled softmax (T = 0.8840) |
| Hyperparameter search | GridSearchCV over C, GroupKFold(4) by (session, block/repetition), scoring = balanced accuracy |
| Prediction time (test set, features precomputed) | 17.1 ms total, 1.73 µs/sample |

## Overall metrics (train vs. test vs. raw baseline)

| Metric | Raw LinearSVC (test) | Feature-engineered Train | Feature-engineered Test | Δ (Test vs. Raw) |
|---|---|---|---|---|
| Accuracy | 0.6847 | 0.9999 | 0.7266 | +0.0418 |
| Balanced accuracy | 0.6847 | 0.9999 | 0.7266 | +0.0418 |
| Macro F1 | 0.6604 | 0.9999 | 0.6883 | +0.0279 |
| Weighted F1 | 0.6604 | 0.9999 | 0.6883 | +0.0279 |
| Cohen's kappa | 0.6532 | 0.9999 | 0.6992 | +0.0460 |
| Matthews corr. coef. (MCC) | 0.6589 | 0.9999 | 0.7077 | +0.0488 |
| Macro ROC AUC (one-vs-rest) | – | – | 0.9367 | – |
| Multiclass log-loss (calibrated) | – | – | 0.9109 | – |

## Per-class metrics (test set)

| Class | Precision | Recall (sensitivity) | Specificity | F1 | ROC AUC | Mean Conf. | Support |
|---|---|---|---|---|---|---|---|
| grip1 | 0.7972 | 1.0000 | 0.9746 | 0.8871 | 0.9830 | 69.8% | 900 |
| grip10 | 0.3204 | 0.1556 | 0.9670 | 0.2094 | 0.9179 | 52.2% | 900 |
| grip2 | 1.0000 | 0.7456 | 1.0000 | 0.8542 | 0.9980 | 74.8% | 900 |
| grip3 | 1.0000 | 0.0267 | 1.0000 | 0.0519 | 0.6908 | 63.5% | 900 |
| grip4 | 0.9504 | 1.0000 | 0.9948 | 0.9746 | 1.0000 | 81.2% | 900 |
| grip5 | 0.8911 | 1.0000 | 0.9878 | 0.9424 | 1.0000 | 92.0% | 900 |
| grip6 | 0.4212 | 0.4456 | 0.9388 | 0.4330 | 0.7826 | 57.5% | 900 |
| grip7 | 0.4502 | 0.9489 | 0.8841 | 0.6107 | 0.9646 | 71.6% | 900 |
| grip8 | 0.8222 | 0.9200 | 0.9801 | 0.8684 | 1.0000 | 86.7% | 900 |
| grip9 | 0.7289 | 0.7500 | 0.9721 | 0.7393 | 0.9662 | 59.3% | 900 |
| rest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 64.2% | 900 |

## Confidence calibration (test set)

| Confidence bucket | Samples | Mean confidence | Actual accuracy |
|---|---|---|---|
| < 50% | 1576 | 40.8% | 43.5% |
| 50% – 70% | 3052 | 61.6% | 63.1% |
| 70% – 85% | 2678 | 77.7% | 79.9% |
| 85% – 95% | 2293 | 90.0% | 93.7% |
| ≥ 95% | 301 | 96.0% | 97.0% |

## Confusion matrix (test set, counts)

Rows = true class, columns = predicted class.

| True \ Pred | grip1 | grip10 | grip2 | grip3 | grip4 | grip5 | grip6 | grip7 | grip8 | grip9 | rest |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **grip1** | 900 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **grip10** | 0 | 140 | 0 | 0 | 0 | 0 | 551 | 0 | 0 | 209 | 0 |
| **grip2** | 229 | 0 | 671 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **grip3** | 0 | 0 | 0 | 24 | 1 | 0 | 0 | 875 | 0 | 0 | 0 |
| **grip4** | 0 | 0 | 0 | 0 | 900 | 0 | 0 | 0 | 0 | 0 | 0 |
| **grip5** | 0 | 0 | 0 | 0 | 0 | 900 | 0 | 0 | 0 | 0 | 0 |
| **grip6** | 0 | 0 | 0 | 0 | 0 | 110 | 401 | 168 | 179 | 42 | 0 |
| **grip7** | 0 | 0 | 0 | 0 | 46 | 0 | 0 | 854 | 0 | 0 | 0 |
| **grip8** | 0 | 72 | 0 | 0 | 0 | 0 | 0 | 0 | 828 | 0 | 0 |
| **grip9** | 0 | 225 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 675 | 0 |
| **rest** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 900 |

## Confusion matrix (row-normalized = recall)

| True \ Pred | grip1 | grip10 | grip2 | grip3 | grip4 | grip5 | grip6 | grip7 | grip8 | grip9 | rest |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **grip1** | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **grip10** | 0.0% | 15.6% | 0.0% | 0.0% | 0.0% | 0.0% | 61.2% | 0.0% | 0.0% | 23.2% | 0.0% |
| **grip2** | 25.4% | 0.0% | 74.6% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **grip3** | 0.0% | 0.0% | 0.0% | 2.7% | 0.1% | 0.0% | 0.0% | 97.2% | 0.0% | 0.0% | 0.0% |
| **grip4** | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **grip5** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **grip6** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 12.2% | 44.6% | 18.7% | 19.9% | 4.7% | 0.0% |
| **grip7** | 0.0% | 0.0% | 0.0% | 0.0% | 5.1% | 0.0% | 0.0% | 94.9% | 0.0% | 0.0% | 0.0% |
| **grip8** | 0.0% | 8.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 92.0% | 0.0% | 0.0% |
| **grip9** | 0.0% | 25.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 75.0% | 0.0% |
| **rest** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |

## Most frequent misclassifications

| True | Predicted as | Count | % of true class |
|---|---|---|---|
| grip3 | grip7 | 875 | 97.22% |
| grip10 | grip6 | 551 | 61.22% |
| grip2 | grip1 | 229 | 25.44% |
| grip9 | grip10 | 225 | 25.00% |
| grip10 | grip9 | 209 | 23.22% |
| grip6 | grip8 | 179 | 19.89% |
| grip6 | grip7 | 168 | 18.67% |
| grip6 | grip5 | 110 | 12.22% |

## Cross-validation (training blocks, balanced accuracy)

| C | Mean ± SD | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|
| 0.001 | 0.7794 ± 0.0876 | 0.8674 | 0.8082 | 0.6336 | 0.8085 |
| 0.01 | 0.7874 ± 0.0781 | 0.8533 | 0.8374 | 0.6559 | 0.8030 |
| 0.1 | 0.7879 ± 0.0870 | 0.8524 | 0.8406 | 0.6385 | 0.8200 |

## Test performance by held-out block/repetition

| Session | Block/Rep | Samples | Accuracy | Balanced acc. | Macro F1 |
|---|---|---|---|---|---|
| new_position | 8 | 3300 | 0.8215 | 0.8215 | 0.7653 |
| new_position | 9 | 3300 | 0.6670 | 0.6670 | 0.6320 |
| new_position | 10 | 3300 | 0.6912 | 0.6912 | 0.6428 |
