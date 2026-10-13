# Linear SVM – Results Report (cumulative: day1 + day2)

## Summary

- On the combined held-out test set, balanced accuracy went from **0.7005 → 0.9452** and macro F1 from **0.6927 → 0.9433** compared with the original day-1 model.
- `day1` test set: balanced accuracy 0.9081 → 0.9384 (original → new).
- `day2` test set: balanced accuracy 0.4928 → 0.9536 (original → new).
- Simply retraining the old pipeline on all sessions ("naive cumulative") can *hurt* because sessions have different signal amplitudes per channel (e.g. electrode placement). The new pipeline handles this with per-session normalization and per-session feature blocks.

## Setup

| Item | Value |
|---|---|
| Sessions | day1, day2 |
| Labels | `relabel_output*.txt` (1 = rest, 2–6 = gestures) |
| Train set | repetitions < 8 of every session, 216,475 samples |
| Test set | repetitions ≥ 8 of every session, 99,321 samples (held out) |
| Features | log1p → per-session standardization → raw + degree-2 polynomial terms + trailing moving averages (25, 100, 400) samples and deviations from them → [shared | per-session] augmentation (126 features) |
| Classifier | StandardScaler + LinearSVC (C=0.1, class_weight=balanced, one-vs-rest) |
| Hyperparameter search | GridSearchCV over C, GroupKFold(4) by (session, repetition), scoring = balanced accuracy |
| Prediction time (test set, features precomputed) | 164.7 ms total, 1.66 µs/sample |

## Model comparison (held-out test set)

| Model | day1 bal. acc | day1 macro F1 | day2 bal. acc | day2 macro F1 | all bal. acc | all macro F1 |
|---|---|---|---|---|---|---|
| Original (day 1 only, raw features) | 0.9081 | 0.8973 | 0.4928 | 0.4318 | 0.7005 | 0.6927 |
| Naive cumulative (day 1 + 2, raw features) | 0.7086 | 0.6821 | 0.8429 | 0.8362 | 0.7789 | 0.7588 |
| New features, day1 only | 0.9363 | 0.9286 | – | – | – | – |
| New features, day2 only | – | – | 0.9504 | 0.9517 | – | – |
| **New cumulative model (day 1 + 2)** | 0.9384 | 0.9312 | 0.9536 | 0.9551 | 0.9452 | 0.9433 |

- *Original*: the first model (day 1 only, 4 raw log-features). It scores near chance-level on day 2.
- *Naive cumulative*: same pipeline as the original, retrained on both days.
- *New features, dayN only*: the new pipeline trained on one day only (tested on that day only). Comparing it with the cumulative model shows how much the other day's data adds.

## Before / after: original model vs. new cumulative model

| Test set | Metric | Original | New | Δ |
|---|---|---|---|---|
| day1 | Accuracy | 0.9379 | 0.9576 | +0.0197 |
| day1 | Balanced accuracy | 0.9081 | 0.9384 | +0.0303 |
| day1 | Macro F1 | 0.8973 | 0.9312 | +0.0339 |
| day1 | Weighted F1 | 0.9386 | 0.9580 | +0.0194 |
| day1 | Cohen's kappa | 0.9060 | 0.9356 | +0.0295 |
| day1 | Matthews corr. coef. (MCC) | 0.9066 | 0.9364 | +0.0298 |
| day2 | Accuracy | 0.7313 | 0.9707 | +0.2395 |
| day2 | Balanced accuracy | 0.4928 | 0.9536 | +0.4608 |
| day2 | Macro F1 | 0.4318 | 0.9551 | +0.5233 |
| day2 | Weighted F1 | 0.6872 | 0.9705 | +0.2833 |
| day2 | Cohen's kappa | 0.5914 | 0.9559 | +0.3645 |
| day2 | Matthews corr. coef. (MCC) | 0.6027 | 0.9560 | +0.3533 |
| all | Accuracy | 0.8331 | 0.9643 | +0.1312 |
| all | Balanced accuracy | 0.7005 | 0.9452 | +0.2447 |
| all | Macro F1 | 0.6927 | 0.9433 | +0.2506 |
| all | Weighted F1 | 0.8271 | 0.9641 | +0.1370 |
| all | Cohen's kappa | 0.7471 | 0.9459 | +0.1988 |
| all | Matthews corr. coef. (MCC) | 0.7514 | 0.9463 | +0.1949 |

## Overall metrics, new model (all sessions)

| Metric | Train | Test |
|---|---|---|
| Accuracy | 0.9828 | 0.9643 |
| Balanced accuracy | 0.9841 | 0.9452 |
| Macro F1 | 0.9780 | 0.9433 |
| Weighted F1 | 0.9828 | 0.9641 |
| Cohen's kappa | 0.9747 | 0.9459 |
| Matthews corr. coef. (MCC) | 0.9748 | 0.9463 |
| Macro ROC AUC (one-vs-rest) | – | 0.9962 |

## Per-class metrics, new model (test set, all sessions)

| Class | Precision | Recall (sensitivity) | Specificity | F1 | ROC AUC | Support |
|---|---|---|---|---|---|---|
| 1 (rest) | 0.9918 | 0.9846 | 0.9901 | 0.9882 | 0.9994 | 54346 |
| 2 | 0.9900 | 0.9991 | 0.9990 | 0.9945 | 1.0000 | 8728 |
| 3 | 0.8743 | 0.9876 | 0.9834 | 0.9275 | 0.9986 | 10377 |
| 4 | 0.9772 | 0.8060 | 0.9982 | 0.8834 | 0.9918 | 8556 |
| 5 | 0.8689 | 0.9457 | 0.9875 | 0.9057 | 0.9901 | 8013 |
| 6 | 0.9734 | 0.9482 | 0.9973 | 0.9606 | 0.9973 | 9301 |

## Per-class F1 by session: original vs. new

| Class | day1 original | day1 new | day1 Δ | day2 original | day2 new | day2 Δ |
|---|---|---|---|---|---|---|
| 1 (rest) | 0.9852 | 0.9908 | +0.0056 | 0.9749 | 0.9856 | +0.0107 |
| 2 | 0.9920 | 0.9955 | +0.0035 | 0.9583 | 0.9936 | +0.0353 |
| 3 | 0.8673 | 0.8599 | -0.0075 | 0.6521 | 0.9878 | +0.3358 |
| 4 | 0.8206 | 0.8696 | +0.0490 | 0.0000 | 0.8982 | +0.8982 |
| 5 | 0.8208 | 0.9412 | +0.1204 | 0.0056 | 0.8742 | +0.8686 |
| 6 | 0.8978 | 0.9301 | +0.0323 | 0.0000 | 0.9912 | +0.9912 |

## Confusion matrix, new model (test set, counts)

Rows = true class, columns = predicted class.

| True \ Pred | 1 (rest) | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| **1 (rest)** | 53510 | 88 | 233 | 75 | 319 | 121 |
| **2** | 8 | 8720 | 0 | 0 | 0 | 0 |
| **3** | 9 | 0 | 10248 | 0 | 0 | 120 |
| **4** | 53 | 0 | 783 | 6896 | 824 | 0 |
| **5** | 349 | 0 | 0 | 86 | 7578 | 0 |
| **6** | 25 | 0 | 457 | 0 | 0 | 8819 |

## Confusion matrix, new model (row-normalized = recall)

| True \ Pred | 1 (rest) | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| **1 (rest)** | 98.5% | 0.2% | 0.4% | 0.1% | 0.6% | 0.2% |
| **2** | 0.1% | 99.9% | 0.0% | 0.0% | 0.0% | 0.0% |
| **3** | 0.1% | 0.0% | 98.8% | 0.0% | 0.0% | 1.2% |
| **4** | 0.6% | 0.0% | 9.2% | 80.6% | 9.6% | 0.0% |
| **5** | 4.4% | 0.0% | 0.0% | 1.1% | 94.6% | 0.0% |
| **6** | 0.3% | 0.0% | 4.9% | 0.0% | 0.0% | 94.8% |

## Most frequent misclassifications, new model

| True | Predicted as | Count | % of true class |
|---|---|---|---|
| 4 | 5 | 824 | 9.63% |
| 4 | 3 | 783 | 9.15% |
| 6 | 3 | 457 | 4.91% |
| 5 | 1 (rest) | 349 | 4.36% |
| 1 (rest) | 5 | 319 | 0.59% |
| 1 (rest) | 3 | 233 | 0.43% |
| 1 (rest) | 6 | 121 | 0.22% |
| 3 | 6 | 120 | 1.16% |

## Rest detection vs. gesture discrimination (all sessions)

| Task | Original acc. | Original F1 | New acc. | New F1 |
|---|---|---|---|---|
| Rest vs. any gesture (binary) | 0.9782 | 0.9760 | 0.9871 | 0.9858 |
| Which gesture (both true & pred are gestures) | 0.6731 | 0.6486 | 0.9490 | 0.9480 |

## Cross-validation (training repetitions, balanced accuracy)

| C | Mean ± SD | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|
| 0.1 | 0.8840 ± 0.0588 | 0.8029 | 0.9634 | 0.8631 | 0.9064 |
| 1.0 | 0.8715 ± 0.0743 | 0.7732 | 0.9631 | 0.8301 | 0.9195 |
| 10.0 | 0.8741 ± 0.0694 | 0.7899 | 0.9587 | 0.8237 | 0.9240 |

## Test performance by session and repetition, new model

| Session | Repetition | Samples | Accuracy | Balanced acc. | Macro F1 |
|---|---|---|---|---|---|
| day1 | 8 | 14886 | 0.9622 | 0.9518 | 0.9413 |
| day1 | 9 | 14547 | 0.9431 | 0.9280 | 0.9147 |
| day1 | 10 | 19511 | 0.9649 | 0.9356 | 0.9321 |
| day2 | 8 | 15038 | 0.9532 | 0.9290 | 0.9333 |
| day2 | 9 | 14913 | 0.9785 | 0.9681 | 0.9707 |
| day2 | 10 | 14352 | 0.9704 | 0.9632 | 0.9595 |
| day2 | 11 | 6074 | 0.9959 | 0.9974 | 0.9936 |

## Notes and caveats

- **New sessions need calibration.** Per-session normalization uses each day's own mean/std. For a new day, record a short calibration block (rest plus a few gestures) to compute these stats. Unless that day is also added to training, the model uses only the shared weights for it. To add a day: put its files next to the others, add it to `SESSIONS` in `emg_data.py`, then run `train_svm.py` and `evaluate_svm.py`.
- **Temporal features are causal.** Moving averages only look at past samples, so the model can run in real time. It needs about 400 samples of history to fill the longest window.
- Normalization stats are computed from training repetitions only. CV folds share those stats (they use no labels), so CV scores can be very slightly optimistic. The test set is unaffected.

## Metric definitions

- **Accuracy**: fraction of samples classified correctly. Inflated by the large rest class.
- **Balanced accuracy**: mean recall across classes. Not affected by class imbalance.
- **Precision**: of the samples predicted as class *k*, the fraction that truly are *k*.
- **Recall / sensitivity**: of the samples that truly are class *k*, the fraction predicted as *k*.
- **Specificity**: of the samples that are not class *k*, the fraction correctly not predicted as *k*.
- **F1**: harmonic mean of precision and recall. Macro = unweighted mean over classes; weighted = mean weighted by class support.
- **Cohen's kappa**: agreement corrected for chance (0 = chance, 1 = perfect).
- **MCC**: correlation between true and predicted labels (−1 to 1). Robust to imbalance.
- **ROC AUC**: one-vs-rest area under the ROC curve from SVM decision scores. Measures how well the scores rank class *k* above the others, independent of the threshold.
