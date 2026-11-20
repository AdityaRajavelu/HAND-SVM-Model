"""Train a session-aware linear SVM on EMG data to classify gestures/classes.

Configured via emg_data.SESSIONS (currently set to "NewPosition.txt").
Feature engineering and confidence calibration are in emg_data.EMGModel.
"""
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict

from emg_data import SESSIONS, TEST_REP_START, EMGModel, load_all

X, y, reps, session, group = load_all()

# Hold out the last repetitions/blocks (>= TEST_REP_START) of every session as the test set.
test_mask = reps >= TEST_REP_START
train_mask = ~test_mask
print(f"sessions: {list(SESSIONS)} | channels: {X.shape[1]} | classes: {list(np.unique(y))}")
print(f"train: {train_mask.sum()} samples, test: {test_mask.sum()} samples")

model = EMGModel(class_weight="balanced")
# Normalization statistics come from training repetitions only.
model.fit_normalization(X[train_mask], session[train_mask])
# Features are computed on the full, continuous recordings (temporal windows are
# causal, so a test sample only ever sees past samples), then split.
F = model.transform(X, session)
print(f"feature dimension: {F.shape[1]}")

# Tune C with grouped CV: groups are (session, repetition/block) pairs.
search = GridSearchCV(
    model.make_classifier(),
    {"linearsvc__C": [0.001, 0.01, 0.1]},
    cv=GroupKFold(n_splits=4),
    scoring="balanced_accuracy",
    n_jobs=1,
    return_train_score=False,
)
search.fit(F[train_mask], y[train_mask], groups=group[train_mask])
print("best params:", search.best_params_)
print(f"CV balanced accuracy: {search.best_score_:.4f}")

model.pipeline_ = search.best_estimator_
model.C = search.best_params_["linearsvc__C"]
model.classes_ = model.pipeline_.classes_
model.cv_results_ = {
    k: v for k, v in search.cv_results_.items()
    if k.startswith(("param_", "mean_test", "std_test", "split"))
}

oof_scores = cross_val_predict(
    model.make_classifier(),
    F[train_mask],
    y[train_mask],
    groups=group[train_mask],
    cv=GroupKFold(n_splits=4),
    method="decision_function",
)
temp = model.calibrate_temperature(oof_scores, y[train_mask])
print(f"Calibrated softmax temperature: T = {temp:.4f}")

y_pred, y_conf = model.predict_with_confidence_features(F[test_mask])
y_test, s_test = y[test_mask], session[test_mask]
correct = y_pred == y_test
print(f"\nTest accuracy (all sessions): {accuracy_score(y_test, y_pred):.4f}")
print(
    f"Confidence (test): mean={y_conf.mean():.1%} | "
    f"when correct={y_conf[correct].mean():.1%} | "
    f"when wrong={y_conf[~correct].mean():.1%}"
)
for name in SESSIONS:
    m = s_test == name
    print(f"  {name}: {accuracy_score(y_test[m], y_pred[m]):.4f} (mean conf {y_conf[m].mean():.1%})")
print("\nSample test predictions with confidence (one per class):")
for cls in model.classes_:
    idx = np.flatnonzero(y_test == cls)[0]
    print(f"  true={cls:<7s} -> pred={y_pred[idx]:<7s} (confidence: {y_conf[idx]:.1%})")
print()
print(classification_report(y_test, y_pred, digits=4))
print("Confusion matrix (rows = true, cols = predicted):")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, "linear_svm.joblib")
print("\nSaved model to linear_svm.joblib")
