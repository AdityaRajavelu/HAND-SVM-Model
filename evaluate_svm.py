"""Evaluate the trained linear SVM and write a full metrics report to svm_results.md.

Run train_svm.py first (it produces linear_svm.joblib).
Uses the same split as training: repetitions/blocks >= TEST_REP_START of every
session are the held-out test set.
"""
import time
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, label_binarize
from sklearn.svm import LinearSVC

from emg_data import SESSIONS, TEST_REP_START, WINDOWS, load_all

SESS = list(SESSIONS)

X, y, reps, session, group = load_all()
test_mask = reps >= TEST_REP_START
train_mask = ~test_mask
y_train, y_test = y[train_mask], y[test_mask]
s_test, r_test = session[test_mask], reps[test_mask]

model = joblib.load("linear_svm.joblib")
classes = model.classes_
CLASS_NAMES = {c: ("1 (rest)" if str(c) == "1" and 6 in classes else str(c)) for c in classes}


def overall_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced accuracy": balanced_accuracy_score(y_true, y_pred),
        "Macro F1": f1_score(y_true, y_pred, average="macro"),
        "Weighted F1": f1_score(y_true, y_pred, average="weighted"),
        "Cohen's kappa": cohen_kappa_score(y_true, y_pred),
        "Matthews corr. coef. (MCC)": matthews_corrcoef(y_true, y_pred),
    }


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fmt(v):
    return f"{v:.4f}"


def delta(new, old):
    d = new - old
    return f"{d:+.4f}"


# ------------------------------------------------------------ predictions
F = model.transform(X, session)
t0 = time.perf_counter()
y_pred = model.predict_features(F[test_mask])
pred_time = time.perf_counter() - t0
y_pred_train = model.predict_features(F[train_mask])
scores = model.decision_function_features(F[test_mask])

# Baseline: raw log1p + StandardScaler + LinearSVC on the same training split
raw_baseline = make_pipeline(
    FunctionTransformer(lambda z: np.log1p(np.maximum(z, 0.0))),
    StandardScaler(),
    LinearSVC(C=1.0, class_weight="balanced", dual=False, max_iter=5000),
).fit(X[train_mask], y_train)
raw_pred = raw_baseline.predict(X[test_mask])

test_m = overall_metrics(y_test, y_pred)
train_m = overall_metrics(y_train, y_pred_train)
raw_m = overall_metrics(y_test, raw_pred)

y_test_bin = label_binarize(y_test, classes=classes)
auc_per_class = roc_auc_score(y_test_bin, scores, average=None)
auc_macro = roc_auc_score(y_test_bin, scores, average="macro")

prec, rec, f1, sup = precision_recall_fscore_support(y_test, y_pred, labels=classes)
cm = confusion_matrix(y_test, y_pred, labels=classes)
specificity = []
for i in range(len(classes)):
    tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
    fp = cm[:, i].sum() - cm[i, i]
    specificity.append(tn / (tn + fp))
cm_norm = cm / cm.sum(axis=1, keepdims=True)

# Cross-validation (from training).
cv = model.cv_results_
cv_C = [float(c) for c in cv["param_linearsvc__C"]]
cv_rows = [
    [c, f"{cv['mean_test_score'][i]:.4f} ± {cv['std_test_score'][i]:.4f}"]
    + [fmt(cv[f"split{k}_test_score"][i]) for k in range(4)]
    for i, c in enumerate(cv_C)
]

# Per (session, repetition/block) test breakdown.
rep_rows = []
for s in SESS:
    for r in np.unique(r_test[s_test == s]):
        m = (s_test == s) & (r_test == r)
        rep_rows.append([
            s,
            r,
            m.sum(),
            fmt(accuracy_score(y_test[m], y_pred[m])),
            fmt(balanced_accuracy_score(y_test[m], y_pred[m])),
            fmt(f1_score(y_test[m], y_pred[m], average="macro")),
        ])

# Top confusions.
conf_pairs = sorted(
    ((cm[i, j], classes[i], classes[j]) for i in range(len(classes)) for j in range(len(classes)) if i != j and cm[i, j] > 0),
    reverse=True,
)
top_conf = [
    [CLASS_NAMES[t], CLASS_NAMES[p], n, f"{n / cm[list(classes).index(t)].sum():.2%}"]
    for n, t, p in conf_pairs[:8]
]

# ---------------------------------------------------------------- report
n_feat = F.shape[1]
files_desc = ", ".join(f"`{SESSIONS[s]}` ({s})" for s in SESS)
lines = [
    f"# Linear SVM – Results Report ({', '.join(SESS)})",
    "",
    "## Summary",
    "",
    f"- **Test accuracy:** **{fmt(test_m['Accuracy'])}** (balanced accuracy **{fmt(test_m['Balanced accuracy'])}**, macro F1 **{fmt(test_m['Macro F1'])}**, macro ROC AUC **{fmt(auc_macro)}**).",
    f"- **Improvement over raw-feature linear SVM:** accuracy improved from **{fmt(raw_m['Accuracy'])} → {fmt(test_m['Accuracy'])}** ({delta(test_m['Accuracy'], raw_m['Accuracy'])}) and macro F1 from **{fmt(raw_m['Macro F1'])} → {fmt(test_m['Macro F1'])}** ({delta(test_m['Macro F1'], raw_m['Macro F1'])}) by adding causal time-domain window statistics (moving mean, std, waveform length, min/max/range) and polynomial channel interactions.",
    "",
    "## Setup",
    "",
    md_table(
        ["Item", "Value"],
        [
            ["Data files", files_desc],
            ["Channels", f"{X.shape[1]} EMG channels"],
            ["Classes", ", ".join(str(c) for c in classes)],
            ["Train set", f"blocks/repetitions < {TEST_REP_START}, {train_mask.sum():,} samples"],
            ["Test set", f"blocks/repetitions ≥ {TEST_REP_START}, {test_mask.sum():,} samples (held out)"],
            [
                "Features",
                f"log1p → per-session standardization → causal window stats (mean, deviation, std, waveform length, max, min, range) over {WINDOWS} samples + degree-2 polynomial interactions ({n_feat} features)",
            ],
            ["Classifier", f"StandardScaler + LinearSVC (C={model.C}, class_weight={model.class_weight}, one-vs-rest)"],
            ["Hyperparameter search", "GridSearchCV over C, GroupKFold(4) by (session, block/repetition), scoring = balanced accuracy"],
            [
                "Prediction time (test set, features precomputed)",
                f"{pred_time * 1000:.1f} ms total, {pred_time / test_mask.sum() * 1e6:.2f} µs/sample",
            ],
        ],
    ),
    "",
    "## Overall metrics (train vs. test vs. raw baseline)",
    "",
    md_table(
        ["Metric", "Raw LinearSVC (test)", "Feature-engineered Train", "Feature-engineered Test", "Δ (Test vs. Raw)"],
        [[k, fmt(raw_m[k]), fmt(train_m[k]), fmt(test_m[k]), delta(test_m[k], raw_m[k])] for k in test_m]
        + [["Macro ROC AUC (one-vs-rest)", "–", "–", fmt(auc_macro), "–"]],
    ),
    "",
    "## Per-class metrics (test set)",
    "",
    md_table(
        ["Class", "Precision", "Recall (sensitivity)", "Specificity", "F1", "ROC AUC", "Support"],
        [
            [CLASS_NAMES[c], fmt(prec[i]), fmt(rec[i]), fmt(specificity[i]), fmt(f1[i]), fmt(auc_per_class[i]), sup[i]]
            for i, c in enumerate(classes)
        ],
    ),
    "",
    "## Confusion matrix (test set, counts)",
    "",
    "Rows = true class, columns = predicted class.",
    "",
    md_table(
        ["True \\ Pred"] + [CLASS_NAMES[c] for c in classes],
        [[f"**{CLASS_NAMES[c]}**"] + list(cm[i]) for i, c in enumerate(classes)],
    ),
    "",
    "## Confusion matrix (row-normalized = recall)",
    "",
    md_table(
        ["True \\ Pred"] + [CLASS_NAMES[c] for c in classes],
        [[f"**{CLASS_NAMES[c]}**"] + [f"{v:.1%}" for v in cm_norm[i]] for i, c in enumerate(classes)],
    ),
    "",
    "## Most frequent misclassifications",
    "",
    md_table(["True", "Predicted as", "Count", "% of true class"], top_conf),
    "",
    "## Cross-validation (training blocks, balanced accuracy)",
    "",
    md_table(["C", "Mean ± SD", "Fold 1", "Fold 2", "Fold 3", "Fold 4"], cv_rows),
    "",
    "## Test performance by held-out block/repetition",
    "",
    md_table(["Session", "Block/Rep", "Samples", "Accuracy", "Balanced acc.", "Macro F1"], rep_rows),
    "",
]

with open("svm_results.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("Wrote svm_results.md")
