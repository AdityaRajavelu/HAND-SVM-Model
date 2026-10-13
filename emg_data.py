"""Shared data loading and feature engineering for the EMG linear SVM.

To add a new recording session, append an entry to SESSIONS. Every script
(train_svm.py, evaluate_svm.py) picks it up automatically, so the model is
trained cumulatively on all sessions.
"""
import numpy as np
from pathlib import Path
from scipy.ndimage import uniform_filter1d
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC

# =============================================================================
# TRAINING DATA: one line per recording session.
#   "name": "labeled EMG file",
# The labeled EMG file has one sample per line: label,ch1,ch2,ch3,ch4
# (make it from separate EMG + label files with combine_files.py).
# Repetition numbers are derived automatically from the label transitions.
# To change the training data, edit/replace a line. To add data, add a line.
# To exclude a session, comment its line out with #.
# Relative paths are resolved from this folder.
# =============================================================================
SESSIONS = {
    "day1": "labeled_emg_output.txt",
    "day2": "labeled_emg_output_day2.txt",
}

DATA_DIR = Path(__file__).resolve().parent

# Repetitions >= TEST_REP_START in every session are held out for testing.
TEST_REP_START = 8

# Trailing (causal) moving-average window lengths, in samples.
WINDOWS = (25, 100, 400)


def derive_reps(y):
    """Derive repetition counter per gesture directly from the label sequence.

    Starts at 0 (initial rest). Each time a gesture segment starts (y != 1 and
    different from the previous sample), increments that gesture's repetition
    counter and holds the count through the gesture and following rest until
    the next gesture begins.
    """
    reps = np.zeros(len(y), dtype=int)
    diff = np.diff(np.r_[1, y])
    starts = np.flatnonzero((y != 1) & (diff != 0))
    counts = {}
    for k, st in enumerate(starts):
        lab = int(y[st])
        counts[lab] = counts.get(lab, 0) + 1
        end = starts[k + 1] if k + 1 < len(starts) else len(y)
        reps[st:end] = counts[lab]
    return reps


def load_session(file_entry):
    """Load one session from its labeled EMG file path.

    Column 0 of the labeled EMG file is the gesture label; the remaining
    columns are the EMG channels. Repetition indices are derived from column 0.
    """
    if isinstance(file_entry, (tuple, list)):
        file_entry = file_entry[0]
    data_path = DATA_DIR / file_entry
    data = np.loadtxt(data_path, delimiter=",")
    y = data[:, 0].astype(int)
    X = data[:, 1:]
    reps = derive_reps(y)
    return X, y, reps


def load_all(sessions=None):
    """Concatenate sessions (each kept contiguous and in time order).

    Returns X, y, reps, session (name per sample) and group (a unique
    session/repetition id per sample, used for grouped cross-validation).
    """
    sessions = sessions or list(SESSIONS)
    parts = []
    for name in sessions:
        X, y, reps = load_session(SESSIONS[name])
        parts.append((X, y, reps, np.full(len(y), name)))
    X = np.vstack([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    reps = np.concatenate([p[2] for p in parts])
    session = np.concatenate([p[3] for p in parts])
    group = np.char.add(np.char.add(session, "_rep"), reps.astype(str))
    return X, y, reps, session, group


class EMGModel:
    """Session-aware linear SVM.

    Features, per sample:
      1. log1p of the 4 EMG channels, standardized *per session* (removes
         day-to-day gain changes from electrode placement / skin contact).
      2. Degree-2 polynomial terms (channel products / squares).
      3. Trailing moving averages over WINDOWS and the deviation from each
         (short-term temporal context; causal, so usable in real time).
      4. Domain-adaptation feature augmentation (Daume III, 2007): the feature
         vector is repeated as [shared, session1-only, session2-only, ...]. The
         linear SVM learns weights shared by all days plus per-day corrections,
         so adding a new day doesn't overwrite what was learned on older days.

    The classifier itself is still a single linear SVM (LinearSVC).
    """

    def __init__(self, C=1.0, class_weight="balanced"):
        self.C = C
        self.class_weight = class_weight

    # -- features -----------------------------------------------------------
    def _session_features(self, X, name):
        L = np.log1p(X)
        mu, sd = self.norm_stats_[name]
        N = (L - mu) / sd
        cols = [N, PolynomialFeatures(2, include_bias=False).fit_transform(N)]
        for k in WINDOWS:
            sm = uniform_filter1d(N, size=k, axis=0, origin=(k - 1) // 2)  # trailing window
            cols += [sm, N - sm]
        return np.hstack(cols)

    def transform(self, X, session):
        """X must be in time order within each session."""
        base = None
        for name in np.unique(session):
            m = session == name
            F = self._session_features(X[m], name)
            if base is None:
                base = np.empty((len(X), F.shape[1]))
            base[m] = F
        blocks = [base] + [base * (session == s)[:, None] for s in self.sessions_]
        return np.hstack(blocks)

    # -- sklearn-like API ---------------------------------------------------
    def fit_normalization(self, X, session):
        """Per-session mean/std of log1p(EMG). Unsupervised (no labels used)."""
        self.sessions_ = list(np.unique(session))
        self.norm_stats_ = {}
        for name in self.sessions_:
            L = np.log1p(X[session == name])
            self.norm_stats_[name] = (L.mean(0), L.std(0) + 1e-9)
        return self

    def make_classifier(self):
        """Unfitted StandardScaler + LinearSVC pipeline (operates on transformed features)."""
        return make_pipeline(
            StandardScaler(),
            LinearSVC(C=self.C, class_weight=self.class_weight, dual=False, max_iter=5000),
        )

    def fit_features(self, F, y):
        """Fit the linear SVM on already-transformed features."""
        self.pipeline_ = self.make_classifier().fit(F, y)
        self.classes_ = self.pipeline_.classes_
        return self

    def fit(self, X, y, session):
        self.fit_normalization(X, session)
        return self.fit_features(self.transform(X, session), y)

    def predict_features(self, F):
        return self.pipeline_.predict(F)

    def decision_function_features(self, F):
        return self.pipeline_.decision_function(F)

    def predict(self, X, session):
        return self.predict_features(self.transform(X, session))

    def decision_function(self, X, session):
        return self.decision_function_features(self.transform(X, session))
