"""Shared data loading and feature engineering for the EMG linear SVM.

To change or add recording sessions, edit SESSIONS below. Every script
(train_svm.py, evaluate_svm.py) picks it up automatically.
"""
from pathlib import Path
import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC

# =============================================================================
# TRAINING DATA: one line per recording session.
#   "name": "data file",
# Each line of the data file has the class label first, followed by space- or
# comma-separated EMG channel values (e.g. "A 0.269,0.209" or "1,0.32,20.72,...").
# To change the training data, edit/replace a line. To add data, add a line.
# To exclude a session, comment its line out with #.
# =============================================================================
SESSIONS = {
    "raw_input": "RawInput.txt",
    # "combined": "combined.txt",
    # "day1": "labeled_emg_output.txt",
    # "day2": "labeled_emg_output_day2.txt",
}

DATA_DIR = Path(__file__).resolve().parent

# Repetitions >= TEST_REP_START in every session are held out for testing.
TEST_REP_START = 8
NUM_BLOCKS_IF_CONTIGUOUS = 10

# Trailing (causal) window lengths, in samples.
WINDOWS = (10, 25, 50, 100, 200, 400)
POLY_WINDOWS = (25, 100, 400)


def derive_reps(y):
    """Derive repetition / block indices (1..10) from the label sequence.

    - If gestures are recorded as separate repetitions interleaved with rest
      (label 1), counts each occurrence of each gesture (1..N).
    - If classes are recorded in single contiguous blocks (e.g. 5,000 samples
      of A, then B, C, D), subdivides each contiguous segment into
      NUM_BLOCKS_IF_CONTIGUOUS (10) equal time-ordered blocks (1..10) so
      blocks 1..7 are used for training/CV and blocks 8..10 for testing.
    """
    n = len(y)
    # Find contiguous segments where label is constant
    changes = np.r_[0, np.flatnonzero(y[1:] != y[:-1]) + 1, n]
    seg_labels = [y[changes[i]] for i in range(len(changes) - 1)]

    # Check if non-rest classes already appear at least TEST_REP_START times
    has_rest = any(str(lbl) == "1" for lbl in seg_labels)
    non_rest = [lbl for lbl in seg_labels if not (has_rest and str(lbl) == "1")]
    _, seg_counts = np.unique(non_rest, return_counts=True)

    if len(seg_counts) > 0 and seg_counts.min() >= TEST_REP_START:
        reps = np.zeros(n, dtype=int)
        diff = np.diff(np.r_[y[0], y])
        starts = np.flatnonzero((y != 1) & (diff != 0))
        if len(starts) == 0 or y[0] != 1:
            starts = np.unique(np.r_[0 if y[0] != 1 else [], starts]).astype(int)
        counts = {}
        for k, st in enumerate(starts):
            lab = y[st]
            counts[lab] = counts.get(lab, 0) + 1
            end = starts[k + 1] if k + 1 < len(starts) else n
            reps[st:end] = counts[lab]
        return reps

    # Otherwise, split each contiguous segment into 1..NUM_BLOCKS_IF_CONTIGUOUS
    reps = np.zeros(n, dtype=int)
    for i in range(len(changes) - 1):
        st, end = changes[i], changes[i + 1]
        seg_len = end - st
        block_ids = (np.arange(seg_len) * NUM_BLOCKS_IF_CONTIGUOUS) // seg_len + 1
        reps[st:end] = block_ids
    return reps


def load_session(file_entry):
    """Load one session from its labeled EMG file path.

    Supports both string labels ("A 0.269,0.209") and numeric labels
    ("1,0.32,20.72,0.28,0.22") with any number of EMG channels.
    """
    if isinstance(file_entry, (tuple, list)):
        file_entry = file_entry[0]
    data_path = DATA_DIR / file_entry
    labels, rows = [], []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.replace(",", " ").split()
            labels.append(parts[0])
            rows.append([float(v) for v in parts[1:]])
    X = np.asarray(rows, dtype=float)
    try:
        y = np.asarray([int(float(lbl)) for lbl in labels], dtype=int)
    except ValueError:
        y = np.asarray(labels)
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
    """Session-aware linear SVM with causal time-domain EMG features.

    Features, per sample:
      1. log1p of the EMG channels, standardized *per session*, plus
         instantaneous absolute change |L[t] - L[t-1]|.
      2. Over trailing causal windows in WINDOWS:
         - moving average (m) and deviation from average (L - m)
         - moving standard deviation (sd)
         - waveform length / mean absolute change (wl)
         - moving max, moving min, and peak-to-peak range (mx - mn)
      3. Degree-2 polynomial interaction terms over the raw channels and
         key window means/stds.
      4. When multiple sessions are present, appends per-session domain
         adaptation blocks [shared, session1, session2, ...].
    """

    def __init__(self, C=0.01, class_weight="balanced"):
        self.C = C
        self.class_weight = class_weight

    # -- features -----------------------------------------------------------
    def _session_features(self, X, name):
        L = np.log1p(np.maximum(X, 0.0))
        mu, sd_norm = self.norm_stats_[name]
        N = (L - mu) / sd_norm
        dN = np.abs(np.vstack([np.zeros((1, N.shape[1])), np.diff(N, axis=0)]))
        cols = [N, dN]
        poly_inputs = [N]
        for k in WINDOWS:
            org = (k - 1) // 2  # trailing (causal) window
            m = uniform_filter1d(N, size=k, axis=0, origin=org)
            sq = uniform_filter1d(N * N, size=k, axis=0, origin=org)
            sd = np.sqrt(np.maximum(sq - m * m, 0.0))
            wl = uniform_filter1d(dN, size=k, axis=0, origin=org)
            mx = maximum_filter1d(N, size=k, axis=0, origin=org)
            mn = minimum_filter1d(N, size=k, axis=0, origin=org)
            cols += [m, N - m, sd, wl, mx, mn, mx - mn]
            if k in POLY_WINDOWS:
                poly_inputs += [m, sd]
        base = np.hstack(cols)
        poly = PolynomialFeatures(2, include_bias=False).fit_transform(np.hstack(poly_inputs))
        return np.hstack([base, poly])

    def transform(self, X, session):
        """X must be in time order within each session."""
        base = None
        for name in np.unique(session):
            m = session == name
            F = self._session_features(X[m], name)
            if base is None:
                base = np.empty((len(X), F.shape[1]))
            base[m] = F
        if len(self.sessions_) <= 1:
            return base
        blocks = [base] + [base * (session == s)[:, None] for s in self.sessions_]
        return np.hstack(blocks)

    # -- sklearn-like API ---------------------------------------------------
    def fit_normalization(self, X, session):
        """Per-session mean/std of log1p(EMG). Unsupervised (no labels used)."""
        self.sessions_ = list(np.unique(session))
        self.norm_stats_ = {}
        for name in self.sessions_:
            L = np.log1p(np.maximum(X[session == name], 0.0))
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
