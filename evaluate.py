"""
evaluate.py
Runs all four detectors on the held-out test set and prints a results table.
Also runs the H2 ablation comparison and a McNemar test vs the tuned rule.

Usage:
    python evaluate.py

Prerequisites (run automatically if missing):
    python data_generator.py
    python lstm_detector.py
    python lstm_detector.py --ablate
"""

import os
import sys
import json
import subprocess
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score
)
from scipy.stats import chi2


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data(data_dir="data"):
    npz  = np.load(os.path.join(data_dir, "sessions.npz"))
    X_tr = npz["X_train"].astype(np.float32)
    y_tr = npz["y_train"].astype(np.int32)
    X_va = npz["X_val"].astype(np.float32)
    y_va = npz["y_val"].astype(np.int32)
    X_te = npz["X_test"].astype(np.float32)
    y_te = npz["y_test"].astype(np.int32)
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def flatten(X):
    """Flatten (N, SEQ_LEN, N_FEATURES) -> (N, SEQ_LEN*N_FEATURES) for sklearn."""
    return X.reshape(len(X), -1)


def mcnemar_test(y_true, preds_a, preds_b):
    """
    McNemar test: are classifier A and B significantly different?
    Returns (b, c, statistic, p_value).
    H0: no difference. Uses continuity correction.
    """
    b = int(np.sum((preds_a == y_true) & (preds_b != y_true)))
    c = int(np.sum((preds_a != y_true) & (preds_b == y_true)))
    if b + c == 0:
        return b, c, 0.0, 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p    = float(1 - chi2.cdf(stat, df=1))
    return b, c, float(stat), p


def print_separator(char="-", width=70):
    print(char * width)


# ── Rule-based baseline (tuned two-sided threshold) ───────────────────────────

def fit_rule(X_val, y_val):
    """
    Coarse 20×20 grid search over (T_low, T_high) on normalised
    token_reuse_interval (feature index 1).  Predicts attack if ANY event in
    a session falls in the open interval (T_low, T_high).
    Returns (T_low, T_high, best_val_f1).
    """
    reuse = X_val[:, :, 1]
    best = (0.0, 1.0, 0.0)
    for t_low in np.linspace(0.0, 0.3, 20):
        for t_high in np.linspace(t_low + 0.01, 1.0, 20):
            preds = ((reuse > t_low) & (reuse < t_high)).any(axis=1).astype(int)
            f1 = f1_score(y_val, preds, zero_division=0)
            if f1 > best[2]:
                best = (float(t_low), float(t_high), float(f1))
    return best


def apply_rule(X, T_low, T_high):
    reuse = X[:, :, 1]
    return ((reuse > T_low) & (reuse < T_high)).any(axis=1).astype(int)


# ── Baseline 2: Isolation Forest ─────────────────────────────────────────────

def train_isolation_forest(X_train, contamination=0.20, seed=42):
    clf = IsolationForest(n_estimators=100, contamination=contamination,
                          random_state=seed, n_jobs=-1)
    clf.fit(flatten(X_train))
    return clf


def predict_isolation_forest(clf, X_test):
    return (clf.predict(flatten(X_test)) == -1).astype(int)


# ── Baseline 3: One-Class SVM ─────────────────────────────────────────────────

def train_ocsvm(X_train_normal, nu=0.05):
    clf = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
    clf.fit(flatten(X_train_normal))
    return clf


def predict_ocsvm(clf, X_test):
    return (clf.predict(flatten(X_test)) == -1).astype(int)


# ── Metrics helper ────────────────────────────────────────────────────────────

def compute_metrics(y_true, preds, probs=None):
    m = {
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall":    float(recall_score(y_true, preds, zero_division=0)),
        "f1":        float(f1_score(y_true, preds, zero_division=0)),
        "auc_roc":   float(roc_auc_score(y_true, probs)) if probs is not None else float("nan"),
    }
    return m


# ── Main ──────────────────────────────────────────────────────────────────────

def run_prerequisite(script, extra_args=None):
    cmd = [sys.executable, script] + (extra_args or [])
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        sys.exit(f"ERROR: {script} failed.")
    print(result.stdout)


def main():
    data_dir  = "data"
    model_dir = "models"

    # ── Step 0: ensure prerequisites ─────────────────────────────────────────
    if not os.path.exists(os.path.join(data_dir, "sessions.npz")):
        print("Data not found — generating...")
        run_prerequisite("data_generator.py")

    lstm_full_path    = os.path.join(model_dir, "lstm_full_results.json")
    lstm_ablated_path = os.path.join(model_dir, "lstm_ablated_results.json")

    if not os.path.exists(lstm_full_path):
        print("LSTM (full) not trained — training now...")
        run_prerequisite("lstm_detector.py")

    if not os.path.exists(lstm_ablated_path):
        print("LSTM (ablated) not trained — training now...")
        run_prerequisite("lstm_detector.py", ["--ablate"])

    # ── Step 1: load data ─────────────────────────────────────────────────────
    print("\nLoading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(data_dir)
    X_train_normal = X_train[y_train == 0]
    print(f"  Test set: {len(y_test)} sessions "
          f"({y_test.sum()} attack, {(y_test==0).sum()} normal)")

    with open(os.path.join(data_dir, "norm_stats.json")) as f:
        norm_stats = json.load(f)
    reuse_max = norm_stats["maxs"][1]   # token_reuse_interval max (seconds)

    # ── Step 2: load LSTM results ─────────────────────────────────────────────
    with open(lstm_full_path)    as f: lstm_full    = json.load(f)
    with open(lstm_ablated_path) as f: lstm_ablated = json.load(f)

    # ── Step 3: tune and apply rule-based baseline ────────────────────────────
    print("\nTuning rule threshold on validation set...")
    T_low, T_high, rule_val_f1 = fit_rule(X_val, y_val)
    T_low_s  = T_low  * reuse_max
    T_high_s = T_high * reuse_max
    print(f"  T_low  = {T_low:.4f} ({T_low_s:.1f} s)")
    print(f"  T_high = {T_high:.4f} ({T_high_s:.1f} s)")
    print(f"  Val F1 = {rule_val_f1:.4f}")

    rule_preds   = apply_rule(X_test, T_low, T_high)
    rule_metrics = compute_metrics(y_test, rule_preds)

    # ── Step 4: train and evaluate unsupervised baselines ────────────────────
    print("\nTraining Isolation Forest baseline...")
    if_clf     = train_isolation_forest(X_train, contamination=0.20)
    if_preds   = predict_isolation_forest(if_clf, X_test)
    if_metrics = compute_metrics(y_test, if_preds)

    print("Training One-Class SVM baseline...")
    ocsvm_clf     = train_ocsvm(X_train_normal, nu=0.05)
    ocsvm_preds   = predict_ocsvm(ocsvm_clf, X_test)
    ocsvm_metrics = compute_metrics(y_test, ocsvm_preds)

    # ── Step 5: Results table ─────────────────────────────────────────────────
    print_separator("=")
    print("  EVALUATION RESULTS — Held-out Test Set")
    print_separator("=")

    header = f"{'Detector':<32} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC-ROC':>9}"
    print(header)
    print_separator()

    rows = [
        ("LSTM - full features (proposed)", lstm_full),
        ("LSTM - no temporal features (H2)", lstm_ablated),
        ("Rule-based (tuned threshold)",     rule_metrics),
        ("Isolation Forest",                 if_metrics),
        ("One-Class SVM",                    ocsvm_metrics),
    ]

    for name, m in rows:
        f1_val  = m["f1"]
        flag    = " <- H1 MET" if "full" in name and f1_val >= 0.88 else ""
        auc     = m.get("auc_roc", float("nan"))
        auc_str = f"{auc:.4f}" if not np.isnan(auc) else "  N/A  "
        print(f"  {name:<30} {m['precision']:>10.4f} {m['recall']:>8.4f} "
              f"{f1_val:>8.4f} {auc_str:>9}{flag}")

    print_separator()
    print(f"  Rule threshold (normalised): T_low={T_low:.4f}, T_high={T_high:.4f}")
    print(f"  Rule threshold (seconds)   : T_low={T_low_s:.1f} s, T_high={T_high_s:.1f} s  "
          f"[val F1={rule_val_f1:.4f}]")
    print_separator()

    # ── Step 6: H1 assessment ─────────────────────────────────────────────────
    lstm_f1 = lstm_full["f1"]
    print(f"\n  H1 (LSTM F1 >= 88%): {'CONFIRMED' if lstm_f1 >= 0.88 else 'NOT CONFIRMED'}  [{lstm_f1:.4f}]")

    # ── Step 7: H2 assessment ─────────────────────────────────────────────────
    ablated_f1  = lstm_ablated["f1"]
    improvement = lstm_f1 - ablated_f1
    h2_met      = improvement >= 0.10
    print(f"  H2 (temporal features improve F1 by >= 10pp):")
    print(f"     Full LSTM F1:    {lstm_f1:.4f}")
    print(f"     Ablated LSTM F1: {ablated_f1:.4f}")
    print(f"     Improvement:     {improvement:+.4f}  ->  {'CONFIRMED' if h2_met else 'NOT CONFIRMED'}")

    # ── Step 8: McNemar test (H0: LSTM == tuned rule) ────────────────────────
    preds_npz   = np.load(os.path.join(model_dir, "lstm_full_preds.npz"))
    lstm_preds  = preds_npz["preds"].astype(int)
    y_true_lstm = preds_npz["y_true"].astype(int)

    # rule_preds computed above on the same X_test; verify y_test alignment
    assert np.array_equal(y_true_lstm, y_test), "y_true mismatch between LSTM preds and test set"

    b, c, stat, p = mcnemar_test(y_true_lstm, lstm_preds, rule_preds)
    print(f"\n  H0 (McNemar test — LSTM vs tuned rule):")
    print(f"     Contingency: b={b}  c={c}")
    print(f"     Statistic={stat:.4f}  p={p:.6f}")
    if p < 0.05:
        print(f"     H0 rejected at a=0.05 -- classifiers are significantly different.")
    else:
        print(f"     H0 NOT rejected at a=0.05 -- no significant difference detected.")

    # ── Step 9: save full results ─────────────────────────────────────────────
    all_results = {
        "lstm_full":        lstm_full,
        "lstm_ablated":     lstm_ablated,
        "rule_based":       {**rule_metrics,
                             "T_low": T_low, "T_high": T_high,
                             "T_low_s": T_low_s, "T_high_s": T_high_s,
                             "val_f1": rule_val_f1},
        "isolation_forest": if_metrics,
        "ocsvm":            ocsvm_metrics,
        "h1_confirmed":     bool(lstm_f1 >= 0.88),
        "h2_confirmed":     bool(h2_met),
        "h2_improvement":   float(improvement),
        "mcnemar":          {"b": b, "c": c, "stat": stat, "p": p,
                             "h0_rejected": bool(p < 0.05)},
    }
    os.makedirs(model_dir, exist_ok=True)
    out_path = os.path.join(model_dir, "all_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  Full results saved to {out_path}")
    print_separator("=")


if __name__ == "__main__":
    main()
