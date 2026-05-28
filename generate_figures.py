#!/usr/bin/env python3
"""generate_figures.py — publication-ready figures for the DRM thesis.

Usage:
    python generate_figures.py                          # all six figures
    python generate_figures.py --figure training_curves # single figure
    python generate_figures.py --out_dir my_figs/
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_curve, auc, confusion_matrix

from evaluate import fit_rule, apply_rule

# ── Color palette ─────────────────────────────────────────────────────────────
LSTM_FULL    = "#1f77b4"
LSTM_ABLATED = "#d62728"
RULE         = "#ff7f0e"
IF_COLOR     = "#2ca02c"
OCSVM_COLOR  = "#9467bd"
NORMAL_COLOR = "#808080"
ATTACK_COLOR = "#d62728"


def set_style():
    matplotlib.rcParams.update({
        "font.family":      "sans-serif",
        "font.size":        11,
        "axes.titlesize":   12,
        "axes.labelsize":   11,
        "legend.fontsize":  10,
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
    })


def flatten(X):
    return X.reshape(len(X), -1)


def save_fig(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_artifacts(data_dir="data", model_dir="models"):
    npz     = np.load(os.path.join(data_dir, "sessions.npz"))
    X_train = npz["X_train"].astype(np.float32)
    y_train = npz["y_train"].astype(np.int32)
    X_val   = npz["X_val"].astype(np.float32)
    y_val   = npz["y_val"].astype(np.int32)
    X_test  = npz["X_test"].astype(np.float32)
    y_test  = npz["y_test"].astype(np.int32)

    with open(os.path.join(data_dir, "norm_stats.json"))   as f:
        norm_stats = json.load(f)
    with open(os.path.join(data_dir, "feature_names.json")) as f:
        feature_names = json.load(f)
    with open(os.path.join(model_dir, "lstm_full_results.json"))    as f:
        lstm_full_res = json.load(f)
    with open(os.path.join(model_dir, "lstm_ablated_results.json")) as f:
        lstm_ablated_res = json.load(f)
    with open(os.path.join(model_dir, "all_results.json")) as f:
        all_results = json.load(f)

    fp = np.load(os.path.join(model_dir, "lstm_full_preds.npz"))
    ap = np.load(os.path.join(model_dir, "lstm_ablated_preds.npz"))

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "norm_stats":        norm_stats,
        "feature_names":     feature_names,
        "lstm_full_res":     lstm_full_res,
        "lstm_ablated_res":  lstm_ablated_res,
        "all_results":       all_results,
        "full_probs":        fp["probs"].astype(np.float32),
        "full_preds":        fp["preds"].astype(np.int32),
        "full_threshold":    float(fp["threshold"][0]),
        "ablated_probs":     ap["probs"].astype(np.float32),
        "ablated_preds":     ap["preds"].astype(np.int32),
        "ablated_threshold": float(ap["threshold"][0]),
    }


# ── Figure 1: training_curves ────────────────────────────────────────────────

def generate_training_curves(arts, out_dir):
    full_hist    = arts["lstm_full_res"]["history"]
    ablated_hist = arts["lstm_ablated_res"]["history"]

    fe  = [h["epoch"]   for h in full_hist]
    fv  = [h["val_f1"]  for h in full_hist]
    fl  = [h["loss"]    for h in full_hist]
    ae  = [h["epoch"]   for h in ablated_hist]
    av  = [h["val_f1"]  for h in ablated_hist]
    al  = [h["loss"]    for h in ablated_hist]

    full_best    = arts["lstm_full_res"]["best_epoch"]
    ablated_best = arts["lstm_ablated_res"]["best_epoch"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("LSTM training dynamics: full vs ablated model", fontsize=13)

    for ax, yf, ya, ylabel, title in [
        (ax1, fv, av, "Validation F1",  "Validation F1"),
        (ax2, fl, al, "Training loss",  "Training loss"),
    ]:
        ax.plot(fe, yf, color=LSTM_FULL,    lw=1.8, label="LSTM full")
        ax.plot(ae, ya, color=LSTM_ABLATED, lw=1.8, label="LSTM ablated")
        ax.axvline(full_best,    color=LSTM_FULL,    linestyle="--", alpha=0.8, lw=1.2)
        ax.axvline(ablated_best, color=LSTM_ABLATED, linestyle="--", alpha=0.8, lw=1.2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
        _clean_ax(ax)

    ax1.set_ylim(0, 1)
    fig.tight_layout()
    save_fig(fig, out_dir, "training_curves")


# ── Figure 2: confusion_matrices ─────────────────────────────────────────────

def generate_confusion_matrices(arts, out_dir):
    y_test     = arts["y_test"]
    lstm_preds = arts["full_preds"]

    rb = arts["all_results"].get("rule_based", {})
    if "T_low" in rb and "T_high" in rb:
        T_low, T_high = rb["T_low"], rb["T_high"]
    else:
        T_low, T_high, _ = fit_rule(arts["X_val"], arts["y_val"])
    rule_preds = apply_rule(arts["X_test"], T_low, T_high)

    def draw_cm(ax, cm, title):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_pct   = cm / np.where(row_sums > 0, row_sums, 1) * 100
        ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(title)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Attack"])
        ax.set_yticklabels(["Normal", "Attack"])
        ax.set_xlabel("Predicted: Normal / Attack")
        ax.set_ylabel("True: Normal / Attack")
        thresh = cm.max() / 2
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)",
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=11)

    cm_lstm = confusion_matrix(y_test, lstm_preds)
    cm_rule = confusion_matrix(y_test, rule_preds)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    draw_cm(ax1, cm_lstm, "LSTM full")
    draw_cm(ax2, cm_rule, "Tuned rule")
    fig.tight_layout()
    save_fig(fig, out_dir, "confusion_matrices")


# ── Figure 3: roc_curves ──────────────────────────────────────────────────────

def generate_roc_curves(arts, out_dir):
    y_test         = arts["y_test"]
    X_train        = arts["X_train"]
    y_train        = arts["y_train"]
    X_train_normal = X_train[y_train == 0]

    if_clf = IsolationForest(n_estimators=100, contamination=0.20,
                              random_state=42, n_jobs=-1)
    if_clf.fit(flatten(X_train))
    if_scores = -if_clf.decision_function(flatten(arts["X_test"]))

    ocsvm_clf = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    ocsvm_clf.fit(flatten(X_train_normal))
    ocsvm_scores = -ocsvm_clf.decision_function(flatten(arts["X_test"]))

    detectors = [
        ("LSTM full",        arts["full_probs"],    LSTM_FULL),
        ("LSTM ablated",     arts["ablated_probs"], LSTM_ABLATED),
        ("Isolation Forest", if_scores,             IF_COLOR),
        ("One-Class SVM",    ocsvm_scores,          OCSVM_COLOR),
    ]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1, label="Random")

    for name, scores, color in detectors:
        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC={roc_auc_val:.4f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves — test set")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    _clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, out_dir, "roc_curves")


# ── Figure 4: reuse_interval_distribution ────────────────────────────────────

def generate_reuse_distribution(arts, out_dir):
    import matplotlib.lines as mlines

    ns       = arts["norm_stats"]
    ri_min   = ns["mins"][1]
    ri_range = ns["maxs"][1] - ns["mins"][1]

    rb = arts["all_results"].get("rule_based", {})
    if "T_low_s" in rb:
        T_low_s  = rb["T_low_s"]
        T_high_s = rb["T_high_s"]
        T_low    = rb["T_low"]
        T_high   = rb["T_high"]
    else:
        T_low, T_high, _ = fit_rule(arts["X_val"], arts["y_val"])
        T_low_s  = T_low  * ri_range + ri_min
        T_high_s = T_high * ri_range + ri_min

    X_all = np.concatenate([arts["X_train"], arts["X_val"], arts["X_test"]])
    y_all = np.concatenate([arts["y_train"], arts["y_val"], arts["y_test"]])

    max_norm    = X_all[:, :, 1].max(axis=1)
    max_sec_raw = max_norm * ri_range + ri_min   # may be exactly 0.0

    # Count zero-reuse sessions before filtering (issue 1)
    n_normal_zero = int((max_sec_raw[y_all == 0] == 0).sum())
    n_attack_zero = int((max_sec_raw[y_all == 1] == 0).sum())

    # Only histogram sessions with a positive reuse interval
    mask_pos   = max_sec_raw > 0
    normal_sec = max_sec_raw[(y_all == 0) & mask_pos]
    attack_sec = max_sec_raw[(y_all == 1) & mask_pos]

    bins = np.logspace(-1.0, np.log10(max_sec_raw.max() * 1.1), 60)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(normal_sec, bins=bins, alpha=0.5, color=NORMAL_COLOR, label="Normal sessions")
    ax.hist(attack_sec, bins=bins, alpha=0.5, color=ATTACK_COLOR, label="Attack sessions")

    ax.axvspan(T_low_s, T_high_s, alpha=0.15, color="#4ab5e8",
               label=f"Rule-based detection window ({T_low_s:.0f}–{T_high_s:.0f} s)")

    # Zone boundary lines (issue 2) — thin dashed, outside the window shading
    for xb in [1, 4, 30, 150, 200, 500]:
        ax.axvline(xb, color="dimgrey", linestyle="--", lw=0.9, alpha=0.4)

    ax.set_xscale("log")
    ax.set_xlim(1e-1, None)

    # Omitted-count annotation — upper-left in axes coords (issue 1)
    annot_text = (
        f"{n_normal_zero} normal sessions with zero reuse omitted from "
        "log-scale view (no rule can flag them anyway)"
    )
    ax.text(0.01, 0.97, annot_text, transform=ax.transAxes,
            fontsize=9, color="slategrey", style="italic",
            va="top", ha="left")

    # Main data legend (upper right), kept alive with add_artist
    leg1 = ax.legend(loc="upper right", fontsize=9)
    ax.add_artist(leg1)

    # Second zone legend (lower right) — issue 2
    zone_handles = [
        mlines.Line2D([], [], color="dimgrey", linestyle="--", lw=1.2,
                      label="Retry zone (1–4 s)"),
        mlines.Line2D([], [], color="dimgrey", linestyle="--", lw=1.2,
                      label="Classic replay zone (30–150 s)"),
        mlines.Line2D([], [], color="dimgrey", linestyle="--", lw=1.2,
                      label="Rewatch zone (200–500 s)"),
    ]
    ax.legend(handles=zone_handles, loc="lower right", fontsize=9, title="Zones")

    ax.set_xlabel("Maximum token reuse interval per session (seconds, log scale)")
    ax.set_ylabel("Number of sessions")
    ax.set_title("Reuse interval distribution by class — why a simple rule cannot separate")
    ax.grid(alpha=0.3)
    _clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, out_dir, "reuse_distribution")
    print(f"    Omitted annotation: \"{annot_text}\"")


# ── Figure 5: f1_comparison ───────────────────────────────────────────────────

def generate_f1_comparison(arts, out_dir):
    all_res = arts["all_results"]
    entries = [
        ("LSTM full",        all_res["lstm_full"]["f1"],        True),
        ("LSTM ablated",     all_res["lstm_ablated"]["f1"],     False),
        ("Rule-based",       all_res["rule_based"]["f1"],       False),
        ("Isolation Forest", all_res["isolation_forest"]["f1"], False),
        ("One-Class SVM",    all_res["ocsvm"]["f1"],            False),
    ]
    entries.sort(key=lambda x: x[1], reverse=True)
    names  = [e[0] for e in entries]
    values = [e[1] for e in entries]
    colors = [LSTM_FULL if e[2] else "#aaaaaa" for e in entries]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names, values, color=colors, height=0.55)

    xmax = max(values)
    for bar, val in zip(bars, values):
        ax.text(val + 0.004, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10)

    ax.axvline(0.88, color="grey", linestyle="--", lw=1.5)
    ax.text(0.881, len(names) - 0.65, "H1 target", fontsize=9, color="grey")
    ax.set_xlim(0, xmax + 0.10)
    ax.set_xlabel("F1 Score")
    ax.set_title("Detection performance on held-out test set (F1)")
    ax.grid(axis="x", alpha=0.3)
    _clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, out_dir, "f1_comparison")


# ── Figure 6: h2_ablation ─────────────────────────────────────────────────────

def generate_h2_ablation(arts, out_dir):
    all_res = arts["all_results"]
    full_m  = all_res["lstm_full"]
    abl_m   = all_res["lstm_ablated"]

    metrics = ["precision", "recall", "f1", "auc_roc"]
    labels  = ["Precision", "Recall", "F1", "AUC-ROC"]
    full_v  = [full_m[m] for m in metrics]
    abl_v   = [abl_m[m]  for m in metrics]

    x = np.arange(len(metrics))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - w / 2, full_v, w, label="LSTM full",    color=LSTM_FULL)
    bars2 = ax.bar(x + w / 2, abl_v,  w, label="LSTM ablated", color=LSTM_ABLATED, alpha=0.85)

    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score")
    ax.set_title("H2 ablation — temporal features ablation impact")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, out_dir, "h2_ablation")


# ── Dispatch ──────────────────────────────────────────────────────────────────

DISPATCH = {
    "training_curves":    generate_training_curves,
    "confusion_matrices": generate_confusion_matrices,
    "roc_curves":         generate_roc_curves,
    "reuse_distribution": generate_reuse_distribution,
    "f1_comparison":      generate_f1_comparison,
    "h2_ablation":        generate_h2_ablation,
}


# ── Sanity checks ─────────────────────────────────────────────────────────────

def print_sanity_checks(arts, out_dir):
    from sklearn.metrics import roc_auc_score

    y_test         = arts["y_test"]
    X_train        = arts["X_train"]
    y_train        = arts["y_train"]
    X_train_normal = X_train[y_train == 0]

    print("\n-- Sanity checks ----------------------------------------------")

    # AUC for all detectors
    lstm_auc = float(roc_auc_score(y_test, arts["full_probs"]))
    ab_auc   = float(roc_auc_score(y_test, arts["ablated_probs"]))
    expected_lstm = arts["all_results"]["lstm_full"].get("auc_roc", float("nan"))
    expected_abl  = arts["all_results"]["lstm_ablated"].get("auc_roc", float("nan"))
    print(f"  LSTM full   AUC: {lstm_auc:.6f}  (stored: {expected_lstm:.6f})  match={abs(lstm_auc-expected_lstm)<1e-4}")
    print(f"  LSTM ablated AUC: {ab_auc:.6f}  (stored: {expected_abl:.6f})  match={abs(ab_auc-expected_abl)<1e-4}")

    if_clf = IsolationForest(n_estimators=100, contamination=0.20,
                              random_state=42, n_jobs=-1)
    if_clf.fit(flatten(X_train))
    if_scores = -if_clf.decision_function(flatten(arts["X_test"]))
    if_auc = float(roc_auc_score(y_test, if_scores))

    ocsvm_clf = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    ocsvm_clf.fit(flatten(X_train_normal))
    ocsvm_scores = -ocsvm_clf.decision_function(flatten(arts["X_test"]))
    ocsvm_auc = float(roc_auc_score(y_test, ocsvm_scores))

    print(f"  Isolation Forest AUC: {if_auc:.6f}")
    print(f"  One-Class SVM    AUC: {ocsvm_auc:.6f}")

    # Rule window vs distribution
    ns       = arts["norm_stats"]
    ri_min   = ns["mins"][1]
    ri_range = ns["maxs"][1] - ns["mins"][1]
    rb       = arts["all_results"].get("rule_based", {})
    if "T_low_s" in rb:
        T_low_s, T_high_s = rb["T_low_s"], rb["T_high_s"]
        T_low,   T_high   = rb["T_low"],   rb["T_high"]
    else:
        T_low, T_high, _ = fit_rule(arts["X_val"], arts["y_val"])
        T_low_s  = T_low  * ri_range + ri_min
        T_high_s = T_high * ri_range + ri_min

    X_all = np.concatenate([arts["X_train"], arts["X_val"], arts["X_test"]])
    y_all = np.concatenate([arts["y_train"], arts["y_val"], arts["y_test"]])
    max_sec = np.maximum(
        X_all[:, :, 1].max(axis=1) * ri_range + ri_min, 0.0
    )

    in_window_normal = int(((max_sec[y_all == 0] > T_low_s) &
                             (max_sec[y_all == 0] < T_high_s)).sum())
    out_window_attack = int(((max_sec[y_all == 1] <= T_low_s) |
                              (max_sec[y_all == 1] >= T_high_s)).sum())
    print(f"\n  Rule window: [{T_low_s:.2f} s, {T_high_s:.2f} s]")
    print(f"  Normal sessions whose max reuse falls inside window (FP pool): {in_window_normal}")
    print(f"  Attack sessions whose max reuse falls outside window (FN pool): {out_window_attack}")

    # Rule confusion matrix on test set for cross-check
    rule_preds = apply_rule(arts["X_test"], T_low, T_high)
    cm = confusion_matrix(y_test, rule_preds)
    print(f"\n  Rule test confusion matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

    # Output files
    print(f"\n-- Output files in {out_dir}/ ---------------------------------")
    for name in DISPATCH:
        for ext in ("pdf", "png"):
            path = os.path.join(out_dir, f"{name}.{ext}")
            if os.path.exists(path):
                size = os.path.getsize(path)
                print(f"  {name}.{ext:<5}  {size:>9,} bytes")
            else:
                print(f"  {name}.{ext:<5}  MISSING")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate thesis figures")
    parser.add_argument("--figure", choices=list(DISPATCH.keys()),
                        help="Generate only this figure (default: all)")
    parser.add_argument("--out_dir", default="figures",
                        help="Output directory (default: figures/)")
    args = parser.parse_args()

    set_style()
    print("Loading artifacts...")
    arts = load_artifacts()

    targets = [args.figure] if args.figure else list(DISPATCH.keys())
    for name in targets:
        print(f"  Generating {name}...")
        DISPATCH[name](arts, args.out_dir)
        print(f"    -> {args.out_dir}/{name}.pdf  {args.out_dir}/{name}.png")

    if not args.figure:
        print_sanity_checks(arts, args.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
