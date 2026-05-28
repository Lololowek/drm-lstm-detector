"""
data_generator.py
Synthetic DRM licence access event sequence generator.

Normal sessions have three realistic variants (pure / retry / rewatch) so that
~30% of normal sessions contain at least one token reuse, breaking the trivial
threshold rule that previously achieved F1=1.0.

Output: data/sessions.npz  (X_train, y_train, X_val, y_val, X_test, y_test)
        data/feature_names.json
        data/norm_stats.json
        data/sessions_meta.csv

Usage:
    python data_generator.py --n_total 10000 --attack_ratio 0.20 --seed 42
    python data_generator.py --verify-only
"""

import argparse
import csv
import json
import os

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
SEQ_LEN = 10
N_DEVICE_BUCKETS = 256
FEATURE_NAMES = [
    "delta_t_norm",
    "token_reuse_interval",
    "device_bucket",
    "position_decile",
]
N_FEATURES = len(FEATURE_NAMES)
RNG_SEED = 42


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _jitter(intervals, rng):
    """Gaussian jitter σ=10% of each interval; clip to ≥0.1 s."""
    sigma = 0.1 * np.abs(intervals)
    return np.maximum(intervals + rng.normal(0, sigma), 0.1)


def _events_from_spec(lids, intervals, devices, event_labels):
    """
    Build a list of event dicts from pre-planned licence IDs, intervals,
    per-event devices, and per-event malicious labels.
    token_reuse_interval is derived from the cumulative timestamps.
    """
    timestamps = np.cumsum(intervals)
    seen = {}
    events = []
    for i, (lid, dev, lbl) in enumerate(zip(lids, devices, event_labels)):
        t = float(timestamps[i])
        reuse = 0.0 if lid not in seen else t - seen[lid]
        seen[lid] = t
        events.append({
            "licence_id": lid,
            "device": dev,
            "timestamp": t,
            "delta_t": float(intervals[i]),
            "token_reuse_interval": reuse,
            "position": i,
            "label": lbl,
        })
    return events


# ── Session generators ────────────────────────────────────────────────────────

def make_normal_session(rng, session_id):
    """
    Three variants chosen per-session:
      70% pure normal  — all 10 licence IDs unique
      15% retry        — one ID repeated after 1–4 s  (transient error)
      15% rewatch      — one ID repeated after 200–500 s  (re-watch)
    """
    n = SEQ_LEN
    intervals = rng.lognormal(mean=3.33, sigma=0.6, size=n)
    lids = [f"lic_{session_id}_{i}" for i in range(n)]
    device = int(rng.integers(0, N_DEVICE_BUCKETS))

    roll = rng.random()
    if roll < 0.15:                                      # retry
        p = int(rng.integers(0, n - 1))                 # orig pos in [0,8]
        lids[p + 1] = lids[p]
        intervals[p + 1] = rng.uniform(1.0, 4.0)
    elif roll < 0.30:                                    # rewatch
        p = int(rng.integers(0, n - 1))
        lids[p + 1] = lids[p]
        intervals[p + 1] = rng.uniform(200.0, 500.0)
    # else: pure normal (70%) — lids already unique

    return _events_from_spec(lids, _jitter(intervals, rng), [device] * n, [0] * n)


def make_replay_session(rng, session_id):
    """
    Three attack variants:
      50% classic replay    — src∈[0,4], tgt∈[src+2,9], natural timing
      25% mimicked-retry    — src+1 target, 1–4 s gap but flanking δt<1 s
      25% multi-replay      — two replays of the same source licence ID

    Device variation: 10% of sessions give the replayed event a different device.
    """
    n = SEQ_LEN
    intervals = rng.lognormal(mean=3.33, sigma=0.6, size=n)
    device = int(rng.integers(0, N_DEVICE_BUCKETS))
    lids = [f"lic_{session_id}_{i}" for i in range(n)]
    labels = [0] * n
    devices = [device] * n

    roll = rng.random()
    if roll < 0.50:                                      # classic replay
        src = int(rng.integers(0, 5))                   # [0,4]
        tgt = int(rng.integers(src + 2, n))             # [src+2,9]
        lids[tgt] = lids[src]
        labels[tgt] = 1

    elif roll < 0.75:                                    # mimicked-retry
        src = int(rng.integers(0, n - 1))               # [0,8]
        tgt = src + 1
        lids[tgt] = lids[src]
        labels[tgt] = 1
        intervals[src] = rng.uniform(0.1, 0.9)          # short δt before
        intervals[tgt] = rng.uniform(1.0, 4.0)          # reuse gap (looks like retry)
        if tgt + 1 < n:
            intervals[tgt + 1] = rng.uniform(0.1, 0.9) # short δt after → burst signature

    else:                                                # multi-replay
        src = int(rng.integers(0, 5))
        available = list(range(src + 2, n))             # always ≥4 elements
        for tgt in sorted(rng.choice(available, size=2, replace=False)):
            lids[tgt] = lids[src]
            labels[tgt] = 1

    # Device variation in ~10% of attack sessions
    if rng.random() < 0.10:
        alt_dev = (device + int(rng.integers(1, N_DEVICE_BUCKETS))) % N_DEVICE_BUCKETS
        for i, lbl in enumerate(labels):
            if lbl == 1:
                devices[i] = alt_dev

    return _events_from_spec(lids, _jitter(intervals, rng), devices, labels)


# ── Feature extraction ────────────────────────────────────────────────────────

def session_to_feature_matrix(events):
    """Convert event list to (SEQ_LEN, N_FEATURES) raw (un-normalised) array."""
    mat = np.zeros((SEQ_LEN, N_FEATURES), dtype=np.float32)
    for i, ev in enumerate(events):
        mat[i, 0] = ev["delta_t"]
        mat[i, 1] = ev["token_reuse_interval"]
        mat[i, 2] = ev["device"] / (N_DEVICE_BUCKETS - 1)
        mat[i, 3] = ev["position"] / (SEQ_LEN - 1)
    return mat


# ── Dataset assembly ──────────────────────────────────────────────────────────

def build_dataset(n_total, attack_ratio, seed):
    rng = np.random.default_rng(seed)
    n_attack = int(n_total * attack_ratio)
    n_normal = n_total - n_attack
    print(f"Generating {n_normal} normal + {n_attack} attack sessions...")

    X_list, y_list, meta = [], [], []
    for sid in range(n_normal):
        events = make_normal_session(rng, f"n{sid}")
        X_list.append(session_to_feature_matrix(events))
        y_list.append(0)
        meta.append({"session_id": f"n{sid}", "label": 0, "n_events": SEQ_LEN})

    for sid in range(n_attack):
        events = make_replay_session(rng, f"a{sid}")
        X_list.append(session_to_feature_matrix(events))
        y_list.append(1)
        meta.append({"session_id": f"a{sid}", "label": 1, "n_events": SEQ_LEN})

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int32)
    idx = rng.permutation(n_total)
    return X[idx], y[idx], [meta[i] for i in idx]


def stratified_split(X, y, train_frac=0.70, val_frac=0.15, seed=RNG_SEED):
    """Stratified shuffle split preserving class ratios in each partition."""
    rng = np.random.default_rng(seed + 1)
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_tr = int(len(idx) * train_frac)
        n_va = int(len(idx) * val_frac)
        tr.extend(idx[:n_tr])
        va.extend(idx[n_tr:n_tr + n_va])
        te.extend(idx[n_tr + n_va:])
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


def normalise(X_train, X_val, X_test):
    """Min-max scaling fit on train only, applied to all splits."""
    flat = X_train.reshape(-1, N_FEATURES)
    mins = flat.min(axis=0)
    maxs = flat.max(axis=0)
    ranges = np.where(maxs - mins > 0, maxs - mins, 1.0)

    def scale(X):
        return (X - mins) / ranges

    return scale(X_train), scale(X_val), scale(X_test), {
        "mins": mins.tolist(), "maxs": maxs.tolist()
    }


# ── Verification ──────────────────────────────────────────────────────────────

def verify_dataset(data_dir):
    data = np.load(os.path.join(data_dir, "sessions.npz"))
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_va, y_va = data["X_val"],   data["y_val"]
    X_te, y_te = data["X_test"],  data["y_test"]

    print("\n=== Dataset Verification ===")
    print(f"Train: {len(y_tr)} | Val: {len(y_va)} | Test: {len(y_te)}")
    print(f"Train attack rate: {y_tr.mean():.2%}")

    tri = FEATURE_NAMES.index("token_reuse_interval")
    # After min-max with min=0, normalised>0 iff raw>0
    has_reuse = lambda X: X[:, :, tri].max(axis=1) > 1e-6

    norm_reuse = has_reuse(X_tr[y_tr == 0]).mean()
    atk_reuse  = has_reuse(X_tr[y_tr == 1]).mean()
    print(f"Normal train sessions with reuse > 0: {norm_reuse:.2%}  (target ~30%)")
    print(f"Attack train sessions with reuse > 0: {atk_reuse:.2%}  (must be 100%)")

    rule_pred = has_reuse(X_te).astype(int)
    tp = int(((rule_pred == 1) & (y_te == 1)).sum())
    fp = int(((rule_pred == 1) & (y_te == 0)).sum())
    fn = int(((rule_pred == 0) & (y_te == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    print(f"Rule-based F1 on test (target 0.50–0.75): {f1:.4f}")
    if not 0.50 <= f1 <= 0.75:
        print(f"WARNING: F1 {f1:.4f} is outside the target range [0.50, 0.75]!")
    print("=== End Verification ===\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_total",      type=int,   default=10000)
    parser.add_argument("--attack_ratio", type=float, default=0.20)
    parser.add_argument("--seed",         type=int,   default=RNG_SEED)
    parser.add_argument("--out_dir",      type=str,   default="data")
    parser.add_argument("--verify-only",  action="store_true",
                        help="Load an existing dataset and run verification only.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.verify_only:
        verify_dataset(args.out_dir)
        return

    X, y, meta = build_dataset(args.n_total, args.attack_ratio, args.seed)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = stratified_split(X, y, seed=args.seed)
    X_tr, X_va, X_te, norm_stats = normalise(X_tr, X_va, X_te)

    np.savez_compressed(
        os.path.join(args.out_dir, "sessions.npz"),
        X_train=X_tr, y_train=y_tr,
        X_val=X_va,   y_val=y_va,
        X_test=X_te,  y_test=y_te,
    )
    with open(os.path.join(args.out_dir, "norm_stats.json"), "w") as f:
        json.dump(norm_stats, f, indent=2)
    with open(os.path.join(args.out_dir, "feature_names.json"), "w") as f:
        json.dump(FEATURE_NAMES, f)
    with open(os.path.join(args.out_dir, "sessions_meta.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["session_id", "label", "n_events"])
        writer.writeheader()
        writer.writerows(meta)

    print(f"Saved to {args.out_dir}/")
    verify_dataset(args.out_dir)


if __name__ == "__main__":
    main()
