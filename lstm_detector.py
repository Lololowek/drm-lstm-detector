"""
lstm_detector.py
LSTM-based sequence anomaly detector for DRM licence replay attacks.

Architecture: single LSTM layer (64 hidden) + FC + sigmoid
Loss: binary cross-entropy
Optimiser: Adam lr=1e-3
Threshold: tuned per-epoch on validation set; best checkpoint restored after training.

Usage:
    python lstm_detector.py                   # train and save model
    python lstm_detector.py --epochs 80       # explicit epoch budget
    python lstm_detector.py --ablate          # train WITHOUT temporal features (H2 test)
"""

import argparse
import copy
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "models"

# Feature indices (must match data_generator.py FEATURE_NAMES order)
TEMPORAL_FEATURE_IDX = [0, 1]   # delta_t_norm, token_reuse_interval
ALL_FEATURE_IDX = [0, 1, 2, 3]


# ── Model ────────────────────────────────────────────────────────────────────

class LSTMDetector(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.sigmoid(self.fc(out[:, -1, :])).squeeze(-1)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(data_dir="data", feature_idx=None):
    npz  = np.load(os.path.join(data_dir, "sessions.npz"))
    X_tr = npz["X_train"].astype(np.float32)
    y_tr = npz["y_train"].astype(np.float32)
    X_va = npz["X_val"].astype(np.float32)
    y_va = npz["y_val"].astype(np.float32)
    X_te = npz["X_test"].astype(np.float32)
    y_te = npz["y_test"].astype(np.float32)

    if feature_idx is not None:
        X_tr = X_tr[:, :, feature_idx]
        X_va = X_va[:, :, feature_idx]
        X_te = X_te[:, :, feature_idx]

    def to_ds(X, y):
        return TensorDataset(torch.tensor(X), torch.tensor(y))

    return to_ds(X_tr, y_tr), to_ds(X_va, y_va), to_ds(X_te, y_te), X_te, y_te


# ── Training helpers ──────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimiser, criterion):
    """One full pass over the training set. Returns mean BCE loss."""
    model.train()
    total_loss = 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        optimiser.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * len(y_b)
    return total_loss / len(loader.dataset)


def evaluate_val(model, val_loader):
    """Collect sigmoid outputs and labels from the validation set."""
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            probs.extend(model(X_b.to(DEVICE)).cpu().numpy())
            labels.extend(y_b.numpy())
    return np.array(probs, dtype=np.float32), np.array(labels)


def tune_threshold_from_probs(probs, labels):
    """Grid-search threshold in [0.05, 0.95] that maximises val F1."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(labels, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, float(best_f1)


# ── Training loop with early stopping + best checkpoint ──────────────────────

def train(model, train_loader, val_loader, epochs=80, lr=1e-3, patience=15):
    """
    Train with per-epoch threshold tuning, best-checkpoint saving, and
    patience-based early stopping.
    Returns (history, best_epoch, best_val_f1, best_threshold).
    """
    optimiser   = torch.optim.Adam(model.parameters(), lr=lr)
    criterion   = nn.BCELoss()
    history     = []
    best_val_f1  = 0.0
    best_threshold = 0.5
    best_state  = copy.deepcopy(model.state_dict())
    best_epoch  = 1
    no_improve  = 0

    for epoch in range(1, epochs + 1):
        avg_loss          = train_one_epoch(model, train_loader, optimiser, criterion)
        probs, labels     = evaluate_val(model, val_loader)
        threshold, val_f1 = tune_threshold_from_probs(probs, labels)
        lr_now            = optimiser.param_groups[0]["lr"]

        history.append({"epoch": epoch, "loss": avg_loss,
                        "val_f1": val_f1, "threshold": threshold, "lr": lr_now})

        if val_f1 > best_val_f1:
            best_val_f1    = val_f1
            best_threshold = threshold
            best_state     = copy.deepcopy(model.state_dict())
            best_epoch     = epoch
            no_improve     = 0
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            marker = " *" if epoch == best_epoch else ""
            print(f"  Epoch {epoch:3d}/{epochs} | loss={avg_loss:.4f} | "
                  f"val_F1={val_f1:.4f} | thr={threshold:.2f}{marker}")

        if no_improve >= patience:
            print(f"  Early stop at epoch {epoch} (no improvement for {patience} epochs)")
            break

    model.load_state_dict(best_state)
    return history, best_epoch, best_val_f1, best_threshold


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model, test_loader, threshold):
    """Returns (metrics_dict, probs float32, y_true int8, preds int8)."""
    probs, labels = evaluate_val(model, test_loader)   # reuse helper
    labels = labels.astype(np.int8)
    preds  = (probs >= threshold).astype(np.int8)

    metrics = {
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall":    float(recall_score(labels, preds, zero_division=0)),
        "f1":        float(f1_score(labels, preds, zero_division=0)),
        "auc_roc":   float(roc_auc_score(labels, probs)),
        "threshold": threshold,
        "n_test":    int(len(labels)),
        "n_attack":  int(labels.sum()),
    }
    return metrics, probs, labels, preds


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="data")
    parser.add_argument("--epochs",      type=int,   default=80)
    parser.add_argument("--patience",    type=int,   default=15,
                        help="Early-stop if val F1 does not improve for this many epochs.")
    parser.add_argument("--batch_size",  type=int,   default=128)
    parser.add_argument("--hidden_size", type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--ablate",      action="store_true",
                        help="Train WITHOUT temporal features (H2 ablation test)")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(MODEL_DIR, exist_ok=True)

    label       = "ablated" if args.ablate else "full"
    feature_idx = [2, 3] if args.ablate else None

    print(f"\n{'='*55}")
    print(f"  LSTM Detector -- features: "
          f"{'NON-TEMPORAL ONLY (ablation)' if args.ablate else 'FULL'}")
    print(f"  Device: {DEVICE}  |  Epochs: {args.epochs}  |  Patience: {args.patience}")
    print(f"{'='*55}\n")

    train_ds, val_ds, test_ds, X_te_raw, y_te_raw = load_data(args.data_dir, feature_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    input_size = 2 if args.ablate else 4
    model = LSTMDetector(input_size=input_size, hidden_size=args.hidden_size).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTraining...")
    history, best_epoch, best_val_f1, best_threshold = train(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, patience=args.patience,
    )

    print(f"\nBest epoch: {best_epoch} | Val F1: {best_val_f1:.4f} | "
          f"Threshold: {best_threshold:.2f}")

    print("\nEvaluating on held-out test set...")
    metrics, probs, y_true, preds = evaluate_model(model, test_loader, best_threshold)

    h1_tag = " [H1 MET]" if metrics["f1"] >= 0.88 and not args.ablate else ""
    print(f"\n  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-score  : {metrics['f1']:.4f}{h1_tag}")
    print(f"  AUC-ROC   : {metrics['auc_roc']:.4f}")

    model_path   = os.path.join(MODEL_DIR, f"lstm_{label}.pt")
    results_path = os.path.join(MODEL_DIR, f"lstm_{label}_results.json")
    preds_path   = os.path.join(MODEL_DIR, f"lstm_{label}_preds.npz")

    torch.save(model.state_dict(), model_path)

    results = {
        "model":       f"LSTM ({label})",
        "ablated":     args.ablate,
        "epochs_run":  history[-1]["epoch"],
        "best_epoch":  best_epoch,
        "hidden_size": args.hidden_size,
        **metrics,
        "history": history,
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    np.savez_compressed(
        preds_path,
        probs=probs,
        preds=preds,
        y_true=y_true,
        threshold=np.array([best_threshold], dtype=np.float32),
    )

    print(f"\nSaved: {model_path}, {results_path}, {preds_path}")
    return metrics


if __name__ == "__main__":
    main()
