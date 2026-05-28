"""
inference.py
Model loading and per-session classification — no HTTP imports.
All feature engineering lives here; server.py knows nothing about tensors.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lstm_detector import LSTMDetector

SEQ_LEN = 10
N_FEATURES = 4


@dataclass(eq=False)
class Detector:
    """Holds the loaded model and the normalisation statistics."""
    model:     Any          # LSTMDetector in eval mode on self.device
    threshold: float
    mins:      np.ndarray   # shape (N_FEATURES,) — training-set min per feature
    ranges:    np.ndarray   # shape (N_FEATURES,) — maxs-mins, clamped to ≥1e-6
    device:    str

    def _build_features(self, events: list[dict]) -> np.ndarray:
        """Build a (SEQ_LEN, N_FEATURES) raw feature matrix from event dicts."""
        mat  = np.zeros((SEQ_LEN, N_FEATURES), dtype=np.float32)
        seen: dict[str, float] = {}          # licence_id -> last timestamp

        for i, ev in enumerate(events):
            ts  = float(ev["timestamp"])
            lid = str(ev["licence_id"])
            pos = int(ev.get("position", i))

            # feature 0: delta_t (0 for first event)
            mat[i, 0] = ts - float(events[i - 1]["timestamp"]) if i > 0 else 0.0

            # feature 1: token_reuse_interval (0 on first occurrence)
            mat[i, 1] = ts - seen[lid] if lid in seen else 0.0
            seen[lid] = ts

            # feature 2: device_bucket — md5 (not built-in hash) for cross-restart stability
            first_byte = hashlib.md5(str(ev["device_id"]).encode("utf-8")).digest()[0]
            mat[i, 2]  = first_byte / 255.0

            # feature 3: position_decile
            mat[i, 3] = pos / (SEQ_LEN - 1)

        return mat

    def classify_session(self, events: list[dict]) -> dict:
        """Classify a fully-buffered (10-event) session. Returns label/prob/threshold."""
        mat = self._build_features(events)
        X   = (mat - self.mins) / self.ranges      # min-max normalise
        t   = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = float(self.model(t).item())
        return {
            "label":         int(prob >= self.threshold),
            "prob":          prob,
            "threshold":     self.threshold,
            "classified_at": time.time(),
        }


def load_detector(model_dir: str = "models", data_dir: str = "data") -> Detector:
    """Load weights, threshold, and normalisation stats from disk."""
    with open(os.path.join(model_dir, "lstm_full_results.json")) as f:
        res = json.load(f)
    threshold = float(res["threshold"])

    with open(os.path.join(data_dir, "norm_stats.json")) as f:
        ns = json.load(f)
    mins   = np.array(ns["mins"], dtype=np.float32)
    ranges = np.maximum(np.array(ns["maxs"], dtype=np.float32) - mins, 1e-6)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = LSTMDetector(input_size=N_FEATURES, hidden_size=64)
    state  = torch.load(
        os.path.join(model_dir, "lstm_full.pt"),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    return Detector(model=model, threshold=threshold, mins=mins, ranges=ranges, device=device)
