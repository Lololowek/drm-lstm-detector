# Intelligent DRM Testing System Based on AI

LSTM-based detector for DRM licence replay attacks in video streaming sessions. Diploma project, Astana IT University, Cybersecurity (B058), 2026.

## Overview

Digital Rights Management (DRM) systems protect premium video content by issuing short-lived licence tokens that authorise playback. Replay attacks subvert this by re-submitting captured licence requests, either to extend authorised viewing beyond the licence window or to share access across devices. Existing rule-based detectors fail in practice because legitimate viewer behaviour — brief retries after network drops (1–4 s), device switches, and rewatches (200–500 s) — produces reuse-interval signals that heavily overlap with attack traffic, leaving no clean threshold a static rule can exploit.

This project contributes a synthetic-but-realistic session dataset of 10 000 labelled sessions (8 000 normal, 2 000 attack across three replay variants), a single-layer LSTM sequence model that learns temporal patterns across all ten events in a session simultaneously, and a hypothesis-driven evaluation against three baselines — a grid-searched two-sided interval rule, Isolation Forest, and One-Class SVM — with a McNemar significance test. The proposed LSTM detector achieves F1 = 0.976 and AUC-ROC = 0.999 on the held-out test set, statistically significantly outperforming every baseline (McNemar p < 0.001 vs the tuned rule).

## Repository structure

```
.
├── data_generator.py      Synthetic DRM session generator (normal + 3 attack variants)
├── lstm_detector.py       LSTM training with early stopping and best-checkpoint restore
├── evaluate.py            Multi-detector evaluation + McNemar significance test
├── generate_figures.py    Six publication-ready figures from saved artifacts
├── show_logs.py           Pretty-prints sample sessions for presentation screenshots
├── data/                  Generated dataset (sessions.npz, meta CSV, norm stats)
├── models/                Trained model weights, per-sample preds, results JSON
├── figures/               PDF + PNG figures for thesis and slides
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10 or newer
- `pip install -r requirements.txt`
- Optional: CUDA-capable GPU (LSTM training falls back to CPU automatically; approximately 5 minutes for 80 epochs on CPU)

## Quick start — reproduce everything

Run the following commands in order from the project root. Each step writes its outputs to disk and the next step reads them; no manual file management is required.

```bash
python data_generator.py --n_total 10000 --attack_ratio 0.20 --seed 42
python lstm_detector.py
python lstm_detector.py --ablate
python evaluate.py
python generate_figures.py
```

End-to-end runtime: approximately 10 minutes on CPU.

## Detailed usage

### data_generator.py

Generates the synthetic DRM session dataset and writes normalised train/val/test splits to `data/`.

| Flag | Default | Description |
|---|---|---|
| `--n_total` | 10000 | Total number of sessions to generate |
| `--attack_ratio` | 0.20 | Fraction of sessions that are attack sessions |
| `--seed` | 42 | Global RNG seed for full reproducibility |
| `--out_dir` | `data` | Output directory |

Outputs: `data/sessions.npz`, `data/sessions_meta.csv`, `data/norm_stats.json`, `data/feature_names.json`.

Verification: the script prints class balance and a check that approximately 30% of normal training sessions contain at least one non-zero reuse interval (the deliberate overlap that prevents trivial separation).

### lstm_detector.py

Trains the LSTM anomaly detector. Each epoch tunes the classification threshold on the validation set; the checkpoint with the best validation F1 is restored at the end of training.

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 80 | Maximum training epochs |
| `--patience` | 15 | Early-stopping patience in epochs |
| `--hidden_size` | 64 | LSTM hidden units |
| `--lr` | 1e-3 | Adam learning rate |
| `--batch_size` | 128 | Mini-batch size |
| `--seed` | 42 | RNG seed (Python, NumPy, PyTorch) |
| `--ablate` | off | Drop temporal features (delta_t, reuse_interval) for the H2 ablation run |
| `--data_dir` | `data` | Directory containing `sessions.npz` |

Without `--ablate`: writes `models/lstm_full.pt`, `models/lstm_full_results.json`, `models/lstm_full_preds.npz`.

With `--ablate`: writes the same set prefixed `lstm_ablated_*`. The ablated model trains on only the two non-temporal features (device bucket, position decile), providing the H2 counterfactual.

Verification: the script prints a per-epoch progress table; the final line reports the best epoch, threshold, and test metrics. Expected full-model F1 ≥ 0.97.

### evaluate.py

Runs all five detectors on the held-out test set and writes a combined results file. If `data/sessions.npz` or the LSTM result files are missing it re-runs the prerequisite scripts automatically via subprocess.

Outputs: `models/all_results.json` containing per-detector metrics, tuned rule thresholds (normalised and in seconds), H1/H2 confirmation flags, and the McNemar contingency table.

Verification: the script prints a formatted results table and concludes with H1 and H2 assessment lines. Expected output includes `H1 (LSTM F1 >= 88%): CONFIRMED`.

### generate_figures.py

Reads saved artifacts only — no retraining. Briefly re-fits Isolation Forest and One-Class SVM (seconds) to extract anomaly scores for ROC curves.

```bash
python generate_figures.py                              # all six figures
python generate_figures.py --figure training_curves     # single figure
python generate_figures.py --out_dir my_dir/            # custom output directory
```

Allowed `--figure` values: `training_curves`, `confusion_matrices`, `roc_curves`, `reuse_distribution`, `f1_comparison`, `h2_ablation`.

Outputs: each figure written as both `figures/<name>.pdf` (vector, for LaTeX) and `figures/<name>.png` (300 dpi, for slides).

### show_logs.py

Prints two normal sessions and two replay-attack sessions as formatted event tables. Useful for screenshots in presentations. No arguments; no file output.

```bash
python show_logs.py
```

## Results

Test set: 1 500 sessions (1 200 normal, 300 attack), held out from the 10 000-session dataset.

| Detector | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|
| LSTM — full features (proposed) | 0.9897 | 0.9633 | 0.9764 | 0.9990 |
| LSTM — no temporal features (H2 ablation) | 0.2000 | 1.0000 | 0.3333 | 0.5126 |
| Rule-based (tuned threshold, 10–210 s window) | 0.9641 | 0.6267 | 0.7596 | N/A |
| Isolation Forest | 0.2049 | 0.1933 | 0.1990 | 0.6542 |
| One-Class SVM | 0.0909 | 0.0167 | 0.0282 | 0.5012 |

The full LSTM detector statistically significantly outperforms every baseline; see Hypotheses below.

## Hypotheses

### H1 — Detection accuracy

**Hypothesis:** The LSTM detector achieves F1 ≥ 0.88 on the held-out test set.

**Test:** Direct evaluation of F1 score on the 1 500-session test split.

**Outcome: CONFIRMED.** F1 = 0.9764, exceeding the 0.88 target by 9.6 percentage points.

### H2 — Temporal features matter

**Hypothesis:** Including temporal sequence features (inter-event delta, token reuse interval) improves F1 by at least 10 percentage points over the same LSTM architecture trained without them.

**Test:** Ablation study — identical architecture and hyperparameters, trained on only the two non-temporal features (device bucket, position decile).

**Outcome: CONFIRMED.** Full model F1 = 0.9764 vs ablated F1 = 0.3333; improvement = +64.3 pp, far exceeding the 10 pp threshold. The ablated model degenerates to predicting all sessions as attack (recall = 1.0, precision = 0.20, matching the dataset attack ratio).

### H0 — LSTM vs tuned rule are statistically equivalent

**Hypothesis (null):** The LSTM full model and the best tuned interval rule make errors on the same sessions (no significant difference in per-sample decisions).

**Test:** McNemar test on paired binary predictions over the 1 500-session test set.

**Outcome: REJECTED** at alpha = 0.05. McNemar statistic with continuity correction: b = 110 (LSTM correct, rule wrong), c = 5 (rule correct, LSTM wrong). p < 0.001. The two classifiers are significantly different; the LSTM commits far fewer errors.

## Dataset notes

The dataset is entirely synthetic, generated by `data_generator.py` with a fixed random seed to guarantee reproducibility. The generator deliberately injects legitimate licence reuse patterns into normal sessions — brief retry bursts (1–4 s) after simulated network drops and long-interval rewatches (200–500 s) — so that the reuse-interval feature alone cannot separate classes, forcing the LSTM to exploit the full temporal context of each session.

## Authors

- **Baimuratov N. N.** — Machine Learning: model design and architecture, feature engineering, training and hyperparameter tuning, evaluation methodology
- **Adambalinov Y. J.** — Systems and Research: system architecture, data generation pipeline, evaluation framework, literature review

**Supervisor:** A. E. Abiche, Master of Science, Astana IT University

## License

Academic use, Astana IT University 2026. Contact authors for other use.
