"""
Compares WF1 vs EWF1 Optuna HPO on a 1M-row stratified HIGGS subset.

Study A: maximize WF1  (standard)
Study B: maximize EWF1 = WF1 / (1 + LAMBDA * co2_g)
         where co2_g is hardware-corrected CO2 in grams for that trial.
         No normalization needed — the penalty is absolute and computed
         directly per trial. LAMBDA=0.1 is chosen so that a typical
         expensive trial (~10g) yields a denominator of ~2, comparable
         to the min-max range used in the thesis.

Post-hoc: pools all 80 trials, applies the same EWF1 formula to every
trial uniformly, and reports which config each approach found.

Usage:
    python scripts/tune_mlp_ewf1_comparison.py [wf1|ewf1|both|analyze]
    Default: both

Resume: safe to re-run — Optuna picks up from its SQLite DB and the
trial CSV is append-only.
"""
import csv
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import optuna
import requests
import torch
import torch.nn as nn
from codecarbon import EmissionsTracker
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping

from config import BASE_DIR, CV_FOLDS, RANDOM_STATE
from models.power_monitor import CPUPowerMonitor, compute_corrected_co2
from models.utils import load_data

LAMBDA = 0.1        # penalty weight; co2 in grams → denominator ~ [1.2, 2.5] for typical trials
N_TRIALS = 40
HIGGS_NROWS = 500_000

RESULTS_DIR = BASE_DIR / "results" / "ewf1_comparison"
EMISSIONS_DIR = BASE_DIR / "emissions" / "ewf1_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Data ──────────────────────────────────────────────────────────────────────
print(f"Loading HIGGS {HIGGS_NROWS:,} rows (stratified)...")
os.environ["TEST_NROWS"] = str(HIGGS_NROWS)
X, y = load_data("higgs")
X_array = X.to_numpy().astype(np.float32)
y_array = y.to_numpy().astype(np.int64)
input_dim = X_array.shape[1]
device = "cuda" if torch.cuda.is_available() else "cpu"
cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
wf1_scorer = make_scorer(f1_score, average="weighted")
print(f"Loaded: {X_array.shape}, device={device}\n")


# ── MLP (identical search space to existing tune_mlp.py) ─────────────────────
class MLPModule(nn.Module):
    def __init__(self, input_dim, num_classes=2, layer_sizes=[256, 128], dropout_rate=0.2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for out_dim in layer_sizes:
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, X):
        return self.network(X)


def build_pipeline(trial):
    n_layers = trial.suggest_int("n_layers", 1, 4)
    layer_sizes = [
        trial.suggest_categorical(f"layer_{i}", [64, 128, 256, 512])
        for i in range(n_layers)
    ]
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [1024, 4096, 8192])

    net = NeuralNetClassifier(
        module=MLPModule,
        module__input_dim=input_dim,
        module__num_classes=2,
        module__layer_sizes=layer_sizes,
        module__dropout_rate=dropout_rate,
        max_epochs=50,
        lr=lr,
        iterator_train__batch_size=batch_size,
        iterator_valid__batch_size=batch_size,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        iterator_train__shuffle=True,
        device=device,
        verbose=0,
        callbacks=[EarlyStopping(patience=10)],
    )
    return make_pipeline(StandardScaler(), net)


# ── Per-trial measurement ─────────────────────────────────────────────────────
def measure_trial(trial, study_type):
    """Train one trial; return (wf1, co2_g, duration_s)."""
    pipe = build_pipeline(trial)
    tracker = EmissionsTracker(
        project_name=f"ewf1_cmp_{study_type}_t{trial.number}",
        output_dir=str(EMISSIONS_DIR),
        log_level="error",
        save_to_file=True,
    )
    cpu_monitor = CPUPowerMonitor()
    cpu_monitor.start()
    tracker.start()
    t0 = time.time()

    wf1 = cross_val_score(pipe, X_array, y_array, cv=cv, scoring=wf1_scorer).mean()

    duration_s = time.time() - t0
    tracker.stop()
    cpu_result = cpu_monitor.stop()
    co2_g = compute_corrected_co2(tracker, cpu_result) * 1000  # kg → g

    return float(wf1), float(co2_g), float(duration_s)


# ── EWF1 formula ─────────────────────────────────────────────────────────────
def ewf1(wf1, co2_g):
    return wf1 / (1 + LAMBDA * co2_g)


# ── Trial logging ─────────────────────────────────────────────────────────────
_FIELDS = ["trial", "wf1", "co2_g", "ewf1", "duration_s", "params"]


def log_trial(study_type, trial_number, wf1, co2_g, duration_s, params):
    path = RESULTS_DIR / f"trials_{study_type}.csv"
    file_exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "trial": trial_number,
            "wf1": round(wf1, 6),
            "co2_g": round(co2_g, 4),
            "ewf1": round(ewf1(wf1, co2_g), 6),
            "duration_s": round(duration_s, 1),
            "params": json.dumps(params),
        })


# ── Objectives ────────────────────────────────────────────────────────────────
def make_wf1_objective():
    def objective(trial):
        wf1, co2_g, duration_s = measure_trial(trial, "wf1")
        log_trial("wf1", trial.number, wf1, co2_g, duration_s, trial.params)
        print(
            f"  [WF1 ] t{trial.number:02d}: WF1={wf1:.4f}  "
            f"CO2={co2_g:.2f}g  EWF1={ewf1(wf1, co2_g):.4f}  {duration_s/60:.1f}min"
        )
        return wf1
    return objective


def make_ewf1_objective():
    def objective(trial):
        wf1, co2_g, duration_s = measure_trial(trial, "ewf1")
        score = ewf1(wf1, co2_g)
        log_trial("ewf1", trial.number, wf1, co2_g, duration_s, trial.params)
        print(
            f"  [EWF1] t{trial.number:02d}: WF1={wf1:.4f}  "
            f"CO2={co2_g:.2f}g  EWF1={score:.4f}  {duration_s/60:.1f}min"
        )
        return score
    return objective


# ── Run a study ───────────────────────────────────────────────────────────────
def run_study(study_type):
    print(f"\n{'='*62}")
    print(f"  Study: {study_type.upper()}  |  {N_TRIALS} trials  |  HIGGS {HIGGS_NROWS:,} rows  |  λ={LAMBDA}")
    print(f"{'='*62}")

    objective = make_wf1_objective() if study_type == "wf1" else make_ewf1_objective()

    study = optuna.create_study(
        study_name=f"mlp_higgs1m_{study_type}",
        storage=f"sqlite:///{BASE_DIR}/optuna_ewf1_cmp_{study_type}.db",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        load_if_exists=True,
    )

    already_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = N_TRIALS - already_done
    if remaining <= 0:
        print(f"  Already complete ({already_done} trials). Skipping.")
        return

    if already_done:
        print(f"  Resuming from trial {already_done} ({remaining} remaining)")

    t_start = time.time()
    study.optimize(objective, n_trials=remaining)
    total_min = (time.time() - t_start) / 60

    best = study.best_trial
    print(f"\n  Done in {total_min:.1f} min")
    print(f"  Best trial {best.number}: objective={study.best_value:.4f}")
    print(f"  Best params: {best.params}")

    with open(RESULTS_DIR / f"best_{study_type}.json", "w") as f:
        json.dump({
            "study_type": study_type,
            "best_trial": best.number,
            "best_objective": study.best_value,
            "best_params": best.params,
            "total_min": round(total_min, 1),
            "n_trials": already_done + remaining,
            "lambda": LAMBDA,
        }, f, indent=2)


# ── Post-hoc analysis ─────────────────────────────────────────────────────────
def analyze():
    rows = []
    for study_type in ["wf1", "ewf1"]:
        path = RESULTS_DIR / f"trials_{study_type}.csv"
        if not path.exists():
            print(f"  No trial log for {study_type}, skipping.")
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                rows.append({
                    "study_type": study_type,
                    "trial": int(row["trial"]),
                    "wf1": float(row["wf1"]),
                    "co2_g": float(row["co2_g"]),
                    "ewf1": float(row["ewf1"]),
                    "duration_s": float(row["duration_s"]),
                    "params": row["params"],
                })

    if not rows:
        print("  No trial data. Run studies first.")
        return

    print(f"\n{'='*62}")
    print(f"  Post-hoc comparison  |  λ={LAMBDA}  |  {len(rows)} total trials")
    print(f"{'='*62}")

    for study_type in ["wf1", "ewf1"]:
        sr = [r for r in rows if r["study_type"] == study_type]
        if not sr:
            continue
        best_ewf1 = max(sr, key=lambda r: r["ewf1"])
        best_wf1  = max(sr, key=lambda r: r["wf1"])
        mean_co2  = sum(r["co2_g"] for r in sr) / len(sr)

        print(f"\n  [{study_type.upper()}]  n={len(sr)}  mean CO2={mean_co2:.2f}g")
        print(f"    Best by EWF1 → t{best_ewf1['trial']:02d}: "
              f"WF1={best_ewf1['wf1']:.4f}  CO2={best_ewf1['co2_g']:.2f}g  EWF1={best_ewf1['ewf1']:.4f}")
        print(f"    Best by WF1  → t{best_wf1['trial']:02d}: "
              f"WF1={best_wf1['wf1']:.4f}  CO2={best_wf1['co2_g']:.2f}g  EWF1={best_wf1['ewf1']:.4f}")
        print(f"    Params (best EWF1): {best_ewf1['params']}")

    out = RESULTS_DIR / "all_trials.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["study_type", "trial", "wf1", "co2_g", "ewf1", "duration_s", "params"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Saved to {out}")


# ── Notification ──────────────────────────────────────────────────────────────
def notify(msg):
    try:
        requests.post(
            "https://ntfy.sh/eron_thesis_higgs_run_123",
            data=msg.encode("utf-8"),
            headers={"Title": "Thesis: EWF1 Comparison", "Priority": "default", "Tags": "tada"},
            timeout=5,
        )
    except Exception:
        pass


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    if mode not in ("wf1", "ewf1", "both", "analyze"):
        print(f"Unknown mode '{mode}'. Use: wf1 | ewf1 | both | analyze")
        sys.exit(1)

    if mode in ("wf1", "both"):
        run_study("wf1")
        notify(f"WF1 study fertig! ({N_TRIALS} trials, HIGGS 1M)")

    if mode in ("ewf1", "both"):
        run_study("ewf1")
        notify(f"EWF1 study fertig! ({N_TRIALS} trials, HIGGS 1M)")

    if mode in ("both", "analyze"):
        analyze()
        notify("EWF1-Vergleich komplett!")

    if not os.environ.get("NO_SHUTDOWN") and mode == "both":
        os.system("shutdown /s /t 30")
        print("\nPC wird in 30 Sekunden heruntergefahren. 'shutdown /a' zum Abbrechen.")
