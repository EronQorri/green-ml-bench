"""
run_mlp_variation.py — MLP architecture variation experiment.

Trains 12 MLP configurations on HIGGS (3 widths x 4 depths) to quantify how
architectural complexity affects CO2 emissions and F1-score.

Non-architectural hyperparameters are set to neutral, architecture-independent
defaults rather than the HIGGS tuning values. The tuning optimised lr,
dropout_rate, and batch_size jointly with a specific architecture, so reusing
them would implicitly favour that architecture. Chosen defaults:
    lr            = 1e-3   (Adam default, Kingma & Ba 2015)
    dropout_rate  = 0.3    (standard mid-range regularisation)
    batch_size    = 4096   (mid-range of the Optuna search space)

Must be run as Administrator for CPUPowerMonitor sensor access.
"""

import sys
import time
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
from codecarbon import EmissionsTracker
from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping

_ROOT = Path(__file__).parent.parent
sys.path.append(str(_ROOT))
sys.path.append(str(_ROOT / "models"))

from config import BASE_DIR, RANDOM_STATE
from power_monitor import CPUPowerMonitor, compute_corrected_co2, print_cpu_summary
from utils import load_data_split, save_inference_time, save_results

DATASET = "higgs"
EPOCHS = 200
NTFY_CHANNEL = "eron_thesis_higgs_run_123"

FIXED_LR = 1e-3
FIXED_DROPOUT = 0.3
FIXED_BATCH_SIZE = 4096

# 3 widths x 4 depths = 12 configs, width-first so each depth curve completes
# before moving to the next width
ARCHITECTURES = [
    (w,) * d
    for w in [128, 256, 512]
    for d in [1, 2, 3, 4]
]


class MLPModule(nn.Module):
    def __init__(self, input_dim, num_classes, layer_sizes, dropout_rate):
        super().__init__()
        layers = []
        in_dim = input_dim
        for out_dim in layer_sizes:
            layers += [nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(), nn.Dropout(dropout_rate)]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, X):
        return self.network(X)


def run_arch(layer_sizes, X_train, X_test, y_train, y_test, input_dim, num_classes, nrows, device):
    torch.manual_seed(RANDOM_STATE)
    label = f"{len(layer_sizes)}x{layer_sizes[0]}"
    model_name = f"MLP_{label}"
    print(f"\n>>> {model_name} | layers={list(layer_sizes)}")

    net = NeuralNetClassifier(
        module=MLPModule,
        module__input_dim=input_dim,
        module__num_classes=num_classes,
        module__layer_sizes=layer_sizes,
        module__dropout_rate=FIXED_DROPOUT,
        max_epochs=EPOCHS,
        lr=FIXED_LR,
        iterator_train__batch_size=FIXED_BATCH_SIZE,
        iterator_valid__batch_size=FIXED_BATCH_SIZE,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        iterator_train__shuffle=True,
        device=device,
        verbose=0,
        callbacks=[EarlyStopping(patience=10)],
    )
    pipeline = make_pipeline(StandardScaler(), net)

    tracker = EmissionsTracker(
        output_dir=str(BASE_DIR / "emissions"),
        project_name=f"mlp_var_{label}_{DATASET}",
    )
    cpu_monitor = CPUPowerMonitor()

    cpu_monitor.start()
    tracker.start()
    t0 = time.time()

    pipeline.fit(X_train, y_train)

    training_time = time.time() - t0
    emissions_cc = tracker.stop()
    cpu_result = cpu_monitor.stop()
    co2 = compute_corrected_co2(tracker, cpu_result)
    print_cpu_summary(cpu_result, tracker.final_emissions_data.cpu_energy)

    y_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average="weighted")

    single_row = X_test[:1]
    pipeline.predict(single_row)  # warmup
    _times = []
    for _ in range(100):
        _t0 = time.perf_counter()
        pipeline.predict(single_row)
        _times.append(time.perf_counter() - _t0)
    inference_time = float(np.median(_times))

    save_results(
        model_name, DATASET,
        test_accuracy,
        test_f1,
        co2, emissions_cc, cpu_result,
        training_time, nrows,
        tracker,
    )
    save_inference_time(model_name, DATASET, co2, nrows, inference_time)

    print(f"    OK — {training_time / 60:.1f} min | F1={test_f1:.4f} | CO2={co2:.2e} kg")
    return True


def _notify(title, body):
    try:
        requests.post(
            f"https://ntfy.sh/{NTFY_CHANNEL}",
            data=body.encode("utf-8"),
            headers={"Title": title, "Priority": "default", "Tags": "bell"},
            timeout=5,
        )
    except Exception:
        pass


# ── Main ──────────────────────────────────────────────────────────────────────

overall_start = time.time()
failures = []

print("=" * 60)
print("MLP Architecture Variation — HIGGS")
print(f"Architectures: {[f'{len(a)}x{a[0]}' for a in ARCHITECTURES]}")
print(f"Fixed: lr={FIXED_LR}, dropout={FIXED_DROPOUT}, batch_size={FIXED_BATCH_SIZE}")
print("=" * 60)

print(f"Loading {DATASET}...")
X_train_df, X_test_df, y_train_df, y_test_df = load_data_split(DATASET)
X_train = X_train_df.to_numpy().astype(np.float32)
X_test = X_test_df.to_numpy().astype(np.float32)
y_train = y_train_df.to_numpy().astype(np.int64)
y_test = y_test_df.to_numpy().astype(np.int64)
input_dim = X_train.shape[1]
num_classes = int(np.unique(y_train).size)
nrows = len(X_train) + len(X_test)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"device={device} | input_dim={input_dim} | nrows={nrows:,}")

for layer_sizes in ARCHITECTURES:
    try:
        run_arch(layer_sizes, X_train, X_test, y_train, y_test, input_dim, num_classes, nrows, device)
        _notify(f"MLP_{'{}x{}'.format(len(layer_sizes), layer_sizes[0])} [{DATASET}] fertig", "")
    except Exception as e:
        label = f"{len(layer_sizes)}x{layer_sizes[0]}"
        print(f"    FAILED: {label} — {e}")
        failures.append(label)

total_min = (time.time() - overall_start) / 60
print("\n" + "=" * 60)
print(f"DONE — {total_min:.1f} min total")
print("No failures." if not failures else f"Failures: {failures}")
print("=" * 60)

_notify(
    "Thesis: MLP Variation fertig!",
    f"Dauer: {total_min:.1f} min. Failures: {failures or 'keine'}",
)

