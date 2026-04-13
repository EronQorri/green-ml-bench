import os
import sys
import argparse
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from models.utils import load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from config import RANDOM_STATE
import optuna
import torch
import torch.nn as nn
import numpy as np
import requests
import time
import json
from codecarbon import EmissionsTracker

def parse_args():
    parser = argparse.ArgumentParser(description="Tune MLP hyperparameters with Optuna")
    parser.add_argument("dataset", nargs="?", default="wine", choices=["wine", "credit", "higgs"])
    parser.add_argument("--n-trials", type=int, default=None, help="Number of Optuna trials")
    parser.add_argument("--max-epochs", type=int, default=None, help="Max epochs per trial")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument(
        "--tune-sample-size",
        type=int,
        default=None,
        help="Use a stratified subset for tuning (e.g. 1000000). Uses full data if not set.",
    )
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split size")
    parser.add_argument("--timeout", type=int, default=None, help="Optional timeout in seconds")
    return parser.parse_args()


args = parse_args()
DATASET = args.dataset

mlp_config = {
    "wine": {"num_classes": 3},
    "credit": {"num_classes": 2},
    "higgs": {"num_classes": 2},
}
X, y = load_data(DATASET)
X_array = X.to_numpy().astype(np.float32)
y_array = y.to_numpy().astype(np.int64)
input_dim = X_array.shape[1]
num_classes = mlp_config[DATASET]["num_classes"]
device = "cuda" if torch.cuda.is_available() else "cpu"

if DATASET == "higgs":
    n_trials = args.n_trials if args.n_trials is not None else 20
    max_epochs = args.max_epochs if args.max_epochs is not None else 20
    patience = args.patience if args.patience is not None else 5
    tune_sample_size = args.tune_sample_size if args.tune_sample_size is not None else 1_000_000
else:
    n_trials = args.n_trials if args.n_trials is not None else 50
    max_epochs = args.max_epochs if args.max_epochs is not None else 50
    patience = args.patience if args.patience is not None else 10
    tune_sample_size = args.tune_sample_size

if tune_sample_size is not None and tune_sample_size < len(X_array):
    X_array, _, y_array, _ = train_test_split(
        X_array,
        y_array,
        train_size=tune_sample_size,
        random_state=RANDOM_STATE,
        stratify=y_array,
    )

X_train, X_val, y_train, y_val = train_test_split(
    X_array,
    y_array,
    test_size=args.val_size,
    random_state=RANDOM_STATE,
    stratify=y_array,
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val = scaler.transform(X_val).astype(np.float32)

class MLPModule(nn.Module):
    def __init__(self, input_dim, num_classes, layer_sizes=[256, 128], dropout_rate=0.2):
        super(MLPModule, self).__init__()
        layers = []
        current_in_dim = input_dim
        for out_dim in layer_sizes:
            layers.append(nn.Linear(current_in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_in_dim = out_dim
        layers.append(nn.Linear(current_in_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, X):
        return self.network(X)

def objective(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3 if DATASET == "higgs" else 4)
    layer_candidates = [64, 128, 256] if DATASET == "higgs" else [64, 128, 256, 512]
    layer_sizes = [
        trial.suggest_categorical(f"layer_{i}", layer_candidates)
        for i in range(n_layers)
    ]
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    batch_size_choices = [4096, 8192, 16384] if DATASET == "higgs" else [1024, 4096, 8192]
    batch_size = trial.suggest_categorical("batch_size", batch_size_choices)

    net = NeuralNetClassifier(
        module=MLPModule,
        module__input_dim=input_dim,
        module__num_classes=num_classes,
        module__layer_sizes=layer_sizes,
        module__dropout_rate=dropout_rate,
        max_epochs=max_epochs,
        lr=lr,
        iterator_train__batch_size=batch_size,
        iterator_valid__batch_size=batch_size,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        iterator_train__shuffle=True,
        device=device,
        verbose=0,
        callbacks=[EarlyStopping(patience=patience)],
    )

    net.fit(X_train, y_train)
    y_pred = net.predict(X_val)
    return f1_score(y_val, y_pred, average="weighted")

start_time = time.time()

tracker = EmissionsTracker(
    project_name=f"tune_mlp_{DATASET}",
    output_dir=os.path.dirname(os.path.abspath(__file__)),
    log_level="error",
)
tracker.start()

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
)
study.optimize(objective, n_trials=n_trials, timeout=args.timeout)

tracker.stop()

total_time = (time.time() - start_time) / 60

print(f"\nBest F1: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
print(f"Tuning duration: {total_time:.2f} minutes")
print(
    f"Config: dataset={DATASET}, n_trials={n_trials}, max_epochs={max_epochs}, "
    f"patience={patience}, tune_sample_size={tune_sample_size}, val_size={args.val_size}"
)

output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
params_path = os.path.join(output_dir, f"best_params_mlp_{DATASET}.json")
with open(params_path, "w") as f:
    json.dump({
        "best_f1": study.best_value,
        "best_params": study.best_params,
        "tuning_duration_min": total_time,
        "dataset": DATASET,
    }, f, indent=4)
print(f"Best params saved to {params_path}")

try:
    requests.post(
        "https://ntfy.sh/eron_thesis_higgs_run_123",
        data=f"MLP Tuning auf {DATASET} abgeschlossen! Dauer: {total_time:.2f} Minuten. Best F1: {study.best_value:.4f}".encode(
            encoding="utf-8"
        ),
        headers={
            "Title": f"Thesis: MLP Tuning {DATASET} fertig!",
            "Priority": "urgent",
            "Tags": "tada,party_popper",
        },
    )
except Exception as e:
    print(f"Konnte keine Benachrichtigung senden: {e}")

os.system("shutdown /s /t 30")
print("PC wird in 30 Sekunden heruntergefahren. 'shutdown /a' zum Abbrechen.")