import os
import sys
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from models.utils import load_data, save_best_params
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from config import config, RANDOM_STATE, CV_FOLDS, BASE_DIR
import optuna
import torch
import torch.nn as nn
import numpy as np
import requests
import time
import json
from codecarbon import EmissionsTracker

# NOTE: MLP Higgs tuning (run 2026-04-21 22:33 → crash 2026-04-26 11:15, PC MCE/overheat)
# Completed 45/49 trials in 108.70 h before crash. Best F1 seen: 0.7807 (params not recorded).
# Best params for higgs in best_params.json are taken from MLP credit run (fallback).
# CO₂ extrapolated to full 49 trials: 118.36 h × 68.83 W (avg MLP hw power) = 8146.6 Wh → 3.1039 kg CO₂
# Method: linear scale by trial ratio (49/45); power from HardwareMonitor MLP wine+credit avg.

DATASET = sys.argv[1] if len(sys.argv) > 1 else "wine"
N_TRIALS = int(os.environ.get("N_TRIALS", 40))
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
cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

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
        module__num_classes=num_classes,
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

    pipe = make_pipeline(StandardScaler(), net)
    return cross_val_score(
        pipe, X_array, y_array, cv=cv, scoring=make_scorer(f1_score, average="weighted")
    ).mean()

start_time = time.time()

tracker = EmissionsTracker(
    project_name=f"tune_mlp_{DATASET}",
    output_dir=str(BASE_DIR / "emissions"),
    log_level="error",
)
tracker.start()

study = optuna.create_study(
    study_name=f"mlp_{DATASET}",
    storage=f"sqlite:///{BASE_DIR}/optuna_mlp_{DATASET}.db",
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    load_if_exists=True,
)
study.optimize(objective, n_trials=N_TRIALS)

emissions_kg = tracker.stop()

total_time = (time.time() - start_time) / 60

print(f"\nBest F1: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
print(f"Tuning duration: {total_time:.2f} minutes")

save_best_params("mlp", DATASET, {
    "best_f1": study.best_value,
    "best_params": study.best_params,
    "tuning_duration_min": total_time,
    "emissions_kg": emissions_kg,
    "n_trials": N_TRIALS,
})
print("Best params saved to models/best_params.json")

if not os.environ.get("NO_NOTIFY"):
    try:
        requests.post(
            f"https://ntfy.sh/eron_thesis_higgs_run_123",
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

if not os.environ.get("NO_SHUTDOWN"):
    os.system("shutdown /s /t 30")
    print("PC wird in 30 Sekunden heruntergefahren. 'shutdown /a' zum Abbrechen.")