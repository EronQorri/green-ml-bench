# models/tune/tune_mlp.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import load_data
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from config import config, RANDOM_STATE, CV_FOLDS
import optuna
import torch
import torch.nn as nn
import numpy as np

DATASET = sys.argv[1] if len(sys.argv) > 1 else 'wine'

mlp_config = {
    "wine":   {"num_classes": 3},
    "credit": {"num_classes": 2},
    "higgs":  {"num_classes": 2},
}

X, y = load_data(DATASET)
X_array = X.to_numpy().astype(np.float32)
y_array = y.to_numpy().astype(np.int64)

input_dim  = X_array.shape[1]
num_classes = mlp_config[DATASET]["num_classes"]

cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    n_layers     = trial.suggest_int("n_layers", 1, 4)
    layer_sizes  = [
        trial.suggest_categorical(f"layer_{i}", [64, 128, 256, 512])
        for i in range(n_layers)
    ]
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    lr           = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    batch_size   = trial.suggest_categorical("batch_size", [256, 1024, 4096, 8192])
    patience     = trial.suggest_int("patience", 5, 20)

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
        callbacks=[EarlyStopping(patience=patience)],
    )

    pipe = make_pipeline(StandardScaler(), net)
    score = cross_val_score(
        pipe, X_array, y_array, cv=cv,
        scoring=make_scorer(f1_score, average='weighted')
    ).mean()
    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print(f"\nBest F1: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")