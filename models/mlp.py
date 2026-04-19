import os
import sys
import time
import numpy as np
from skorch.callbacks import EarlyStopping
import torch
import torch.nn as nn
from torch import nn
from skorch import NeuralNetClassifier
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from codecarbon import EmissionsTracker

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_data, save_inference_time, save_results, load_best_params
from power_monitor import CPUPowerMonitor, compute_corrected_co2, print_cpu_summary
from config import BASE_DIR, config, RANDOM_STATE, CV_FOLDS

EPOCHS = 200  # the earlyStopping will be reached before the 200 anyway
DATASET = sys.argv[1] if len(sys.argv) > 1 else "wine"

cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


class MLPModule(nn.Module):
    # parameters won't be used anyway as they are overwritten by the neuralnet
    def __init__(
        self, input_dim, num_classes, layer_sizes=[256, 128], dropout_rate=0.2
    ):
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


num_classes = {"wine": 3, "credit": 2, "higgs": 2}[DATASET]

_p = load_best_params("mlp", DATASET)["best_params"]
layer_sizes = [_p[f"layer_{i}"] for i in range(_p["n_layers"])]
dropout_rate = _p["dropout_rate"]
lr = _p["lr"]
batch_size = _p["batch_size"]

X, y = load_data(DATASET)

X_array = X.to_numpy().astype(np.float32)
y_array = y.to_numpy().astype(np.int64)

nrows = config[DATASET].get("nrows")
input_dim = X_array.shape[1]

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(RANDOM_STATE)

net = NeuralNetClassifier(
    module=MLPModule,
    module__input_dim=input_dim,
    module__num_classes=num_classes,
    module__layer_sizes=layer_sizes,
    module__dropout_rate=dropout_rate,
    max_epochs=EPOCHS,
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

pipeline = make_pipeline(StandardScaler(), net)

tracker = EmissionsTracker(
    output_dir=str(BASE_DIR / "emissions"), project_name=f"mlp_{DATASET}"
)
cpu_monitor = CPUPowerMonitor(interval=0.5)

cpu_monitor.start()
tracker.start()

start = time.time()
cv_results = cross_validate(
    pipeline,
    X_array,
    y_array,
    cv=cv,
    scoring={"accuracy": "accuracy", "f1": make_scorer(f1_score, average="weighted")},
    return_estimator=True,
)
training_time = time.time() - start
emissions_cc = tracker.stop()
cpu_result = cpu_monitor.stop()

co2_corrected = compute_corrected_co2(tracker, cpu_result)
print_cpu_summary(cpu_result, tracker.final_emissions_data.cpu_energy)


trained_model = cv_results["estimator"][0]
single_row = X_array[:1]

start_inference = time.perf_counter()
_ = trained_model.predict(single_row)
inference_time = time.perf_counter() - start_inference

save_results(
    "MLP_PyTorch",
    DATASET,
    cv_results["test_accuracy"].mean(),
    cv_results["test_f1"].mean(),
    co2_corrected,
    emissions_cc,
    cpu_result,
    training_time,
    nrows,
)
save_inference_time("MLP_PyTorch", DATASET, co2_corrected, nrows, inference_time)
