import os
import sys
import time
import numpy as np
import joblib
from skorch.callbacks import EarlyStopping
import torch
import torch.nn as nn
from torch import nn
from skorch import NeuralNetClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from codecarbon import EmissionsTracker

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_data_split, save_inference_time, save_results, load_best_params, get_nrows
from power_monitor import CPUPowerMonitor, compute_corrected_co2, print_cpu_summary
from config import BASE_DIR, config, RANDOM_STATE

EPOCHS = 200  # the earlyStopping will be reached before the 200 anyway
DATASET = sys.argv[1] if len(sys.argv) > 1 else "wine"


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

X_train, X_test, y_train, y_test = load_data_split(DATASET)

X_train_array = X_train.to_numpy().astype(np.float32)
X_test_array = X_test.to_numpy().astype(np.float32)
y_train_array = y_train.to_numpy().astype(np.int64)
y_test_array = y_test.to_numpy().astype(np.int64)

nrows = get_nrows(DATASET)
input_dim = X_train_array.shape[1]

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
cpu_monitor = CPUPowerMonitor()

cpu_monitor.start()
tracker.start()

start = time.time()
pipeline.fit(X_train_array, y_train_array)
training_time = time.time() - start
emissions_cc = tracker.stop()
cpu_result = cpu_monitor.stop()

co2_corrected = compute_corrected_co2(tracker, cpu_result)
print_cpu_summary(cpu_result, tracker.final_emissions_data.cpu_energy)

y_pred = pipeline.predict(X_test_array)
test_accuracy = accuracy_score(y_test_array, y_pred)
test_f1 = f1_score(y_test_array, y_pred, average="weighted")

saved_models_dir = BASE_DIR / "saved_models"
saved_models_dir.mkdir(exist_ok=True)
joblib.dump(pipeline, saved_models_dir / f"mlp_{DATASET}.joblib")

trained_model = joblib.load(saved_models_dir / f"mlp_{DATASET}.joblib")
single_row = X_test_array[:1]
trained_model.predict(single_row)  # warmup before monitor starts
inference_monitor = CPUPowerMonitor()
inference_monitor.start()
_t_start = time.perf_counter()
_times = []
while time.perf_counter() - _t_start < 30.0:
    _t0 = time.perf_counter()
    trained_model.predict(single_row)
    _times.append(time.perf_counter() - _t0)
inference_cpu_result = inference_monitor.stop()
inference_time = float(np.median(_times))
_n_inferences = len(_times)
_energy_per_inference_wh = (
    inference_cpu_result["energy_wh"] / _n_inferences
    if inference_cpu_result.get("energy_wh") is not None else None
)

save_results(
    "MLP_PyTorch",
    DATASET,
    test_accuracy,
    test_f1,
    co2_corrected,
    emissions_cc,
    cpu_result,
    training_time,
    nrows,
    tracker,
)
save_inference_time("MLP_PyTorch", DATASET, co2_corrected, nrows, inference_time,
                    inference_cpu_result.get("avg_watt"), _energy_per_inference_wh, _n_inferences)
