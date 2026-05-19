import os
import sys
import time
import numpy as np
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_data_split, save_results, save_inference_time, load_best_params, get_nrows
from power_monitor import CPUPowerMonitor, compute_corrected_co2, print_cpu_summary
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score
from codecarbon import EmissionsTracker
from config import BASE_DIR, config, RANDOM_STATE
import time

DATASET = sys.argv[1] if len(sys.argv) > 1 else "wine"

_tuned = load_best_params("xgb", DATASET)["best_params"]

xgb_base = {
    "wine":   {"objective": "multi:softmax", "num_class": 3, "eval_metric": "mlogloss"},
    "credit": {"objective": "binary:logistic", "eval_metric": "logloss"},
    "higgs":  {"objective": "binary:logistic", "eval_metric": "logloss"},
}

X_train, X_test, y_train, y_test = load_data_split(DATASET)
nrows = get_nrows(DATASET)

model = XGBClassifier(**xgb_base[DATASET], **_tuned, random_state=RANDOM_STATE)

tracker = EmissionsTracker(
    output_dir=str(BASE_DIR / "emissions"), project_name=f"xgb_{DATASET}"
)
cpu_monitor = CPUPowerMonitor()

cpu_monitor.start()
tracker.start()

start = time.time()
model.fit(X_train, y_train)
training_time = time.time() - start
emissions_cc = tracker.stop()
cpu_result = cpu_monitor.stop()

co2_corrected = compute_corrected_co2(tracker, cpu_result)
print_cpu_summary(cpu_result, tracker.final_emissions_data.cpu_energy)

y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred, average="weighted")

saved_models_dir = BASE_DIR / "saved_models"
saved_models_dir.mkdir(exist_ok=True)
joblib.dump(model, saved_models_dir / f"xgb_cpu_{DATASET}.joblib")

trained_model = joblib.load(saved_models_dir / f"xgb_cpu_{DATASET}.joblib")
single_row = X_test[:1]
inference_monitor = CPUPowerMonitor()
inference_monitor.start()
trained_model.predict(single_row)  # warmup
_times = []
for _ in range(100):
    _t0 = time.perf_counter()
    trained_model.predict(single_row)
    _times.append(time.perf_counter() - _t0)
inference_time = float(np.median(_times))
inference_cpu_result = inference_monitor.stop()

save_results(
    "XGBoost",
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
save_inference_time("XGBoost", DATASET, co2_corrected, nrows, inference_time, inference_cpu_result.get("avg_watt"))
