import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_data, save_results, save_inference_time, load_best_params, get_nrows
from power_monitor import CPUPowerMonitor, compute_corrected_co2, print_cpu_summary
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import f1_score, make_scorer
from codecarbon import EmissionsTracker
from config import BASE_DIR, config, RANDOM_STATE, CV_FOLDS
import time

DATASET = sys.argv[1] if len(sys.argv) > 1 else "wine"

cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

_tuned = load_best_params("xgb", DATASET)["best_params"]

xgb_base = {
    "wine":   {"objective": "multi:softmax", "num_class": 3, "eval_metric": "mlogloss"},
    "credit": {"objective": "binary:logistic", "eval_metric": "logloss"},
    "higgs":  {"objective": "binary:logistic", "eval_metric": "logloss"},
}

X, y = load_data(DATASET)
nrows = get_nrows(DATASET)

model = XGBClassifier(**xgb_base[DATASET], **_tuned, random_state=RANDOM_STATE)

tracker = EmissionsTracker(
    output_dir=str(BASE_DIR / "emissions"), project_name=f"xgb_{DATASET}"
)
cpu_monitor = CPUPowerMonitor()

cpu_monitor.start()
tracker.start()

start = time.time()
cv_results = cross_validate(
    model,
    X,
    y,
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
single_row = X[:1]

start_inference = time.perf_counter()
_ = trained_model.predict(single_row)
inference_time = time.perf_counter() - start_inference

save_results(
    "XGBoost",
    DATASET,
    cv_results["test_accuracy"].mean(),
    cv_results["test_f1"].mean(),
    co2_corrected,
    emissions_cc,
    cpu_result,
    training_time,
    nrows,
)
save_inference_time("XGBoost", DATASET, co2_corrected, nrows, inference_time)
