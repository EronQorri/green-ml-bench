import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_data, save_results, save_inference_time
from power_monitor import CPUPowerMonitor, compute_corrected_co2, print_cpu_summary

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import f1_score, make_scorer
from codecarbon import EmissionsTracker
from config import config, RANDOM_STATE, CV_FOLDS, BASE_DIR

DATASET = sys.argv[1] if len(sys.argv) > 1 else "wine"
cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

X, y = load_data(DATASET)
nrows = config[DATASET].get("nrows")

pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(solver="sag", random_state=RANDOM_STATE, max_iter=1000),
)

tracker = EmissionsTracker(
    output_dir=str(BASE_DIR / "emissions"), project_name=f"lr_{DATASET}"
)
cpu_monitor = CPUPowerMonitor(interval=0.5)

cpu_monitor.start()
tracker.start()

start = time.time()
cv_results = cross_validate(
    pipeline,
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

# Take the first run in the cv and run a single row on it to measure the inference time
trained_model = cv_results["estimator"][0]
single_row = X[:1]

start_inference = time.perf_counter()
_ = trained_model.predict(single_row)
inference_time = time.perf_counter() - start_inference

save_results(
    "LogisticRegression",
    DATASET,
    cv_results["test_accuracy"].mean(),
    cv_results["test_f1"].mean(),
    co2_corrected,
    emissions_cc,
    cpu_result,
    training_time,
    nrows,
)
save_inference_time("LogisticRegression", DATASET, co2_corrected, nrows, inference_time)
