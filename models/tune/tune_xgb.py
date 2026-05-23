import os
import sys
import time
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_data_split, save_best_params, save_results, get_nrows
from power_monitor import CPUPowerMonitor, compute_corrected_co2
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer
from config import config, RANDOM_STATE, CV_FOLDS
from codecarbon import EmissionsTracker
import optuna

DATASET = sys.argv[1] if len(sys.argv) > 1 else "wine"
N_TRIALS = int(os.environ.get("N_TRIALS", 40))

X_train, X_test, y_train, y_test = load_data_split(DATASET)
nrows = get_nrows(DATASET)
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

xgb_base = {
    "wine": {"objective": "multi:softmax", "num_class": 3, "eval_metric": "mlogloss", "device": "cuda"},
    "credit": {"objective": "binary:logistic", "eval_metric": "logloss", "device": "cuda"},
    "higgs": {"objective": "binary:logistic", "eval_metric": "logloss", "device": "cuda"},
}


def objective(trial):
    params = {
        **xgb_base[DATASET],
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "random_state": RANDOM_STATE,
    }
    model = XGBClassifier(**params)
    return cross_val_score(
        model, X_train, y_train, cv=cv, scoring=make_scorer(f1_score, average="weighted")
    ).mean()


start_time = time.time()

EMISSIONS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "emissions"
)
os.makedirs(EMISSIONS_DIR, exist_ok=True)

tracker = EmissionsTracker(
    project_name=f"tune_xgb_{DATASET}",
    output_dir=EMISSIONS_DIR,
    log_level="error",
)
cpu_monitor = CPUPowerMonitor()
cpu_monitor.start()
tracker.start()

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
)
study.optimize(objective, n_trials=N_TRIALS)

emissions = tracker.stop()
cpu_result = cpu_monitor.stop()
training_time_s = time.time() - start_time
total_time = training_time_s / 60
co2_corrected = compute_corrected_co2(tracker, cpu_result)

print(f"\nBest F1: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
print(f"Tuning duration: {total_time:.2f} minutes")
print(f"Emissions: {emissions:.6e} kg CO2eq")

save_results("tune_XGB", DATASET, None, study.best_value, co2_corrected, emissions, cpu_result, training_time_s, nrows, tracker)

save_best_params("xgb", DATASET, {
    "best_f1": study.best_value,
    "best_params": study.best_params,
    "tuning_duration_min": total_time,
    "emissions_kg": emissions,
    "n_trials": N_TRIALS,
})
print("Best params saved to models/best_params.json")
