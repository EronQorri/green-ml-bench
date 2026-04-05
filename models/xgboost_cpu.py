import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_data, save_results, save_inference_time
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import f1_score, make_scorer
from codecarbon import EmissionsTracker
from config import BASE_DIR, config, RANDOM_STATE, CV_FOLDS
import time

DATASET = sys.argv[1] if len(sys.argv) > 1 else "wine"

cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

xgb_config = {
    "wine": {
        "objective": "multi:softmax",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "n_estimators": 464,
        "max_depth": 7,
        "learning_rate": 0.0139,
        "subsample": 0.779,
        "colsample_bytree": 0.750,
        "min_child_weight": 5,
    },
    "credit": {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 108,
        "max_depth": 3,
        "learning_rate": 0.2878,
        "subsample": 0.964,
        "colsample_bytree": 0.843,
        "min_child_weight": 7,
    },
    "higgs": {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 479,
        "max_depth": 8,
        "learning_rate": 0.0350,
        "subsample": 0.829,
        "colsample_bytree": 0.819,
        "min_child_weight": 4,
    },
}

X, y = load_data(DATASET)
nrows = config[DATASET].get("nrows")

model = XGBClassifier(**xgb_config[DATASET], random_state=RANDOM_STATE)

tracker = EmissionsTracker(
    output_dir=str(BASE_DIR / "emissions"), project_name=f"xgb_{DATASET}"
)
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
emissions = tracker.stop()

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
    emissions,
    training_time,
    nrows,
)
save_inference_time("XGBoost", DATASET, emissions, nrows, inference_time)
