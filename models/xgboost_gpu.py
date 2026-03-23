import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_data, save_results, minimal_preprocess
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score, make_scorer
from codecarbon import EmissionsTracker
from config import config, RANDOM_STATE, CV_FOLDS
import time

DATASET = 'wine'

xgb_config = {
    "wine":   {"objective": "multi:softmax", "num_class": 3, "eval_metric": "mlogloss", "device": "cuda"},
    "credit": {"objective": "binary:logistic", "eval_metric": "logloss", "device": "cuda"},
    "higgs":  {"objective": "binary:logistic", "eval_metric": "logloss", "device": "cuda"},
}

X, y = load_data(DATASET)
X, y = minimal_preprocess(X, y)
nrows = config[DATASET].get("nrows")

tracker = EmissionsTracker(output_dir="emissions", project_name=f"xgb_gpu_{DATASET}")
tracker.start()

start = time.time()
cv_results = cross_validate(
    XGBClassifier(**xgb_config[DATASET], random_state=RANDOM_STATE),
    X, y, cv=CV_FOLDS,
    scoring={
        'accuracy': 'accuracy',
        'f1': make_scorer(f1_score, average='weighted')
    }
)
training_time = time.time() - start
emissions = tracker.stop()

save_results("XGBoost_GPU", DATASET, cv_results['test_accuracy'].mean(), cv_results['test_f1'].mean(), emissions, training_time, nrows)