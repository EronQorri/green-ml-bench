import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_data, save_results, minimal_preprocess

# Neue Imports für die Baseline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score, make_scorer
from codecarbon import EmissionsTracker
from config import config, RANDOM_STATE, CV_FOLDS
import time

DATASETS = ['wine', 'credit', 'higgs']

xgb_config = {
    "wine":   {"objective": "multi:softmax", "num_class": 3, "eval_metric": "mlogloss"},
    "credit": {"objective": "binary:logistic", "eval_metric": "logloss"},
    "higgs":  {"objective": "binary:logistic", "eval_metric": "logloss"},
}

scoring = {
    'accuracy': 'accuracy',
    'f1': make_scorer(f1_score, average='weighted')
}

for dataset in DATASETS:
    X, y = load_data(dataset)
    X, y = minimal_preprocess(X, y)
    nrows = config[dataset].get("nrows")

    lr_pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver='sag', random_state=RANDOM_STATE, n_jobs=-1, max_iter=1000)
    )

    for model_name, clf in [
        ("LogisticRegression", lr_pipeline),
        ("RandomForest", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        ("XGBoost",      XGBClassifier(**xgb_config[dataset], random_state=RANDOM_STATE)),
        ("XGBoost_GPU",  XGBClassifier(**xgb_config[dataset], device='cuda', random_state=RANDOM_STATE)),
    ]:
        print(f"Running {model_name} on {dataset}...")
        
        tracker_name = f"{model_name.replace(' ', '_')}_{dataset}"
        tracker = EmissionsTracker(output_dir="emissions", project_name=tracker_name)
        
        tracker.start()
        start = time.time()
        cv_results = cross_validate(clf, X, y, cv=CV_FOLDS, scoring=scoring)
        training_time = time.time() - start
        emissions = tracker.stop()
        
        save_results(model_name, dataset, cv_results['test_accuracy'].mean(), cv_results['test_f1'].mean(), emissions, training_time, nrows)
        print(f"Done: acc={cv_results['test_accuracy'].mean():.4f}, time={training_time:.2f}s\n")