import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_data, save_results, save_inference_time 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import f1_score, make_scorer
from codecarbon import EmissionsTracker
from config import BASE_DIR, config, RANDOM_STATE, CV_FOLDS


DATASET = sys.argv[1] if len(sys.argv) > 1 else 'wine'
if DATASET == 'higgs':
    print("Skipping Random Forest for Higgs dataset.")
    sys.exit(0)
    
cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


rf_config = {
    "wine": {
        "n_estimators": 221, "max_depth": 5, "min_samples_split": 8,
        "min_samples_leaf": 3, "max_features": "sqrt",
    },
    "credit": {},
    "higgs":  {},
}

X, y = load_data(DATASET)
nrows = config[DATASET].get("nrows")

model = RandomForestClassifier(**rf_config[DATASET], random_state=RANDOM_STATE, n_jobs=-1)

tracker = EmissionsTracker(output_dir=str(BASE_DIR / "emissions"), project_name=f"rf_{DATASET}")
tracker.start()

start = time.time()
cv_results = cross_validate(
    model,
    X, y, cv=cv,
    scoring={
        'accuracy': 'accuracy',
        'f1': make_scorer(f1_score, average='weighted')
    },
    return_estimator=True
)
training_time = time.time() - start
emissions = tracker.stop()

trained_model = cv_results['estimator'][0]
single_row = X[:1]

start_inference = time.perf_counter()
_ = trained_model.predict(single_row)
inference_time = time.perf_counter() - start_inference

save_results("RandomForest", DATASET, cv_results['test_accuracy'].mean(), cv_results['test_f1'].mean(), emissions, training_time, nrows)
save_inference_time("RandomForest", DATASET, emissions, nrows, inference_time)