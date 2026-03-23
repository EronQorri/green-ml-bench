import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_data, save_results, minimal_preprocess
from sklearn.linear_model import LogisticRegression  # Import geändert
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score, make_scorer
from codecarbon import EmissionsTracker
from config import config, RANDOM_STATE, CV_FOLDS
import time

DATASET = 'higgs'

X, y = load_data(DATASET)
X, y = minimal_preprocess(X, y)
nrows = config[DATASET].get("nrows")

# Projektname für CodeCarbon angepasst
tracker = EmissionsTracker(output_dir="emissions", project_name=f"lr_{DATASET}")
tracker.start()

start = time.time()
cv_results = cross_validate(
    # Modell ausgetauscht. max_iter=1000 verhindert Warnungen bei nicht sofort konvergierenden Daten.
    LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1, max_iter=1000),
    X, y, cv=CV_FOLDS,
    scoring={
        'accuracy': 'accuracy',
        'f1': make_scorer(f1_score, average='weighted')
    }
)
training_time = time.time() - start
emissions = tracker.stop()

# Name für die CSV-Speicherung angepasst
save_results("LogisticRegression", DATASET, cv_results['test_accuracy'].mean(), cv_results['test_f1'].mean(), emissions, training_time, nrows)