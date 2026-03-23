import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_data, save_results, minimal_preprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score, make_scorer
from codecarbon import EmissionsTracker
import time

DATASET = 'wine'

X, y = load_data(DATASET)
X, y = minimal_preprocess(X, y)

tracker = EmissionsTracker(output_dir="emissions", project_name=f"rf_{DATASET}")
tracker.start()

start = time.time()
cv_results = cross_validate(
    RandomForestClassifier(random_state=42),
    X, y, cv=5,
    scoring={
        'accuracy': 'accuracy',
        'f1': make_scorer(f1_score, average='weighted')
    }
)
training_time = time.time() - start
emissions = tracker.stop()

save_results("RandomForest", DATASET, cv_results['test_accuracy'].mean(), cv_results['test_f1'].mean(), emissions, training_time)