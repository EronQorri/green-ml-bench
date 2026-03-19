import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_data, save_results

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score, make_scorer
from codecarbon import EmissionsTracker
import time

DATASET = 'credit'
X, y = load_data(DATASET)

tracker = EmissionsTracker(output_dir="emissions", project_name=f"rf_{DATASET}")
tracker.start()

rfc = RandomForestClassifier()

start = time.time()
cv_results = cross_validate(rfc, X, y, cv=5, scoring={
    'accuracy': 'accuracy',
    'f1': make_scorer(f1_score, average='weighted')
})
training_time = time.time() - start

emissions = tracker.stop()

save_results("RandomForest", DATASET, cv_results['test_accuracy'].mean(), cv_results['test_f1'].mean(), emissions, training_time)