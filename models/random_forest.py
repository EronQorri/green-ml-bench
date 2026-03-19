import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score, make_scorer
from codecarbon import EmissionsTracker

DATASET = 'credit'
X, y = load_data(DATASET)

tracker = EmissionsTracker(output_dir="emissions", project_name=f"rf_{DATASET}")
tracker.start()

rfc = RandomForestClassifier()
cv_results = cross_validate(rfc, X, y, cv=5, scoring={
    'accuracy': 'accuracy',
    'f1': make_scorer(f1_score, average='weighted')
})

emissions = tracker.stop()

print(f"Datensatz: {DATASET}")
print(f"Accuracy (CV Mean): {cv_results['test_accuracy'].mean():.4f}")
print(f"F1-Score (CV Mean): {cv_results['test_f1'].mean():.4f}")
print(f"CO₂ Emissionen: {emissions:.2e} kg")