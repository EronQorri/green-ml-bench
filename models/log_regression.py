import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_data, save_results, minimal_preprocess

# Neue Imports für die Pipeline und Skalierung hinzugefügt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import f1_score, make_scorer
from codecarbon import EmissionsTracker
from config import config, RANDOM_STATE, CV_FOLDS, BASE_DIR
import time


DATASET = sys.argv[1] if len(sys.argv) > 1 else 'wine'
cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

X, y = load_data(DATASET)
X, y = minimal_preprocess(X, y)
nrows = config[DATASET].get("nrows")

tracker = EmissionsTracker(output_dir=str(BASE_DIR / "emissions"), project_name=f"lr_{DATASET}")
tracker.start()

start = time.time()
cv_results = cross_validate(
    make_pipeline(
        StandardScaler(),
        LogisticRegression(solver='sag', random_state=RANDOM_STATE, n_jobs=-1, max_iter=1000)
    ),
    X, y, cv=cv,
    scoring={
        'accuracy': 'accuracy',
        'f1': make_scorer(f1_score, average='weighted')
    }
)
training_time = time.time() - start
emissions = tracker.stop()

save_results("LogisticRegression", DATASET, cv_results['test_accuracy'].mean(), cv_results['test_f1'].mean(), emissions, training_time, nrows)