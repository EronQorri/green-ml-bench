import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_data, save_results

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import f1_score, make_scorer
from codecarbon import EmissionsTracker
import time

DATASET = 'credit'

xgb_config = {
    "wine":   {"objective": "multi:softmax", "num_class": 3, "eval_metric": "mlogloss"},
    "credit": {"objective": "binary:logistic", "eval_metric": "logloss"},
    "higgs":  {"objective": "binary:logistic", "eval_metric": "logloss"},
}

X, y = load_data(DATASET)

tracker = EmissionsTracker(output_dir="emissions", project_name=f"xgb_{DATASET}")
tracker.start()

xgbc = XGBClassifier(**xgb_config[DATASET], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
xgbc.fit(X_train, y_train)

start = time.time()
cv_results = cross_validate(xgbc, X, y, cv=5, scoring={
    'accuracy': 'accuracy',
    'f1': make_scorer(f1_score, average='weighted')
})
training_time = time.time() - start


emissions = tracker.stop()

save_results("XGBoost", DATASET, cv_results['test_accuracy'].mean(), cv_results['test_f1'].mean(), emissions, training_time)