# models/tune_xgb.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_data
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer
from config import config, RANDOM_STATE, CV_FOLDS
import optuna

DATASET = sys.argv[1] if len(sys.argv) > 1 else 'wine'

nrows = config[DATASET].get("nrows")
X, y = load_data(DATASET)

cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

xgb_base = {
    "wine":   {"objective": "multi:softmax", "num_class": 3, "eval_metric": "mlogloss"},
    "credit": {"objective": "binary:logistic", "eval_metric": "logloss"},
    "higgs":  {"objective": "binary:logistic", "eval_metric": "logloss"},
}

def objective(trial):
    params = {
        **xgb_base[DATASET],
        "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
        "max_depth":         trial.suggest_int("max_depth", 3, 8),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        "random_state":      RANDOM_STATE,
    }
    model = XGBClassifier(**params)
    score = cross_val_score(
        model, X, y, cv=cv,
        scoring=make_scorer(f1_score, average='weighted')
    ).mean()
    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print(f"\nBest F1: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")