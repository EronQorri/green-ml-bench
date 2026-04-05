import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer
from config import config, RANDOM_STATE, CV_FOLDS
import optuna

DATASET = sys.argv[1] if len(sys.argv) > 1 else "wine"

X, y = load_data(DATASET)

cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", None]
        ),
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }
    model = RandomForestClassifier(**params)
    score = cross_val_score(
        model, X, y, cv=cv, scoring=make_scorer(f1_score, average="weighted")
    ).mean()
    return score


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print(f"\nBest F1: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
