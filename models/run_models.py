import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_data, save_results, minimal_preprocess

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
import numpy as np

import torch
from torch import nn
from skorch import NeuralNetClassifier

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

class MLPModule(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256):
        super(MLPModule, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, X, **kwargs):
        return self.network(X)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(RANDOM_STATE)

for dataset in DATASETS:
    X, y = load_data(dataset)
    X, y = minimal_preprocess(X, y)
    nrows = config[dataset].get("nrows")
    
    X_array = X.to_numpy().astype(np.float32)
    y_array = y.to_numpy().astype(np.int64)

    input_dim = X_array.shape[1]
    num_classes = len(np.unique(y_array))

    lr_pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver='sag', random_state=RANDOM_STATE, n_jobs=-1, max_iter=1000)
    )
    
    mlp_pipeline = make_pipeline(
        StandardScaler(),
        NeuralNetClassifier(
            module=MLPModule,
            module__input_dim=input_dim,
            module__num_classes=num_classes,
            module__hidden_dim=256,
            max_epochs=20,
            lr=0.01,
            criterion=nn.CrossEntropyLoss,
            optimizer=torch.optim.Adam,
            iterator_train__shuffle=True,
            device=device,
            verbose=0
        )
    )

    # --- 4. Add MLP to the Models List ---
    models = [
        ("LogisticRegression", lr_pipeline),
        ("RandomForest", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        ("XGBoost",      XGBClassifier(**xgb_config[dataset], random_state=RANDOM_STATE)),
        ("XGBoost_GPU",  XGBClassifier(**xgb_config[dataset], device='cuda', random_state=RANDOM_STATE)),
        ("MLP_PyTorch",  mlp_pipeline)
    ]

    for model_name, clf in models:
        print(f"Running {model_name} on {dataset}...")
        
        tracker_name = f"{model_name.replace(' ', '_')}_{dataset}"
        tracker = EmissionsTracker(output_dir="emissions", project_name=tracker_name)
        
        tracker.start()
        start = time.time()
        
        # Pass the numpy arrays (X_array, y_array) instead of pandas DataFrames
        cv_results = cross_validate(clf, X_array, y_array, cv=CV_FOLDS, scoring=scoring)
        
        training_time = time.time() - start
        emissions = tracker.stop()
        
        save_results(model_name, dataset, cv_results['test_accuracy'].mean(), cv_results['test_f1'].mean(), emissions, training_time, nrows)
        print(f"Done: acc={cv_results['test_accuracy'].mean():.4f}, time={training_time:.2f}s\n")