import os
import sys
import time
import numpy as np
import torch
from torch import nn
from skorch import NeuralNetClassifier
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from codecarbon import EmissionsTracker

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_data, save_results, minimal_preprocess
from config import BASE_DIR, config, RANDOM_STATE, CV_FOLDS

DATASET = sys.argv[1] if len(sys.argv) > 1 else 'wine'
cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

class MLPModule(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128):
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

mlp_config = {
    "wine":   {"num_classes": 3},
    "credit": {"num_classes": 2}, 
    "higgs":  {"num_classes": 2},
}

X, y = load_data(DATASET)
X, y = minimal_preprocess(X, y)

X_array = X.to_numpy().astype(np.float32)
y_array = y.to_numpy().astype(np.int64)

nrows = config[DATASET].get("nrows")
input_dim = X_array.shape[1]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(RANDOM_STATE)

net = NeuralNetClassifier(
    MLPModule(input_dim=input_dim, num_classes=mlp_config[DATASET]["num_classes"]),
    max_epochs=20,
    lr=0.01,
    iterator_train__batch_size=8192,
    criterion=nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    iterator_train__shuffle=True,
    device=device,
    verbose=0
)

pipeline = make_pipeline(StandardScaler(), net)

tracker = EmissionsTracker(output_dir=str(BASE_DIR / "emissions"), project_name=f"mlp_{DATASET}")
tracker.start()

start = time.time()
cv_results = cross_validate(
    pipeline,
    X_array, y_array, cv=cv,
    scoring={
        'accuracy': 'accuracy',
        'f1': make_scorer(f1_score, average='weighted')
    }
)
training_time = time.time() - start
emissions = tracker.stop()

save_results("MLP_PyTorch", DATASET, cv_results['test_accuracy'].mean(), cv_results['test_f1'].mean(), emissions, training_time, nrows)