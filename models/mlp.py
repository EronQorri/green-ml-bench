import os
import sys
import time
import numpy as np
from skorch.callbacks import EarlyStopping
import torch
import torch.nn as nn
from torch import nn
from skorch import NeuralNetClassifier
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from codecarbon import EmissionsTracker

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_data, save_inference_time, save_results
from config import BASE_DIR, config, RANDOM_STATE, CV_FOLDS

NEURONS = 256
EPOCHS = 200
PATIENCE = 5
DROPOUT_RATE = 0.2
LR = 0.001
BATCH_SIZE = 4096
NUM_LAYERS = 4


DATASET = sys.argv[1] if len(sys.argv) > 1 else 'wine'

cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


class MLPModule(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=NEURONS, dropout_rate=DROPOUT_RATE, num_layers=NUM_LAYERS):
        super(MLPModule, self).__init__()
        
        layers = []
        current_in_dim = input_dim
        current_out_dim = hidden_dim
        
        # Dynamisch die versteckten Schichten aufbauen
        for _ in range(num_layers):
            layers.append(nn.Linear(current_in_dim, current_out_dim))
            layers.append(nn.BatchNorm1d(current_out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            
            # Dimensionen für die nächste Schicht vorbereiten (halbieren)
            current_in_dim = current_out_dim
            current_out_dim = current_out_dim // 2
            
            # Sicherheitscheck, damit die Dimension nicht auf 0 fällt
            if current_out_dim < 1:
                current_out_dim = 1
                
        # Letzte Schicht (Output) hinzufügen
        layers.append(nn.Linear(current_in_dim, num_classes))
        
        # Sternchen-Operator (*) entpackt die Liste in nn.Sequential
        self.network = nn.Sequential(*layers)

    def forward(self, X):
        return self.network(X)

mlp_config = {
    "wine":   {"num_classes": 3},
    "credit": {"num_classes": 2}, 
    "higgs":  {"num_classes": 2},
}

X, y = load_data(DATASET)

X_array = X.to_numpy().astype(np.float32)
y_array = y.to_numpy().astype(np.int64)

nrows = config[DATASET].get("nrows")
input_dim = X_array.shape[1]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(RANDOM_STATE)

net = NeuralNetClassifier(
    module=MLPModule, 
    module__input_dim=input_dim, 
    module__hidden_dim=NEURONS, 
    module__num_classes=mlp_config[DATASET]["num_classes"],
    module__dropout_rate=DROPOUT_RATE,
    module__num_layers=NUM_LAYERS,
    max_epochs=EPOCHS,
    lr=LR,               
    iterator_train__batch_size=BATCH_SIZE, 
    iterator_valid__batch_size=BATCH_SIZE, 
    criterion=nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    iterator_train__shuffle=True,
    device=device,
    verbose=0,
    callbacks=[EarlyStopping(patience=PATIENCE)]
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
    },
    return_estimator=True
)
training_time = time.time() - start
emissions = tracker.stop()


trained_model = cv_results['estimator'][0]
single_row = X_array[:1]

start_inference = time.perf_counter()
_ = trained_model.predict(single_row)
inference_time = time.perf_counter() - start_inference

save_results("MLP_PyTorch", DATASET, cv_results['test_accuracy'].mean(), cv_results['test_f1'].mean(), emissions, training_time, nrows)
save_inference_time("MLP_PyTorch", DATASET, emissions, nrows, inference_time)