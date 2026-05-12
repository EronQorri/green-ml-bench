import sys
import json
from pathlib import Path
import torch.nn as nn
from torchinfo import summary

BASE_DIR = Path(__file__).resolve().parent.parent


class MLPModule(nn.Module):
    def __init__(self, input_dim, num_classes, layer_sizes=[256, 128], dropout_rate=0.2):
        super().__init__()
        layers = []
        current_in_dim = input_dim
        for out_dim in layer_sizes:
            layers.append(nn.Linear(current_in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_in_dim = out_dim
        layers.append(nn.Linear(current_in_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, X):
        return self.network(X)


with open(BASE_DIR / "models" / "best_params.json") as f:
    best_params = json.load(f)

input_dims   = {"wine": 13, "credit": 23, "higgs": 28}
num_classes  = {"wine": 3,  "credit": 2,  "higgs": 2}

for dataset in ["wine", "credit", "higgs"]:
    _p = best_params["mlp"][dataset]["best_params"]
    layer_sizes = [_p[f"layer_{i}"] for i in range(_p["n_layers"])]

    model = MLPModule(
        input_dim=input_dims[dataset],
        num_classes=num_classes[dataset],
        layer_sizes=layer_sizes,
        dropout_rate=_p["dropout_rate"],
    )

    print(f"\n{'='*60}")
    print(f"  {dataset.upper()}  |  layers: {layer_sizes}  |  dropout: {_p['dropout_rate']:.4f}")
    print(f"{'='*60}")
    summary(model, input_size=(1, input_dims[dataset]), col_names=["input_size", "output_size", "num_params"])
