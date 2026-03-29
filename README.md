# Ecological Efficiency of Classification Algorithms

## Setup

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Datasets

The datasets need to be downloaded manually and placed in the following folder structures:

* **Wine:** `wine/wine.data` → [UCI Download](https://archive.ics.uci.edu/dataset/109/wine)
* **Credit Card Clients:** `default_of_credit_card_clients/` → [UCI Download](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
* **HIGGS:** `higgs/higgs.parquet` → [UCI Download](https://archive.ics.uci.edu/dataset/280/higgs)

## Running the Scripts

You can run each model individually for a specific dataset by passing the dataset name as an argument:

```bash
python models/log_regression.py wine
python models/random_forest.py credit
python models/xgboost_cpu.py higgs
python models/xgboost_gpu.py wine
python models/mlp.py credit
```

Valid dataset options are: wine, credit, higgs. If no argument is provided, the scripts default to wine.

To run all models across all datasets automatically:

```bash
python models/run_models.py
```

![Results](results/vergleich.png)