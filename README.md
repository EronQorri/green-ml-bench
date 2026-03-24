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

Before starting the scripts, you need to define the desired dataset at the top of the respective Python files:

```python
DATASET = "wine"  # Options: "wine", "credit", "higgs"
```

You can then run the models individually via the terminal:

```bash
python random_forest.py
python xgboost_model.py
python mlp.py
```
