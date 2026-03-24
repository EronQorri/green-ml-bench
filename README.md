# Ökologische Effizienz von Klassifikationsalgorithmen

## Setup
pip install -r requirements.txt

## Datensätze
Die Datensätze müssen manuell heruntergeladen und in folgende Ordner gelegt werden:

- `wine/wine.data` → https://archive.ics.uci.edu/dataset/109/wine
- `default_of_credit_card_clients/` → https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- `higgs/higgs.parquet` → https://archive.ics.uci.edu/dataset/280/higgs

## Ausführung
Datensatz in der jeweiligen Datei oben ändern:
DATASET = "wine"  ("wine", "credit", "higgs")

python random_forest.py
python xgboost_model.py
python mlp.py
```

Und ein `requirements.txt`:
```
pandas
scikit-learn
xgboost
torch
codecarbon
pyarrow
