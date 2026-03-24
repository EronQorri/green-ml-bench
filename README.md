# Ökologische Effizienz von Klassifikationsalgorithmen

## Setup

Installiere zunächst die benötigten Abhängigkeiten:

```bash
pip install -r requirements.txt
```

## Datensätze

Die Datensätze müssen manuell heruntergeladen und in die folgenden Ordnerstrukturen abgelegt werden:

* **Wine:** `wine/wine.data` → [Zum UCI Download](https://archive.ics.uci.edu/dataset/109/wine)
* **Credit Card Clients:** `default_of_credit_card_clients/` → [Zum UCI Download](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
* **HIGGS:** `higgs/higgs.parquet` → [Zum UCI Download](https://archive.ics.uci.edu/dataset/280/higgs)

## Ausführung

Bevor du die Skripte startest, musst du den gewünschten Datensatz in den jeweiligen Python-Dateien oben definieren:

```python
DATASET = "wine"  # Optionen: "wine", "credit", "higgs"
```

Anschließend kannst du die Modelle einzeln über das Terminal ausführen:

```bash
python random_forest.py
python xgboost_model.py
python mlp.py
```
