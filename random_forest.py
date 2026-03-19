from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score
import pandas as pd
from codecarbon import EmissionsTracker
from config import config

DATASET = "credit"  # "wine", "credit", "higgs"



cfg = config[DATASET]

if DATASET == "higgs":
    df = pd.read_parquet(cfg["path"])
else:
    df = pd.read_csv(cfg["path"], names=cfg["names"], skiprows=cfg["skiprows"], delimiter=cfg["delimiter"])

X = df.drop(cfg["target"], axis=1)
y = df[cfg["target"]]

tracker = EmissionsTracker()
tracker.start()

rfc = RandomForestClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf = rfc.fit(X_train, y_train)
y_pred = clf.predict(X_test)
scores = cross_val_score(rfc, X, y, cv=5)

emissions = tracker.stop()

print(f"Datensatz: {DATASET}")
print(f"Accuracy (CV Mean): {scores.mean():.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(f"CO₂ Emissionen: {emissions} kg")