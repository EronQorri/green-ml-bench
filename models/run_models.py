import subprocess
import sys
import time
from pathlib import Path
import requests

DATASETS = ['wine', 'credit']

SCRIPTS = [
    'log_regression.py',
    'random_forest.py',
    'xgboost_cpu.py',
    'xgboost_gpu.py',
    'mlp.py',
]

start_time = time.time()

for dataset in DATASETS:
    for script in SCRIPTS:
        print(f"\n>>> {script} | {dataset}")
        result = subprocess.run(
            [sys.executable, script, dataset],
            cwd=Path(__file__).parent
        )
        if result.returncode != 0:
            print(f"  FAILED: {script} on {dataset}")

total_time = (time.time() - start_time) / 60

try:
    requests.post(
        "https://ntfy.sh/eron_thesis_higgs_run_123", # Deinen Kanalnamen einsetzen!
        data=f"Alle Skripte sind durchgelaufen! Gesamtdauer: {total_time:.2f} Minuten.".encode(encoding='utf-8'),
        headers={
            "Title": "Thesis: Master-Run beendet!",
            "Priority": "urgent",
            "Tags": "tada,party_popper"
        }
    )
except Exception as e:
    print(f"Konnte keine Benachrichtigung senden: {e}")