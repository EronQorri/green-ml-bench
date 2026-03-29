import subprocess
import sys
from pathlib import Path

DATASETS = ['wine', 'credit', 'higgs']

SCRIPTS = [
    'log_regression.py',
    'random_forest.py',
    'xgboost_cpu.py',
    'xgboost_gpu.py',
    'mlp.py',
]

for dataset in DATASETS:
    for script in SCRIPTS:
        print(f"\n>>> {script} | {dataset}")
        result = subprocess.run(
            [sys.executable, script, dataset],
            cwd=Path(__file__).parent
        )
        if result.returncode != 0:
            print(f"  FAILED: {script} on {dataset}")