# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bachelor thesis benchmarking the **ecological efficiency** of ML classification algorithms (Logistic Regression, Random Forest, XGBoost CPU/GPU, MLP) across three datasets (Wine, Credit Card Default, HIGGS). Metrics tracked: accuracy, weighted F1, CO2 emissions (via CodeCarbon), training time, and inference latency.

## Setup

```bash
pip install -r requirements.txt
```

Datasets must be downloaded manually and placed at:

- `csv_files/wine/wine.data`
- `csv_files/default_of_credit_card_clients/` (CSV)
- `csv_files/higgs/higgs.parquet` (11M rows)

The `.env` file contains an ElectricityMaps API key used by the analysis scripts.

## Running Scripts

### Train all models on all datasets

```bash
python models/run_models.py
```

Outputs results to `results/results.csv` and `results/inference_time.csv`. Tracks emissions to `emissions/`. Note: includes a shutdown command and ntfy.sh notifications (Windows-specific behavior).

### Train a single model on one dataset

```bash
python models/log_regression.py [wine|credit|higgs]
python models/random_forest.py [wine|credit|higgs]
python models/xgboost_cpu.py [wine|credit|higgs]
python models/xgboost_gpu.py [wine|credit|higgs]
python models/mlp.py [wine|credit|higgs]
```

Default dataset is `wine` if no argument is given.

### Hyperparameter tuning

```bash
python models/tune/tune_xgb.py [dataset]          # 40 Optuna trials
python models/tune/tune_rfc.py [dataset]
python models/tune/tune_mlp.py [dataset] --n-trials 20 --max-epochs 20 --patience 5 --tune-sample-size 1000000
```

MLP tuning saves best params to `models/best_params_mlp_{dataset}.json`. The `--tune-sample-size` flag draws a stratified subset of HIGGS to speed up tuning.

### Analysis scripts (require API key)

```bash
python analysis/electricity_mix_germany.py
python analysis/carbon_intensity_analysis.py
```

## Architecture

### Data flow

1. **`config.py`** — central registry of dataset paths, delimiters, target columns, and `nrows` limits
2. **`models/utils.py`** — `load_data()` reads CSV/Parquet using config; `save_results()` appends to `results/results.csv`
3. **Each model script** — loads data, wraps training in a CodeCarbon `EmissionsTracker`, runs 5-fold cross-validation, logs metrics
4. **`models/run_models.py`** — imports and calls each model script sequentially

### Key constants (defined in each model script)

- `RANDOM_STATE = 42` — used for KFold, train_test_split, and model initialization
- `CV_FOLDS = 5` — standard across all models
- Hyperparameters are stored in per-dataset dicts (e.g., `rf_config["higgs"]`) in each script, populated from tuning results

### MLP specifics

- Implemented in PyTorch, wrapped with **Skorch** (`NeuralNetClassifier`) for sklearn compatibility
- Uses `make_pipeline(StandardScaler(), net)` pattern
- Auto-detects CUDA; falls back to CPU
- Early stopping via Skorch's `EarlyStopping` callback

### XGBoost GPU

- Uses `tree_method='gpu_hist'` — requires CUDA. Script will fail without GPU.

### Emissions tracking

- **CodeCarbon** wraps each training run; outputs to `emissions/` with project-specific names
- `models/power_monitor.py` provides Windows HardwareMonitor integration for direct CPU power readings (platform-specific)

### Results schema (`results/results.csv`)

`timestamp, dataset, model, accuracy, f1_weighted, emissions_kg, training_time_s`

Results are **appended**, never overwritten — check for duplicate rows when re-running experiments.

## Planned Experiments (post final run)

All experiments below are to be tackled after `run_all.py` completes.

### 1. XGBoost CPU vs GPU comparison
Direct comparison of training emissions, training time, and accuracy between `xgboost_cpu.py` and `xgboost_gpu.py`. Goal: quantify when GPU acceleration is ecologically justified.

### 2. HIGGS subset scaling (all models)
Script: `run_scaling_all_models.py` (already exists).
Train all models on progressively larger HIGGS subsets to find the point where additional data yields diminishing returns in accuracy relative to CO2 cost. Helps identify an ecologically optimal dataset size.

### 3. XGBoost CPU/GPU breakeven point
Script: `run_xgb_breakeven.py` (already exists).
Find the dataset size at which XGBoost GPU becomes more CO2-efficient than CPU. Requires literature research — check whether similar CPU/GPU breakeven analyses exist in Green AI literature (likely in Schwartz et al. 2020, Patterson et al. 2021, or similar).

### 4. Carbon intensity analysis (seasonal)
Scripts: `analysis/electricity_mix_germany.py`, `analysis/carbon_intensity_analysis.py`.
Compare how training emissions would differ if the same runs were performed in December (high coal share in German grid) vs July (high renewables share). Uses ElectricityMaps API. `.env` must contain a valid API key.

### 5. Deployment-scale lifecycle analysis (own idea)
Training CO2 is a one-time cost; inference CO2 scales with usage. Combine training emissions with per-prediction inference latency to find at what prediction volume the total lifecycle CO2 ranking between models shifts. Complements the breakeven analysis with a deployment perspective.
