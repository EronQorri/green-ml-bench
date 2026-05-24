"""
run_tuning_overhead_analysis.py — Tuned vs. default hyperparameter comparison.

Produces two tables:
  1. HPO overhead ratio: for every tunable model × dataset, how many times larger
     is the HPO cost relative to a single training run?
  2. Tuned vs. default on HIGGS (full 11 M rows): total cost including HPO vs.
     training with scikit-learn / XGBoost / PyTorch defaults, and the resulting
     F1 delta.

LR is excluded from the default comparison because it has no HPO.
RF-HIGGS defaults are excluded (RF excluded from HIGGS in both tuned and default runs).
The XGB HPO run was executed once on the GPU; that cost is attributed to both the
CPU and GPU variants.
"""

import pandas as pd
from pathlib import Path

BASE_DIR  = Path(__file__).parent.parent
RESULTS   = BASE_DIR / "results" / "results.csv"

# ── canonical display labels ────────────────────────────────────────────────
TUNE_KEY = {          # results.csv model name → (display label, hpo_key)
    "tune_RFC": ("Random Forest",   "rfc"),
    "tune_XGB": ("XGBoost",         "xgb"),
    "tune_MLP": ("MLP",             "mlp"),
}
TUNED_MODEL_KEY = {   # results.csv model name → (display label, hpo_key)
    "RandomForest":  ("Random Forest",   "rfc"),
    "XGBoost":       ("XGBoost (CPU)",   "xgb"),
    "XGBoost_GPU":   ("XGBoost (GPU)",   "xgb"),   # same HPO run as CPU
    "MLP_PyTorch":   ("MLP",             "mlp"),
}
DEFAULT_MODEL_KEY = { # results.csv model name → display label
    "RF_default":       "Random Forest",
    "XGB_default":      "XGBoost (CPU)",
    "XGB_GPU_default":  "XGBoost (GPU)",
    "MLP_default":      "MLP",
}


def load():
    df = pd.read_csv(RESULTS)
    df.columns = df.columns.str.strip()
    df["model"]   = df["model"].str.strip()
    df["dataset"] = df["dataset"].str.strip()
    df["nrows"]   = df["nrows"].astype(str).str.strip()
    df["co2eq_kg"] = pd.to_numeric(df["co2eq_kg"], errors="coerce")
    df["f1"]       = pd.to_numeric(df["f1"],       errors="coerce")
    return df


def hpo_overhead_table(df):
    """
    Table 1 – HPO cost as a multiple of one training run, for every
    tunable model × dataset where both HPO and training rows exist.
    """
    tune_df   = df[df["model"].isin(TUNE_KEY)].copy()
    tuned_df  = df[df["model"].isin(TUNED_MODEL_KEY) & (df["nrows"] == "all")].copy()
    tune_df   = tune_df.drop_duplicates(subset=["model", "dataset"], keep="last")
    tuned_df  = tuned_df.drop_duplicates(subset=["model", "dataset"], keep="last")

    rows = []
    for tune_model, (label, hpo_key) in TUNE_KEY.items():
        for _, tr in tune_df[tune_df["model"] == tune_model].iterrows():
            ds = tr["dataset"]
            hpo_co2 = tr["co2eq_kg"]
            # find the matching training run
            train_match = tuned_df[
                (tuned_df["dataset"] == ds) &
                (tuned_df["model"].map(lambda m: TUNED_MODEL_KEY.get(m, (None, None))[1]) == hpo_key)
            ]
            if train_match.empty:
                continue
            # pick the "canonical" variant (CPU for XGB, only one for RF/MLP)
            train_row = train_match.iloc[0]
            train_co2 = train_row["co2eq_kg"]
            rows.append({
                "Model":          label,
                "Dataset":        ds.capitalize(),
                "HPO CO2 (g)":    hpo_co2   * 1e3,
                "Train CO2 (g)":  train_co2 * 1e3,
                "HPO / Train":    hpo_co2 / train_co2,
            })

    return pd.DataFrame(rows)


def tuned_vs_default_table(df):
    """
    Table 2 – Total tuned cost (HPO + training) vs. default training on HIGGS 11 M.
    """
    # tuned training rows – nrows=all for HIGGS
    tuned_df = df[
        df["model"].isin(TUNED_MODEL_KEY) &
        (df["dataset"] == "higgs") &
        (df["nrows"] == "all")
    ].drop_duplicates(subset=["model"], keep="last")

    # HPO rows for HIGGS
    hpo_df = df[
        df["model"].isin(TUNE_KEY) &
        (df["dataset"] == "higgs") &
        (df["nrows"] == "all")
    ].drop_duplicates(subset=["model"], keep="last")
    hpo_by_key = {}
    for _, row in hpo_df.iterrows():
        _, hpo_key = TUNE_KEY[row["model"]]
        hpo_by_key[hpo_key] = row["co2eq_kg"]

    # default training rows – nrows=11000000 (full HIGGS via scaling script)
    default_df = df[
        df["model"].isin(DEFAULT_MODEL_KEY) &
        (df["dataset"] == "higgs") &
        (df["nrows"] == "11000000")
    ].drop_duplicates(subset=["model"], keep="last")

    rows = []
    for tune_model, (label, hpo_key) in TUNED_MODEL_KEY.items():
        tuned_row = tuned_df[tuned_df["model"] == tune_model]
        if tuned_row.empty:
            continue
        tuned_row = tuned_row.iloc[0]

        hpo_co2   = hpo_by_key.get(hpo_key, 0.0)
        train_co2 = tuned_row["co2eq_kg"]
        total_co2 = hpo_co2 + train_co2
        tuned_f1  = tuned_row["f1"]

        # find default counterpart
        default_label = label.replace(" (CPU)", "").replace(" (GPU)", "")
        default_model_name = {v: k for k, v in DEFAULT_MODEL_KEY.items()}.get(
            "XGBoost (CPU)" if "(CPU)" in label else
            "XGBoost (GPU)" if "(GPU)" in label else label
        )
        # rebuild lookup properly
        inv_default = {v: k for k, v in DEFAULT_MODEL_KEY.items()}
        default_model_name = inv_default.get(label)
        if default_model_name is None:
            continue
        def_row = default_df[default_df["model"] == default_model_name]
        if def_row.empty:
            continue
        def_row      = def_row.iloc[0]
        default_co2  = def_row["co2eq_kg"]
        default_f1   = def_row["f1"]

        rows.append({
            "Model":              label,
            "HPO CO2 (g)":        hpo_co2   * 1e3,
            "Train tuned (g)":    train_co2 * 1e3,
            "Total tuned (g)":    total_co2 * 1e3,
            "Default train (g)":  default_co2 * 1e3,
            "CO2 saved (g)":      (total_co2 - default_co2) * 1e3,
            "Tuned F1":           tuned_f1,
            "Default F1":         default_f1,
            "dF1":                tuned_f1 - default_f1,
        })

    return pd.DataFrame(rows)


def print_table1(t):
    print("\n" + "=" * 70)
    print("TABLE 1 -- HPO overhead relative to one training run (nrows=all)")
    print("=" * 70)
    print(f"{'Model':<20} {'Dataset':<10} {'HPO (g)':>12} {'Train (g)':>12} {'HPO/Train':>10}")
    print("-" * 70)
    for _, r in t.iterrows():
        print(f"{r['Model']:<20} {r['Dataset']:<10} "
              f"{r['HPO CO2 (g)']:>12.1f} {r['Train CO2 (g)']:>12.3f} "
              f"{r['HPO / Train']:>9.1f}x")

    print("\n--- LaTeX rows (paste into tabular) ---")
    for _, r in t.iterrows():
        hpo   = f"{r['HPO CO2 (g)']:.1f}"
        train = f"{r['Train CO2 (g)']:.3f}"
        ratio = f"{r['HPO / Train']:.1f}"
        print(f"  {r['Model']} & {r['Dataset']} & {hpo} & {train} & {ratio}\\\\")


def print_table2(t):
    print("\n" + "=" * 95)
    print("TABLE 2 -- Tuned (HPO + train) vs. default training, HIGGS full (11 M rows)")
    print("=" * 95)
    hdr = (f"{'Model':<18} {'HPO (g)':>10} {'Train-T (g)':>12} "
           f"{'Total-T (g)':>12} {'Default (g)':>13} "
           f"{'CO2 saved (g)':>14} {'F1 tuned':>10} {'F1 def':>9} {'dF1':>7}")
    print(hdr)
    print("-" * 95)
    for _, r in t.iterrows():
        print(
            f"{r['Model']:<18} "
            f"{r['HPO CO2 (g)']:>10.1f} "
            f"{r['Train tuned (g)']:>12.3f} "
            f"{r['Total tuned (g)']:>12.1f} "
            f"{r['Default train (g)']:>13.3f} "
            f"{r['CO2 saved (g)']:>14.1f} "
            f"{r['Tuned F1']:>10.4f} "
            f"{r['Default F1']:>9.4f} "
            f"{r['dF1']:>7.4f}"
        )

    print("\nNote: 'CO2 saved' = total-tuned - default. Positive = default is cheaper.")
    print("      XGB HPO ran once on GPU; cost is attributed to both CPU and GPU variants.")

    print("\n--- LaTeX rows (paste into tabular) ---")
    for _, r in t.iterrows():
        hpo   = f"{r['HPO CO2 (g)']:.1f}"
        train = f"{r['Train tuned (g)']:.3f}"
        total = f"{r['Total tuned (g)']:.1f}"
        dflt  = f"{r['Default train (g)']:.3f}"
        saved = f"{r['CO2 saved (g)']:.1f}"
        tf1   = f"{r['Tuned F1']:.4f}"
        df1   = f"{r['Default F1']:.4f}"
        df1d  = f"{r['dF1']:+.4f}"
        print(f"  {r['Model']} & {hpo} & {train} & {total} & {dflt} & {saved} & {tf1} & {df1} & {df1d}\\\\")


if __name__ == "__main__":
    df = load()
    t1 = hpo_overhead_table(df)
    t2 = tuned_vs_default_table(df)
    print_table1(t1)
    print_table2(t2)
