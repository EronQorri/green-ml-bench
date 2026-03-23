import pandas as pd

df = pd.read_csv("./csv_files/higgs/HIGGS.csv.gz", compression='gzip', header=None)
df.columns = ['label'] + [f'feature_{i}' for i in range(1, 29)]
df.to_parquet("csv_files/higgs/higgs.parquet")