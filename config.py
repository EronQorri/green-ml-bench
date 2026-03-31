from pathlib import Path

BASE_DIR = Path(__file__).parent  # = bachelor/


config = {
    "wine": {
        "path": BASE_DIR / "csv_files" / "wine" / "wine.data",
        "target": "Class",
        "names": ["Class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
                  "Total phenols", "Flavanoids", "Nonfavanoid phenols", "Proanthocyanins",
                  "Color Intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"],
        "skiprows": None,
        "delimiter": ",",
        "label_offset": -1,
        "nrows": None
    },
    "credit": {
        "path": BASE_DIR / "csv_files" / "default_of_credit_card_clients" / "default_of_credit_card_clients.csv",        ""
        "target": "default payment next month",
        "names": None,
        "skiprows": 1,
        "delimiter": ";",
        "drop_cols": ["ID"],
        "nrows": None
    },
    "higgs": {
        "path": BASE_DIR / "csv_files" / "higgs" / "higgs.parquet",
        "target": "label",
        "names": None,
        "skiprows": None,
        "delimiter": ",",
        "nrows": None
    }
}

RANDOM_STATE = 42
CV_FOLDS = 5