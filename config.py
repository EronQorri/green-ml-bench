config = {
    "wine": {
        "path": "wine/wine.data",
        "target": "Class",
        "names": ["Class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
                  "Total phenols", "Flavanoids", "Nonfavanoid phenols", "Proanthocyanins",
                  "Color Intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"],
        "skiprows": None,
        "delimiter": ","
    },
    "credit": {
        "path": "default_of_credit_card_clients/default_of_credit_card_clients.csv",
        "target": "default payment next month",
        "names": None,
        "skiprows": 1,
        "delimiter": ";"
    },
    "higgs": {
        "path": "higgs/higgs.parquet",
        "target": "label",
        "names": None,
        "skiprows": None,
        "delimiter": ","
    }
}