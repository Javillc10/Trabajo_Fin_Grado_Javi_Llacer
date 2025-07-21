import pandas as pd
import json
import sys
from src.utils.feature_engineering import FeatureEngineer

def preview_features(csv_path):
    """
    Loads a CSV file, runs FeatureEngineer, and returns the resulting columns as JSON.
    """
    try:
        df = pd.read_csv(csv_path)
        feature_engineer = FeatureEngineer()
        df_featured = feature_engineer.fit_transform(df)
        result_columns = df_featured.columns.tolist()
        return json.dumps(result_columns)
    except Exception as e:
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        result = preview_features(csv_path)
        print(result)
    else:
        print(json.dumps({"error": "CSV path not provided as argument."}))
