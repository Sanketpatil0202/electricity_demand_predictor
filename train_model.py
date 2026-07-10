"""
train_model.py
---------------
Trains a RandomForestRegressor pipeline to predict ElectricityDemand
from Temperature, Humidity, WindSpeed and HourOfDay.

This mirrors the original Jupyter notebook (ElectricityDemand_RandomForest_regressor.ipynb)
but is packaged as a reusable script so the Streamlit app can load the
trained model directly.

Usage:
    python train_model.py                 # uses electricity.csv in this folder
    python train_model.py --data path.csv # use a custom dataset
"""

import argparse
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

FEATURES = ["Temperature", "Humidity", "WindSpeed", "HourOfDay"]
TARGET = "ElectricityDemand"
MODEL_PATH = os.path.join("model", "electricity_demand_model.pkl")


def train(data_path: str = "electricity.csv", model_path: str = MODEL_PATH):
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"'{data_path}' not found. Run `python generate_sample_data.py` "
            "first, or point --data to your own CSV with columns: "
            f"{FEATURES + [TARGET]}"
        )

    df = pd.read_csv(data_path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    x = df[FEATURES]
    y = df[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline(steps=[("model", RandomForestRegressor(random_state=42))])
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)

    metrics = {
        "MSE": mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
        "RMSE": root_mean_squared_error(y_test, y_pred),
    }

    print("Model performance on held-out test set:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({"pipeline": pipeline, "features": FEATURES, "metrics": metrics}, model_path)
    print(f"\nSaved trained model to {model_path}")

    return pipeline, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the electricity demand model")
    parser.add_argument(
        "--data", default="electricity.csv", help="Path to training CSV"
    )
    parser.add_argument(
        "--out", default=MODEL_PATH, help="Path to save the trained model"
    )
    args = parser.parse_args()
    train(args.data, args.out)
