"""
predict_2025.py

This file is responsible for generating predictions for the 2025 Formula 1 season.

It:
- Loads the trained model from model_training.py
- Creates or loads input data representing the 2025 season
- Applies the same feature logic used during training
- Generates predictions for each driver in each race
- Outputs the predicted results in a structured format (e.g., CSV)

This file should NOT retrain the model.
It should only run inference using the trained model.
"""
import joblib
import pandas as pd
import os

from merge_tables import merge_tables
from model_training import engineer_features


def load_data():
    script_dir = os.path.dirname(os.path.abspath(_file_))
    data_dir = os.path.join(script_dir, "F1Data")

    dataframes = {
        "results": pd.read_csv(os.path.join(data_dir, "results.csv")),
        "races": pd.read_csv(os.path.join(data_dir, "races.csv")),
        "drivers": pd.read_csv(os.path.join(data_dir, "drivers.csv")),
        "constructors": pd.read_csv(os.path.join(data_dir, "constructors.csv")),
        "qualifying": pd.read_csv(os.path.join(data_dir, "qualifying.csv")),
        "status": pd.read_csv(os.path.join(data_dir, "status.csv")),
    }

    return merge_tables(dataframes)


def main():
    print("Loading trained model...")
    model = joblib.load("f1_trained_model.pkl")

    print("Loading data...")
    df = load_data()

    print("Engineering features...")
    df = engineer_features(df)

    latest_year = int(df["year"].max())
    df_latest = df[df["year"] == latest_year].copy()

    feature_cols = [
        "grid",
        "quali_position",
        "prev_finish",
        "rolling_finish_5",
        "constructor_avg_finish",
        "constructor_avg_points",
        "driver_races",
        "driver_avg_points",
        "grid_quali_diff"
    ]

    X = df_latest[feature_cols]

    print("Generating predictions...")
    predictions = model.predict(X)

    df_latest["predicted_finish"] = predictions

    df_latest.to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv")


if _name_ == "_main_":
    main()
