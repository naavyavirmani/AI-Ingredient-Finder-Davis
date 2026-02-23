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
import os
import pandas as pd
import joblib

from merge_tables import merge_tables
from model_training import engineer_features, assign_unique_positions


def load_data():
    """
    Load and merge raw F1 data (same approach as model_training.py).
    """
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

    df = merge_tables(dataframes)
    return df


def prepare_inference_features(df):
    """
    Must match the feature columns used in model_training.py -> prepare_features().
    """
    feature_cols = [
        "grid",
        "quali_position",
        "prev_finish",
        "rolling_finish_5",
        "constructor_avg_finish",
        "constructor_avg_points",
        "driver_races",
        "driver_avg_points",
        "grid_quali_diff",
    ]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns for prediction: {missing}")

    return df[feature_cols]


def predict_for_latest_year_as_proxy(df):
    """
    Kaggle dataset usually doesn't include real 2025 results.
    So we use the latest year in the dataset as the input (proxy),
    then output predictions that represent "next season" style inference.
    """
    if "year" not in df.columns:
        raise KeyError("Expected 'year' column after merging, but it was not found.")

    latest_year = int(df["year"].max())
    df_pred = df[df["year"] == latest_year].copy()
    return df_pred, latest_year


def main():
    model_path = "f1_trained_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Run: python3 model_training.py first to generate it."
        )

    model = joblib.load(model_path)
    print(f"Loaded model: {model_path}")

    # Load + merge
    df = load_data()

    # Engineer features (same as training)
    df = engineer_features(df)

    # Use latest year as proxy input
    df_pred, latest_year = predict_for_latest_year_as_proxy(df)
    print(f"Using {latest_year} as proxy input year.")

    # Build X and predict
    X = prepare_inference_features(df_pred)
    raw_preds = model.predict(X)

    # Store raw predictions
    df_pred["pred_finish_raw"] = raw_preds

    # Assign unique finishing positions per race (1–20)
    # Important: apply uniqueness PER raceId, not across the whole dataset.
    if "raceId" not in df_pred.columns:
        raise KeyError("Expected 'raceId' column for per-race prediction grouping, but it was not found.")

    predicted_positions = []
    for race_id, group in df_pred.groupby("raceId"):
        preds = group["pred_finish_raw"].values
        unique_positions = assign_unique_positions(preds)
        predicted_positions.extend(unique_positions)

    df_pred["predicted_finish_position"] = predicted_positions

    # Output
    # Some merged tables may not contain a 'name' column; keep output safe.
    desired_cols = [
        "year",
        "raceId",
        "driverId",
        "constructorId",
        "grid",
        "quali_position",
        "pred_finish_raw",
        "predicted_finish_position",
    ]
    safe_cols = [c for c in desired_cols if c in df_pred.columns]
    out = df_pred[safe_cols].copy()

    out_file = f"predictions_{latest_year}_proxy_for_2025.csv"
    out.to_csv(out_file, index=False)

    print(f"Saved predictions to: {out_file}")
    print("Preview (first 20 rows):")
    print(out.head(20).to_string(index=False))


if _name_ == "_main_":
    main()
