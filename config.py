"""
config.py

This file stores all configuration values and constants used across the project.

It:
- Defines file paths
- Stores model parameters
- Stores feature lists
- Defines the target variable
- Sets random seeds for reproducibility

Any value that might otherwise be hard-coded
should live in this file.
"""
DATA_FOLDER = "F1Data/"

DRIVER_STANDINGS_FILE = DATA_FOLDER + "driver_standings.csv"
RACES_FILE = DATA_FOLDER + "races.csv"

MODEL_SAVE_PATH = "f1_season_model.pkl"


TRAIN_UNTIL_YEAR = 2024  
TARGET = "points"

TEST_SIZE = 0.2
RANDOM_SEED = 42


N_ESTIMATORS = 200
MAX_DEPTH = 8
