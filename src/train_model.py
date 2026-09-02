"""
Feature engineering, model training/comparison, and 7-day forecasting.

Pulled out of notebooks/02_ML_Training.ipynb so the notebook and the
Streamlit app (app/app.py) share one implementation instead of two
copies that can silently drift apart.
"""

from __future__ import annotations

import datetime

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

TARGET = "Daily_Usage_m3"
CATEGORICAL_COLS = ["Zone", "Concrete_Quality"]
NUMERIC_COLS = ["Labor_Count", "Weather_Rain", "Truck_Travel_Min", "day", "month", "weekday"]
FEATURES = CATEGORICAL_COLS + NUMERIC_COLS


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day/month/weekday features derived from the Date column."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["day"] = df["Date"].dt.day
    df["month"] = df["Date"].dt.month
    df["weekday"] = df["Date"].dt.weekday
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", "passthrough", NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    ])


def get_models(random_state: int = 42) -> dict:
    return {
        "RandomForest": RandomForestRegressor(n_estimators=120, random_state=random_state),
        "XGBoost": XGBRegressor(n_estimators=120, learning_rate=0.08, random_state=random_state),
        "LightGBM": LGBMRegressor(n_estimators=120, random_state=random_state, verbosity=-1),
    }


def train_and_compare(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Train RandomForest, XGBoost, and LightGBM and compare them.

    Returns (trained_pipelines, results) where results maps model name
    to {"r2": ..., "mae": ...}.
    """
    df = engineer_features(df)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    trained_models, results = {}, {}
    for name, regressor in get_models(random_state).items():
        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", regressor),
        ])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        results[name] = {
            "r2": r2_score(y_test, preds),
            "mae": mean_absolute_error(y_test, preds),
        }
        trained_models[name] = pipeline

    return trained_models, results


def pick_best_model(trained_models: dict, results: dict):
    """Select the model with the highest R2 score."""
    best_name = max(results, key=lambda name: results[name]["r2"])
    return best_name, trained_models[best_name]


def forecast_next_days(
    model,
    df_history: pd.DataFrame,
    zones: list[str],
    days: int = 7,
    labor_range=(10, 30),
    weather_rain_prob: float = 0.15,
    travel_range=(15, 90),
    quality_choices=("GOOD", "WARNING", "REJECTED"),
    seed: int | None = None,
) -> pd.DataFrame:
    """Forecast usage for the next `days` days, one row per zone per day.

    `df_history` must already contain a parsed 'Date' column (i.e. it can
    be the output of engineer_features / simulate_iot_data) so we know
    where the forecast should start from.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    last_date = pd.to_datetime(df_history["Date"]).max()
    next_dates = [last_date + datetime.timedelta(days=i) for i in range(1, days + 1)]

    rows = []
    for date in next_dates:
        for zone in zones:
            labor = int(rng.integers(*labor_range))
            rain = int(rng.random() < weather_rain_prob)
            travel = int(rng.integers(*travel_range))
            quality = str(rng.choice(quality_choices))
            rows.append([date, zone, labor, rain, travel, quality])

    future = pd.DataFrame(
        rows, columns=["Date", "Zone", "Labor_Count", "Weather_Rain", "Truck_Travel_Min", "Concrete_Quality"]
    )
    future = engineer_features(future)
    future["Predicted_Daily_Usage_m3"] = model.predict(future[FEATURES])
    return future
