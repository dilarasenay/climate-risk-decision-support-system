import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

from db import load_data

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "src" / "trained_model.joblib"

FEATURES = [
    "avg_temperature_c",
    "co2_emissions_tons_capita",
    "sea_level_rise_mm",
    "extreme_weather_events",
]

TARGET = "climate_risk_score"


def train_model():
    df = load_data().copy()

    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=FEATURES + [TARGET])

    X = df[FEATURES]
    y = df[TARGET]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )

    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)


def load_model():
    if not MODEL_PATH.exists():
        train_model()

    return joblib.load(MODEL_PATH)


def predict_risk(temp, co2, sea, events):
    model = load_model()

    X = pd.DataFrame([{
        "avg_temperature_c": temp,
        "co2_emissions_tons_capita": co2,
        "sea_level_rise_mm": sea,
        "extreme_weather_events": events,
    }])

    pred = float(model.predict(X)[0])
    return round(max(0, min(100, pred)), 2)