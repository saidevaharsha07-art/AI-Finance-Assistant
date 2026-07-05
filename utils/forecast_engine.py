"""Forecasting helpers for savings, expense, and risk trends."""

from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from utils.finance_features import FEATURES, engineer_record, health_category, money

FORECAST_MODEL_PATH = Path("models/savings_forecaster.joblib")


def load_forecast_model() -> dict | None:
    if FORECAST_MODEL_PATH.exists():
        bundle = joblib.load(FORECAST_MODEL_PATH)
        if bundle.get("is_keras"):
            try:
                import tensorflow as tf

                bundle["model"] = tf.keras.models.load_model(bundle["keras_model_path"])
            except Exception:
                return None
        return bundle
    return None


def forecast(row: dict, health_score: float, model_bundle: dict | None = None) -> dict:
    engineered = engineer_record(row)
    current_savings = float(engineered["savings"])
    current_expenses = float(engineered["total_expenses"])
    income = max(float(engineered["monthly_income"]), 1)
    labels = [f"Month {month}" for month in range(1, 13)]

    if model_bundle:
        frame = pd.DataFrame([engineered], columns=FEATURES)
        prepared = model_bundle["preprocessor"].transform(frame)
        if hasattr(prepared, "toarray"):
            prepared = prepared.toarray()
        raw = model_bundle["model"].predict(prepared)[0]
        monthly_savings = [money(max(value, -income)) for value in raw[:12]]
        engine = model_bundle.get("engine", "Neural forecast model")
    else:
        trend = 1 + min(max(float(engineered["savings_ratio"]), -0.25), 0.35) * 0.08
        pressure = 1 + min(float(engineered["expense_to_income_ratio"]), 1.35) * 0.006
        monthly_savings = []
        for month in range(1, 13):
            value = current_savings * (trend ** month) - max(current_expenses - income, 0) * math.log1p(month) * 0.08
            monthly_savings.append(money(value))
        engine = "Forecast fallback until Keras model is trained"

    cumulative = np.cumsum(monthly_savings).tolist()
    expense_trend = [money(current_expenses * (1 + 0.006 * month)) for month in range(1, 13)]
    risk_trend = [health_category(max(5, min(100, health_score + (cumulative[i] / max(income, 1)) * 1.5))) for i in range(12)]

    return {
        "engine": engine,
        "labels": labels,
        "monthly_savings": monthly_savings,
        "cumulative_savings": [money(value) for value in cumulative],
        "expense_trend": expense_trend,
        "risk_trend": risk_trend,
        "month_1": monthly_savings[0],
        "month_6": money(cumulative[5]),
        "month_12": money(cumulative[11]),
    }
