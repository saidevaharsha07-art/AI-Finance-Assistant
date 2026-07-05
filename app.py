"""Flask web app for the AI-Powered Finance Assistant V2."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for

from train_ml_model import BUDGET_MODEL_PATH, CLASSIFIER_PATH, HEALTH_MODEL_PATH
from utils.advice_engine import generate_advice
from utils.budget_engine import constrained_budget
from utils.explain_engine import explain_score
from utils.finance_features import FEATURES, engineer_record, health_category, money, parse_form
from utils.forecast_engine import forecast, load_forecast_model
from utils.history_manager import get_record, init_db, recent_records, save_record
from utils.investment_engine import investment_plan
from utils.pdf_report import build_pdf

APP_DIR = Path(__file__).resolve().parent

app = Flask(__name__)


def ensure_models() -> None:
    required = [BUDGET_MODEL_PATH, CLASSIFIER_PATH, HEALTH_MODEL_PATH]
    if not all(path.exists() for path in required):
        subprocess.run([sys.executable, "train_ml_model.py"], check=True)


def load_assets() -> dict:
    ensure_models()
    return {
        "budget": joblib.load(BUDGET_MODEL_PATH),
        "classifier": joblib.load(CLASSIFIER_PATH),
        "health": joblib.load(HEALTH_MODEL_PATH),
        "forecast": load_forecast_model(),
    }


assets = load_assets()
init_db()


def classifier_confidence(frame: pd.DataFrame) -> float | None:
    model = assets["classifier"]["model"]
    if not hasattr(model, "predict_proba"):
        return None
    probabilities = model.predict_proba(frame)[0]
    return round(float(np.max(probabilities)) * 100, 1)


def predict_finance(row: dict) -> dict:
    engineered = engineer_record(row)
    frame = pd.DataFrame([engineered], columns=FEATURES)

    raw_budget_pct = assets["budget"]["model"].predict(frame)[0]
    recommended_percentages, recommended_amounts = constrained_budget(raw_budget_pct, engineered)

    spender_class = str(assets["classifier"]["model"].predict(frame)[0])
    confidence = classifier_confidence(frame)
    health_score = round(float(np.clip(assets["health"]["model"].predict(frame)[0], 5, 100)), 2)
    category = health_category(health_score)

    actual_expenses = money(engineered["total_expenses"])
    actual_savings = money(engineered["savings"])
    overspending = actual_expenses > float(engineered["monthly_income"])
    forecast_result = forecast(engineered, health_score, assets["forecast"])

    result = {
        "health_score": health_score,
        "health_category": category,
        "spender_class": spender_class,
        "prediction_confidence": confidence,
        "overspending": overspending,
        "actual_expenses": actual_expenses,
        "actual_savings": actual_savings,
        "recommended_percentages": recommended_percentages,
        "recommended_amounts": recommended_amounts,
        "forecast": forecast_result,
        "model_notes": {
            "health_model": "Random Forest regression model",
            "class_model": "Random Forest classification model",
            "forecast_model": forecast_result["engine"],
        },
        "charts": {
            "income_expenses": {
                "labels": ["Income", "Expenses", "Savings"],
                "values": [money(engineered["monthly_income"]), actual_expenses, max(actual_savings, 0)],
            },
            "budget_allocation": {
                "labels": list(recommended_amounts.keys()),
                "values": list(recommended_amounts.values()),
            },
            "risk": {
                "labels": ["Health Score", "Risk Gap"],
                "values": [health_score, round(100 - health_score, 2)],
            },
            "savings_forecast": {
                "labels": forecast_result["labels"],
                "values": forecast_result["cumulative_savings"],
            },
            "expense_forecast": {
                "labels": forecast_result["labels"],
                "values": forecast_result["expense_trend"],
            },
        },
    }
    result["explanations"] = explain_score(engineered, health_score, assets["health"].get("feature_importance"))
    result["advice"] = generate_advice(engineered, result)
    result["investment_plan"] = investment_plan(engineered, result)
    return result


def default_values() -> dict:
    return {
        "annual_income": 1200000,
        "monthly_income": 100000,
        "emi_loans": 18000,
        "rent_house": 24000,
        "food": 14000,
        "shopping": 9000,
        "education": 7000,
        "medical": 4000,
        "transport": 6000,
        "tax_pf_deductions": 12000,
        "other_expenses": 6000,
        "savings_goal": 25000,
        "dependents": 2,
        "city_tier": "Tier 1",
        "housing_type": "Rented",
        "investment_preference": "Balanced",
        "risk_level": "Medium",
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", defaults=default_values(), records=recent_records())


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return redirect(url_for("index"))
    row = parse_form(request.form)
    result = predict_finance(row)
    record_id = save_record(row, result)
    result["record_id"] = record_id
    records = recent_records()
    comparison = records[0].get("comparison") if records else None
    return render_template("index.html", defaults=row, result=result, records=records, comparison=comparison)


@app.route("/simulate", methods=["POST"])
def simulate():
    payload = request.get_json(silent=True) or {}
    row = parse_form({key: str(value) for key, value in payload.items()})
    result = predict_finance(row)
    return jsonify(result)


@app.route("/history", methods=["GET"])
def history():
    return render_template("index.html", defaults=default_values(), records=recent_records(limit=20))


@app.route("/api/history", methods=["GET"])
def api_history():
    return jsonify(recent_records(limit=20))


@app.route("/download_report/<int:record_id>", methods=["GET"])
def download_report(record_id: int):
    record = get_record(record_id)
    if record is None:
        return redirect(url_for("index"))
    pdf = build_pdf(record["input"], record["result"])
    return send_file(
        pdf,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"finance_report_{record_id}.pdf",
    )

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)