"""Personalized investment plan rules."""

from __future__ import annotations

from utils.finance_features import money


def investment_plan(row: dict, result: dict) -> list[dict]:
    monthly_savings = max(float(result["actual_savings"]), 0)
    risk = str(row.get("risk_level", "Medium"))
    preference = str(row.get("investment_preference", "Balanced"))
    dependents = int(row.get("dependents", 0))
    score = float(result["health_score"])

    investable = max(monthly_savings * 0.75, float(result["recommended_amounts"].get("investments", 0)))
    if score < 50:
        investable = max(monthly_savings * 0.35, 0)

    if score < 60 or dependents >= 2:
        weights = {"Emergency Fund": 35, "Fixed Deposit": 20, "PPF": 15, "Mutual Funds": 15, "Index Funds": 5, "Gold": 5, "NPS": 5}
    elif risk == "High" or preference == "Equity":
        weights = {"Emergency Fund": 15, "Fixed Deposit": 5, "PPF": 10, "Mutual Funds": 30, "Index Funds": 25, "Gold": 5, "NPS": 10}
    elif risk == "Low" or preference == "Debt":
        weights = {"Emergency Fund": 25, "Fixed Deposit": 25, "PPF": 20, "Mutual Funds": 10, "Index Funds": 5, "Gold": 5, "NPS": 10}
    else:
        weights = {"Emergency Fund": 20, "Fixed Deposit": 10, "PPF": 15, "Mutual Funds": 25, "Index Funds": 15, "Gold": 5, "NPS": 10}

    return [
        {
            "name": name,
            "percentage": pct,
            "amount": money(investable * pct / 100),
        }
        for name, pct in weights.items()
    ]
