"""Dynamic advice generation for Finance Assistant V2."""

from __future__ import annotations


def generate_advice(row: dict, result: dict) -> list[str]:
    advice: list[str] = []
    score = float(result["health_score"])
    savings_ratio = float(row["savings_ratio"])
    emi_ratio = float(row["emi_ratio"])
    shopping_ratio = float(row["shopping_ratio"])
    expense_ratio = float(row["expense_to_income_ratio"])
    dependents = int(row.get("dependents", 0))
    risk_level = str(row.get("risk_level", "Medium"))
    investment_preference = str(row.get("investment_preference", "Balanced"))

    if expense_ratio > 1:
        advice.append("Your current expenses are above income. Freeze non-essential spending until the monthly cash flow turns positive.")
    if emi_ratio > 0.35:
        advice.append("Avoid taking new loans because EMI burden is high relative to income; consider refinancing or prepaying costly debt.")
    elif emi_ratio < 0.20 and score >= 70:
        advice.append("Debt burden is controlled, so surplus cash can be directed toward emergency reserves and long-term investments.")
    if shopping_ratio > 0.12:
        advice.append("Reduce shopping expenses because discretionary spending is above the recommended threshold for your income profile.")
    if savings_ratio < 0.15:
        advice.append("Automate savings on salary day and target at least 15-20% of monthly income before discretionary purchases.")
    elif savings_ratio >= 0.25:
        advice.append("Your savings ratio is strong; increase SIPs or long-term goal investments while keeping liquidity intact.")
    if dependents >= 2:
        advice.append("Because you have dependents, keep health insurance and an emergency fund ahead of high-risk investments.")
    if risk_level == "High" and score < 70:
        advice.append("Your risk appetite is high, but the score is not strong yet. Build a 3-6 month emergency fund before aggressive equity exposure.")
    if investment_preference == "Debt" and score >= 80:
        advice.append("Your profile can support a slightly higher growth allocation if it matches your goals and time horizon.")
    if len(advice) < 3:
        advice.append("Review fixed expenses once a quarter and compare actual spending with this recommended allocation.")
    return advice[:6]
