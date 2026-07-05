"""Readable model explanation layer for the finance dashboard."""

from __future__ import annotations


def explain_score(row: dict, health_score: float, feature_importance: dict[str, float] | None = None) -> list[dict]:
    factors: list[dict] = []

    def add(name: str, impact: float, detail: str) -> None:
        factors.append({"name": name, "impact": round(impact, 1), "detail": detail})

    savings_ratio = float(row["savings_ratio"])
    emi_ratio = float(row["emi_ratio"])
    shopping_ratio = float(row["shopping_ratio"])
    discretionary_ratio = float(row["discretionary_ratio"])
    expense_ratio = float(row["expense_to_income_ratio"])
    goal_coverage = float(row["goal_coverage_ratio"])

    if savings_ratio >= 0.25:
        add("Strong savings ratio", 10 + min(savings_ratio * 20, 8), "Your projected savings are comfortably above a healthy monthly baseline.")
    elif savings_ratio < 0.08:
        add("Weak savings ratio", -14, "Low or negative monthly savings is the biggest pressure on the score.")

    if emi_ratio > 0.35:
        add("High EMI burden", -16, "Loan payments are above the recommended comfort zone of income.")
    elif emi_ratio < 0.18:
        add("Controlled EMI burden", 7, "Debt payments are manageable relative to income.")

    if shopping_ratio > 0.14:
        add("High shopping ratio", -9, "Discretionary shopping is above the model's healthy spending pattern.")
    elif discretionary_ratio < 0.14:
        add("Low discretionary spending", 6, "Flexible expenses are controlled, which improves resilience.")

    if expense_ratio > 1:
        add("Overspending", -18, "Total monthly expenses exceed income, creating immediate cash-flow risk.")
    elif expense_ratio < 0.78:
        add("Healthy expense ratio", 8, "Your expenses leave room for savings and investments.")

    if goal_coverage >= 1:
        add("Savings goal covered", 6, "Expected savings can meet or exceed your stated goal.")
    else:
        add("Savings goal gap", -6, "Expected savings are below the monthly goal you entered.")

    if feature_importance:
        top = sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)[:3]
        names = ", ".join(name.replace("_", " ") for name, _ in top)
        add("Model focus", 0, f"The trained model emphasized these drivers for similar users: {names}.")

    return sorted(factors, key=lambda item: abs(item["impact"]), reverse=True)[:6]
