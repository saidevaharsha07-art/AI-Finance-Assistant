"""Hybrid budget logic that combines ML outputs with finance constraints."""

from __future__ import annotations

from utils.finance_features import BUDGET_DISPLAY_NAMES, money


def constrained_budget(ml_percentages: list[float], row: dict) -> tuple[dict[str, float], dict[str, float]]:
    income = max(float(row["monthly_income"]), 1)
    percentages = {
        name: max(0.0, min(45.0, float(pct)))
        for name, pct in zip(BUDGET_DISPLAY_NAMES, ml_percentages)
    }

    emi_ratio = float(row["emi_ratio"]) * 100
    shopping_ratio = float(row["shopping_ratio"]) * 100
    expense_ratio = float(row["expense_to_income_ratio"]) * 100

    percentages["emi_loans"] = min(max(percentages["emi_loans"], min(emi_ratio, 30)), 35)
    percentages["tax_pf"] = min(max(percentages["tax_pf"], float(row["tax_ratio"]) * 100), 25)
    percentages["shopping"] = min(percentages["shopping"], 10 if expense_ratio > 90 else 14)
    if shopping_ratio > 14:
        percentages["shopping"] = min(percentages["shopping"], 9)

    minimum_savings = 18 if float(row["emi_ratio"]) < 0.25 else 12
    percentages["savings"] = max(percentages["savings"], minimum_savings)
    percentages["emergency_fund"] = max(percentages["emergency_fund"], 6)
    percentages["investments"] = max(percentages["investments"], 6)

    total = sum(percentages.values())
    if total > 96:
        protected = {"emi_loans", "tax_pf", "emergency_fund"}
        flexible_total = sum(value for key, value in percentages.items() if key not in protected)
        protected_total = sum(value for key, value in percentages.items() if key in protected)
        scale = max((96 - protected_total) / max(flexible_total, 1), 0.25)
        for key in percentages:
            if key not in protected:
                percentages[key] *= scale

    percentages = {key: round(value, 2) for key, value in percentages.items()}
    amounts = {key: money(income * pct / 100) for key, pct in percentages.items()}
    return percentages, amounts
