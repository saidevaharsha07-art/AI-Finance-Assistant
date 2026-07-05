"""Shared feature engineering used by training and inference."""

from __future__ import annotations

EXPENSE_FIELDS = [
    "emi_loans",
    "rent_house",
    "food",
    "shopping",
    "education",
    "medical",
    "transport",
    "tax_pf_deductions",
    "other_expenses",
]

BASE_FEATURES = [
    "annual_income",
    "monthly_income",
    "emi_loans",
    "rent_house",
    "food",
    "shopping",
    "education",
    "medical",
    "transport",
    "tax_pf_deductions",
    "other_expenses",
    "savings_goal",
    "dependents",
    "city_tier",
    "housing_type",
    "investment_preference",
    "risk_level",
]

ENGINEERED_FEATURES = [
    "total_expenses",
    "savings",
    "savings_ratio",
    "emi_ratio",
    "shopping_ratio",
    "expense_to_income_ratio",
    "essential_ratio",
    "discretionary_ratio",
    "tax_ratio",
    "goal_coverage_ratio",
]

FEATURES = BASE_FEATURES + ENGINEERED_FEATURES
NUMERIC_FEATURES = [
    "annual_income",
    "monthly_income",
    "emi_loans",
    "rent_house",
    "food",
    "shopping",
    "education",
    "medical",
    "transport",
    "tax_pf_deductions",
    "other_expenses",
    "savings_goal",
    "dependents",
] + ENGINEERED_FEATURES
CATEGORICAL_FEATURES = ["city_tier", "housing_type", "investment_preference", "risk_level"]

BUDGET_TARGETS = [
    "rec_rent_house_pct",
    "rec_food_pct",
    "rec_shopping_pct",
    "rec_emi_loans_pct",
    "rec_education_pct",
    "rec_medical_pct",
    "rec_transport_pct",
    "rec_savings_pct",
    "rec_emergency_fund_pct",
    "rec_investments_pct",
    "rec_tax_pf_pct",
]

BUDGET_DISPLAY_NAMES = [
    "rent_house",
    "food",
    "shopping",
    "emi_loans",
    "education",
    "medical",
    "transport",
    "savings",
    "emergency_fund",
    "investments",
    "tax_pf",
]


def money(value: float | int | str) -> float:
    return round(float(value or 0), 2)


def engineer_record(row: dict) -> dict:
    data = dict(row)
    monthly_income = max(float(data.get("monthly_income", 0) or 0), 1.0)
    total_expenses = sum(float(data.get(field, 0) or 0) for field in EXPENSE_FIELDS)
    savings = float(data.get("monthly_income", 0) or 0) - total_expenses
    essential = (
        float(data.get("rent_house", 0) or 0)
        + float(data.get("food", 0) or 0)
        + float(data.get("education", 0) or 0)
        + float(data.get("medical", 0) or 0)
        + float(data.get("transport", 0) or 0)
        + float(data.get("tax_pf_deductions", 0) or 0)
    )
    discretionary = float(data.get("shopping", 0) or 0) + float(data.get("other_expenses", 0) or 0)

    data["total_expenses"] = money(total_expenses)
    data["savings"] = money(savings)
    data["savings_ratio"] = round(savings / monthly_income, 4)
    data["emi_ratio"] = round(float(data.get("emi_loans", 0) or 0) / monthly_income, 4)
    data["shopping_ratio"] = round(float(data.get("shopping", 0) or 0) / monthly_income, 4)
    data["expense_to_income_ratio"] = round(total_expenses / monthly_income, 4)
    data["essential_ratio"] = round(essential / monthly_income, 4)
    data["discretionary_ratio"] = round(discretionary / monthly_income, 4)
    data["tax_ratio"] = round(float(data.get("tax_pf_deductions", 0) or 0) / monthly_income, 4)
    data["goal_coverage_ratio"] = round(max(savings, 0) / max(float(data.get("savings_goal", 0) or 0), 1), 4)
    return data


def health_category(score: float) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Moderate"
    return "Risky"


def parse_form(form: dict[str, str]) -> dict[str, float | int | str]:
    data: dict[str, float | int | str] = {}
    for field in [
        "annual_income",
        "monthly_income",
        "emi_loans",
        "rent_house",
        "food",
        "shopping",
        "education",
        "medical",
        "transport",
        "tax_pf_deductions",
        "other_expenses",
        "savings_goal",
    ]:
        data[field] = money(form.get(field, 0) or 0)
    data["dependents"] = int(form.get("dependents", 0) or 0)
    data["city_tier"] = form.get("city_tier", "Tier 1")
    data["housing_type"] = form.get("housing_type", "Rented")
    data["investment_preference"] = form.get("investment_preference", "Balanced")
    data["risk_level"] = form.get("risk_level", "Medium")
    if float(data["annual_income"]) <= 0 and float(data["monthly_income"]) > 0:
        data["annual_income"] = money(float(data["monthly_income"]) * 12)
    return data
