"""Generate a stronger synthetic finance dataset for Finance Assistant V2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.finance_features import engineer_record

DATASET_PATH = Path("dataset/finance_assistant_dataset.csv")
RANDOM_SEED = 42


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def _spender_label(savings_ratio: float, emi_ratio: float, expense_ratio: float, discretionary_ratio: float) -> str:
    if savings_ratio >= 0.22 and emi_ratio <= 0.22 and expense_ratio <= 0.84 and discretionary_ratio <= 0.18:
        return "Safe"
    if savings_ratio >= 0.08 and emi_ratio <= 0.36 and expense_ratio <= 1.02:
        return "Moderate"
    return "Risky"


def _health_score(row: dict) -> float:
    score = 66
    score += float(row["savings_ratio"]) * 115
    score += min(float(row["goal_coverage_ratio"]), 1.4) * 5
    score -= max(float(row["emi_ratio"]) - 0.25, 0) * 115
    score -= max(float(row["expense_to_income_ratio"]) - 0.82, 0) * 95
    score -= max(float(row["shopping_ratio"]) - 0.12, 0) * 75
    score -= max(float(row["discretionary_ratio"]) - 0.20, 0) * 60
    if row["risk_level"] == "High" and float(row["savings_ratio"]) < 0.15:
        score -= 4
    return round(_clip(score, 5, 100), 2)


def _recommended_budget(row: dict) -> dict[str, float]:
    income = max(float(row["monthly_income"]), 1)
    emi_ratio = float(row["emi_ratio"])
    risk = str(row["risk_level"])
    preference = str(row["investment_preference"])
    dependents = int(row["dependents"])

    rent_pct = 0.24 if row["housing_type"] == "Rented" else 0.14 if row["housing_type"] == "Owned" else 0.08
    rent_pct += 0.02 if row["city_tier"] == "Tier 1" else -0.01 if row["city_tier"] == "Tier 3" else 0
    food_pct = _clip(0.11 + dependents * 0.017, 0.08, 0.19)
    education_pct = _clip(0.02 + dependents * 0.021, 0.02, 0.12)
    medical_pct = _clip(0.035 + dependents * 0.006, 0.03, 0.08)
    transport_pct = _clip(0.055 + (row["city_tier"] == "Tier 1") * 0.015, 0.04, 0.10)
    tax_pct = _clip(float(row["tax_pf_deductions"]) / income, 0.04, 0.24)
    emi_pct = _clip(emi_ratio, 0, 0.34)
    shopping_pct = _clip(0.13 - dependents * 0.008 - (emi_ratio > 0.30) * 0.03, 0.04, 0.14)

    savings_pct = 0.18 + (emi_ratio < 0.20) * 0.05 - (emi_ratio > 0.35) * 0.06
    savings_pct += 0.03 if risk == "Low" else 0.01 if risk == "Medium" else -0.01
    emergency_pct = 0.07 + (dependents >= 2) * 0.02 + (emi_ratio > 0.30) * 0.02
    investment_pct = 0.08 + (risk == "High") * 0.06 + (risk == "Medium") * 0.03
    investment_pct += 0.015 if preference == "Equity" else -0.015 if preference == "Debt" else 0

    budget = {
        "rec_rent_house_pct": _clip(rent_pct, 0.08, 0.31),
        "rec_food_pct": food_pct,
        "rec_shopping_pct": shopping_pct,
        "rec_emi_loans_pct": emi_pct,
        "rec_education_pct": education_pct,
        "rec_medical_pct": medical_pct,
        "rec_transport_pct": transport_pct,
        "rec_savings_pct": _clip(savings_pct, 0.08, 0.32),
        "rec_emergency_fund_pct": _clip(emergency_pct, 0.05, 0.13),
        "rec_investments_pct": _clip(investment_pct, 0.04, 0.22),
        "rec_tax_pf_pct": tax_pct,
    }

    total = sum(budget.values())
    if total > 0.96:
        scale = 0.96 / total
        budget = {key: value * scale for key, value in budget.items()}
    return {key: round(value * 100, 2) for key, value in budget.items()}


def _forecast_targets(row: dict, rng: np.random.Generator) -> dict[str, float]:
    income = float(row["monthly_income"])
    savings = float(row["savings"])
    expense_pressure = max(float(row["expense_to_income_ratio"]) - 0.85, 0)
    savings_momentum = 1 + float(row["savings_ratio"]) * 0.06 - expense_pressure * 0.04
    values = {}
    for month in range(1, 13):
        seasonal = 1 + 0.04 * np.sin(month / 12 * 2 * np.pi)
        noise = rng.normal(0, max(income * 0.01, 450))
        prediction = savings * (savings_momentum ** month) * seasonal - expense_pressure * income * month * 0.025 + noise
        values[f"forecast_savings_m{month}"] = round(_clip(prediction, -income, income * 0.65), 2)
    return values


def create_dataset(rows: int = 7500, output_path: Path = DATASET_PATH) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    records: list[dict] = []

    for _ in range(rows):
        monthly_income = float(np.round(rng.lognormal(mean=11.18, sigma=0.58), -2))
        monthly_income = _clip(monthly_income, 25000, 480000)
        annual_income = monthly_income * 12
        city_tier = rng.choice(["Tier 1", "Tier 2", "Tier 3"], p=[0.42, 0.38, 0.20])
        housing_type = rng.choice(["Rented", "Owned", "Family Home"], p=[0.52, 0.25, 0.23])
        dependents = int(rng.choice([0, 1, 2, 3, 4], p=[0.15, 0.28, 0.32, 0.18, 0.07]))
        risk_level = rng.choice(["Low", "Medium", "High"], p=[0.33, 0.45, 0.22])
        investment_preference = rng.choice(["Debt", "Balanced", "Equity"], p=[0.27, 0.48, 0.25])

        tax_rate = 0.045 + (annual_income > 700000) * 0.055 + (annual_income > 1200000) * 0.055
        tax_rate += (annual_income > 2000000) * 0.055
        tax_pf = monthly_income * _clip(tax_rate + rng.normal(0.025, 0.012), 0.04, 0.24)
        rent_base = 0.24 if housing_type == "Rented" else 0.14 if housing_type == "Owned" else 0.07
        rent_base *= 1.12 if city_tier == "Tier 1" else 0.9 if city_tier == "Tier 3" else 1.0
        rent_house = monthly_income * _clip(rent_base + rng.normal(0, 0.035), 0.035, 0.38)

        has_emi = rng.random() < (0.55 if housing_type == "Owned" else 0.40)
        emi_loans = monthly_income * _clip(rng.normal(0.19, 0.09), 0.02, 0.50) if has_emi else 0
        food = monthly_income * _clip(0.10 + dependents * 0.024 + rng.normal(0, 0.025), 0.06, 0.27)
        shopping = monthly_income * _clip(rng.normal(0.11, 0.055), 0.02, 0.31)
        education = monthly_income * _clip(dependents * rng.normal(0.034, 0.016), 0, 0.20)
        medical = monthly_income * _clip(0.035 + dependents * 0.008 + rng.normal(0, 0.018), 0.01, 0.14)
        transport = monthly_income * _clip(0.055 + (city_tier == "Tier 1") * 0.018 + rng.normal(0, 0.018), 0.02, 0.14)
        other = monthly_income * _clip(rng.normal(0.065, 0.036), 0.01, 0.19)
        savings_goal = monthly_income * _clip(rng.normal(0.21, 0.075), 0.05, 0.42)

        row = {
            "annual_income": round(annual_income, 2),
            "monthly_income": round(monthly_income, 2),
            "emi_loans": round(emi_loans, 2),
            "rent_house": round(rent_house, 2),
            "food": round(food, 2),
            "shopping": round(shopping, 2),
            "education": round(education, 2),
            "education_fees": round(education, 2),
            "medical": round(medical, 2),
            "transport": round(transport, 2),
            "transportation": round(transport, 2),
            "tax_pf_deductions": round(tax_pf, 2),
            "tax_pf": round(tax_pf, 2),
            "other_expenses": round(other, 2),
            "savings_goal": round(savings_goal, 2),
            "dependents": dependents,
            "city_tier": city_tier,
            "housing_type": housing_type,
            "investment_preference": investment_preference,
            "risk_level": risk_level,
        }
        row = engineer_record(row)
        row["financial_health_score"] = _health_score(row)
        row["spender_class"] = _spender_label(
            float(row["savings_ratio"]),
            float(row["emi_ratio"]),
            float(row["expense_to_income_ratio"]),
            float(row["discretionary_ratio"]),
        )
        row.update(_recommended_budget(row))
        row.update(_forecast_targets(row, rng))
        records.append(row)

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    dataset = create_dataset()
    print(f"Created {len(dataset):,} V2 rows at {DATASET_PATH}")
