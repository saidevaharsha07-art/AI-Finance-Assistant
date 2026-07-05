"""Train V2 ML models for budget allocation, health score, and spender class."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from generate_dataset import DATASET_PATH, create_dataset
from utils.finance_features import BUDGET_TARGETS, CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES

MODEL_DIR = Path("models")
REPORT_PATH = MODEL_DIR / "ml_metrics.csv"
BUDGET_MODEL_PATH = MODEL_DIR / "budget_recommender.joblib"
CLASSIFIER_PATH = MODEL_DIR / "spender_classifier.joblib"
HEALTH_MODEL_PATH = MODEL_DIR / "health_score_model.joblib"


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def load_dataset() -> pd.DataFrame:
    required = set(FEATURES + BUDGET_TARGETS + ["financial_health_score", "spender_class"])
    if not DATASET_PATH.exists():
        return create_dataset()
    df = pd.read_csv(DATASET_PATH)
    if not required.issubset(df.columns) or len(df) < 5000:
        return create_dataset()
    return df


def train_budget_model(df: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame]:
    x_train, x_test, y_train, y_test = train_test_split(df[FEATURES], df[BUDGET_TARGETS], test_size=0.2, random_state=42)
    candidates = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=14, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=35,
            max_depth=18,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }
    rows = []
    best_model = None
    best_mae = np.inf
    for name, estimator in candidates.items():
        pipeline = Pipeline([("preprocess", _preprocessor()), ("model", MultiOutputRegressor(estimator))])
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        rows.append({"task": "budget_allocation", "model": name, "mae": mae, "rmse": rmse, "r2_or_accuracy": r2})
        if mae < best_mae:
            best_mae = mae
            best_model = pipeline
    assert best_model is not None
    return best_model, pd.DataFrame(rows)


def train_health_model(df: pd.DataFrame) -> tuple[Pipeline, dict, pd.DataFrame]:
    x_train, x_test, y_train, y_test = train_test_split(
        df[FEATURES], df["financial_health_score"], test_size=0.2, random_state=42
    )
    model = Pipeline(
        [
            ("preprocess", _preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=80,
                    max_depth=18,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = pd.DataFrame(
        [
            {
                "task": "health_score_regression",
                "model": "Random Forest Regressor",
                "mae": mean_absolute_error(y_test, predictions),
                "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
                "r2_or_accuracy": r2_score(y_test, predictions),
            }
        ]
    )
    importances = _feature_importance(model)
    return model, importances, metrics


def _feature_importance(pipeline: Pipeline) -> dict[str, float]:
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    names = list(preprocessor.get_feature_names_out())
    clean_names = [name.split("__", 1)[-1] for name in names]
    values = getattr(model, "feature_importances_", np.zeros(len(clean_names)))
    grouped: dict[str, float] = {}
    for name, value in zip(clean_names, values):
        base = name.split("_Tier", 1)[0].split("_Rented", 1)[0].split("_Owned", 1)[0]
        grouped[base] = grouped.get(base, 0.0) + float(value)
    return dict(sorted(grouped.items(), key=lambda item: item[1], reverse=True))


def train_classifier(df: pd.DataFrame) -> tuple[Pipeline, float, pd.DataFrame]:
    x_train, x_test, y_train, y_test = train_test_split(
        df[FEATURES],
        df["spender_class"],
        test_size=0.2,
        random_state=42,
        stratify=df["spender_class"],
    )
    classifier = Pipeline(
        [
            ("preprocess", _preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=80,
                    max_depth=16,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    classifier.fit(x_train, y_train)
    accuracy = accuracy_score(y_test, classifier.predict(x_test))
    metrics = pd.DataFrame(
        [{"task": "spender_classification", "model": "Random Forest Classifier", "mae": np.nan, "rmse": np.nan, "r2_or_accuracy": accuracy}]
    )
    return classifier, accuracy, metrics


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    df = load_dataset()
    budget_model, budget_metrics = train_budget_model(df)
    health_model, health_importance, health_metrics = train_health_model(df)
    classifier, classifier_accuracy, classifier_metrics = train_classifier(df)

    joblib.dump({"model": budget_model, "features": FEATURES, "targets": BUDGET_TARGETS}, BUDGET_MODEL_PATH)
    joblib.dump(
        {"model": health_model, "features": FEATURES, "feature_importance": health_importance},
        HEALTH_MODEL_PATH,
    )
    joblib.dump(
        {"model": classifier, "features": FEATURES, "labels": ["Safe", "Moderate", "Risky"], "accuracy": classifier_accuracy},
        CLASSIFIER_PATH,
    )
    metrics = pd.concat([budget_metrics, health_metrics, classifier_metrics], ignore_index=True)
    metrics.to_csv(REPORT_PATH, index=False)
    print(metrics.sort_values(["task", "mae"], na_position="last"))
    print(f"Saved budget model to {BUDGET_MODEL_PATH}")
    print(f"Saved health-score model to {HEALTH_MODEL_PATH}")
    print(f"Saved spender classifier to {CLASSIFIER_PATH}")


if __name__ == "__main__":
    main()
