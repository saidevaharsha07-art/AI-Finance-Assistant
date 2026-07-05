"""Train a savings forecasting neural model for Finance Assistant V2.

TensorFlow/Keras is used when available. If TensorFlow is not installed in the
local environment, this script trains a scikit-learn MLPRegressor fallback so the
forecasting pipeline remains runnable for local demos.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from generate_dataset import DATASET_PATH, create_dataset
from utils.finance_features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from utils.forecast_engine import FORECAST_MODEL_PATH

MODEL_DIR = Path("models")
KERAS_MODEL_PATH = MODEL_DIR / "dl_savings_forecaster.keras"
PREPROCESSOR_PATH = MODEL_DIR / "dl_forecast_preprocessor.joblib"
DL_METRICS_PATH = MODEL_DIR / "dl_metrics.txt"
DL_PLOT_PATH = MODEL_DIR / "dl_training_history.png"
FORECAST_TARGETS = [f"forecast_savings_m{month}" for month in range(1, 13)]


def load_dataset() -> pd.DataFrame:
    required = set(FEATURES + FORECAST_TARGETS)
    if not DATASET_PATH.exists():
        return create_dataset()
    df = pd.read_csv(DATASET_PATH)
    if not required.issubset(df.columns):
        return create_dataset()
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else matrix


def train_with_tensorflow(x_train, x_test, y_train, y_test, preprocessor) -> dict:
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(x_train.shape[1],)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.18),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.12),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(12, activation="linear"),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=90,
        batch_size=64,
        verbose=1,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True)],
    )
    predictions = model.predict(x_test).reshape(-1, 12)
    model.save(KERAS_MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(
        {
            "keras_model_path": str(KERAS_MODEL_PATH),
            "preprocessor": preprocessor,
            "engine": "TensorFlow/Keras savings forecaster",
            "is_keras": True,
        },
        FORECAST_MODEL_PATH,
    )
    _plot_history(history.history["loss"], history.history["val_loss"])
    return {"engine": "TensorFlow/Keras", "predictions": predictions}


def train_with_mlp(x_train, x_test, y_train, y_test, preprocessor) -> dict:
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=260,
        early_stopping=True,
        validation_fraction=0.18,
        random_state=42,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    joblib.dump(
        {"model": model, "preprocessor": preprocessor, "engine": "MLP neural-network savings forecaster"},
        FORECAST_MODEL_PATH,
    )
    _plot_history(model.loss_curve_, getattr(model, "validation_scores_", []), validation_is_score=True)
    return {"engine": "Scikit-learn MLP fallback", "predictions": predictions}


def _plot_history(train_values, validation_values, validation_is_score: bool = False) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(train_values, label="Training loss")
    if len(validation_values):
        label = "Validation score" if validation_is_score else "Validation loss"
        plt.plot(validation_values, label=label)
    plt.title("Savings Forecast Neural Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(DL_PLOT_PATH, dpi=150)


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    df = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(df[FEATURES], df[FORECAST_TARGETS], test_size=0.2, random_state=42)

    preprocessor = build_preprocessor()
    x_train_ready = dense(preprocessor.fit_transform(x_train))
    x_test_ready = dense(preprocessor.transform(x_test))

    try:
        trained = train_with_tensorflow(x_train_ready, x_test_ready, y_train.values, y_test.values, preprocessor)
    except Exception as exc:
        print(f"TensorFlow training unavailable ({exc}). Training MLP neural fallback.")
        trained = train_with_mlp(x_train_ready, x_test_ready, y_train.values, y_test.values, preprocessor)

    predictions = trained["predictions"]
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    DL_METRICS_PATH.write_text(
        f"Savings forecast model\nEngine: {trained['engine']}\nMAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR2: {r2:.3f}\n",
        encoding="utf-8",
    )
    print(DL_METRICS_PATH.read_text(encoding="utf-8"))
    print(f"Saved forecast bundle to {FORECAST_MODEL_PATH}")
    print(f"Saved training graph to {DL_PLOT_PATH}")


if __name__ == "__main__":
    main()
