# AI-Powered Finance Assistant V2

An upgraded major-project-level finance assistant built with Flask, scikit-learn, neural forecasting, Chart.js, SQLite history, and PDF reporting.

## V2 Features

- Clean financial input form and dashboard
- ML spender classification: `Safe`, `Moderate`, or `Risky`
- ML financial health score regression out of 100
- Health category: `Excellent`, `Good`, `Moderate`, or `Risky`
- ML-informed monthly budget allocation
- Future savings forecast for 1, 6, and 12 months
- Expense trend forecast
- Explainable AI section showing score drivers
- Dynamic personalized advice
- Suggested investment plan
- What-if analysis endpoint and UI panel
- PDF financial report download
- SQLite history with score/savings/class comparison

## Project Structure

```text
app.py
generate_dataset.py
train_ml_model.py
train_dl_model.py
requirements.txt
README.md
dataset/
models/
notebooks/
static/
templates/
utils/
```

The `utils/` package contains the V2 intelligence modules:

- `finance_features.py` shared feature engineering
- `budget_engine.py` hybrid ML + rule budget constraints
- `forecast_engine.py` savings and expense forecasting
- `explain_engine.py` score explanations
- `advice_engine.py` personalized advice
- `investment_engine.py` investment recommendations
- `history_manager.py` SQLite history
- `pdf_report.py` PDF generation

## Dataset

The dataset generator creates 7,500 realistic synthetic household finance rows with:

- annual and monthly income
- EMI/loans
- rent/house, food, shopping, education, medical, transport, tax/PF, other expenses
- savings goal, savings, total expenses
- dependents, city tier, housing type, investment preference, risk level
- engineered features: savings ratio, EMI ratio, shopping ratio, expense-to-income ratio, essential ratio, discretionary ratio, tax ratio, goal coverage ratio
- financial health score and spender class
- 12-month savings forecast targets

## Train Models

```bash
python generate_dataset.py
python train_ml_model.py
python train_dl_model.py
```

`train_ml_model.py` saves:

- `models/budget_recommender.joblib`
- `models/health_score_model.joblib`
- `models/spender_classifier.joblib`
- `models/ml_metrics.csv`

`train_dl_model.py` uses TensorFlow/Keras when available. If TensorFlow is not installed, it trains a scikit-learn MLP neural fallback so the forecast feature remains runnable locally.

## Run the App

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Routes

- `/` main dashboard
- `/predict` prediction flow
- `/simulate` what-if simulation JSON endpoint
- `/history` recent prediction history
- `/download_report/<record_id>` PDF financial report
- `/api/history` JSON history

## Note

This project is for educational and portfolio use. It is not certified financial, tax, or investment advice.
