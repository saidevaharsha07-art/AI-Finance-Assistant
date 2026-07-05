# AI-Powered Finance Assistant

AI-Powered Finance Assistant is a major-project-level intelligent personal finance web application built using **Flask, Machine Learning, Deep Learning, SQLite, Chart.js, and PDF reporting**.  
It helps users analyze their financial condition, classify spending behavior, estimate future savings, generate budget recommendations, simulate financial changes, and download a structured financial report.

The project combines **data-driven financial analysis** with **explainable AI insights** to make personal budgeting and forecasting more interactive, understandable, and practical.

---

# Overview

Managing personal finances can be difficult when users do not clearly understand where their money goes, whether they are overspending, how healthy their financial profile is, or how their current habits will affect future savings.

**AI-Powered Finance Assistant** is designed to solve this by acting as a smart finance dashboard that accepts user financial details and then uses **machine learning, forecasting, feature engineering, and rule-based financial logic** to provide:

- a **financial health score**
- a **spender classification**
- **future savings forecasts**
- **budget allocation suggestions**
- **personalized financial advice**
- **investment recommendations**
- **what-if financial simulations**
- **PDF financial report generation**
- **history tracking using SQLite**

This project is built as a **major project / portfolio project** to demonstrate the use of **ML + DL + full-stack web development** in a practical financial domain.

---

# Problem Statement

Many people struggle to evaluate their financial health because financial information is often scattered across income, expenses, savings, loans, and lifestyle spending.  
A user may know their monthly salary and expenses, but still not know:

- whether their spending pattern is safe or risky
- how much they should allocate to different categories
- whether they are likely to meet future savings goals
- which financial factors are hurting their financial health score
- what practical steps they should take to improve their financial profile

The goal of this project is to build an **AI-powered finance assistant** that can analyze user financial data and generate intelligent insights such as **financial health scoring, spender classification, savings forecasting, budgeting advice, explainable score factors, and downloadable reports**.

---

# How the System Works

The application follows a complete **input → feature engineering → prediction → explanation → reporting** pipeline.

## 1. User Input
The user enters financial details such as:

- monthly income
- EMI / loan payments
- rent / housing expenses
- food expenses
- shopping expenses
- education expenses
- medical expenses
- transport expenses
- tax / PF deductions
- other expenses
- savings goal
- current savings
- dependents
- city tier
- housing type
- investment preference
- risk level

---

## 2. Feature Engineering
The system converts raw financial inputs into meaningful financial indicators such as:

- savings ratio
- EMI ratio
- shopping ratio
- expense-to-income ratio
- essential spending ratio
- discretionary spending ratio
- tax ratio
- goal coverage ratio

These engineered features are used by the ML/DL models and logic engines to better understand the user's financial behavior.

---

## 3. Machine Learning Predictions
The app loads trained ML models and predicts:

- **Spender Class** → `Safe`, `Moderate`, or `Risky`
- **Financial Health Score** → score out of 100
- **Budget Recommendation Signals** for financial allocation assistance

---

## 4. Forecasting Engine
The forecasting module estimates:

- future monthly savings
- 6-month savings forecast
- 12-month savings forecast
- expense trend forecast

The project supports **deep learning / neural forecasting** when available, and uses a fallback model when necessary so the system remains runnable locally.

---

## 5. Explainability Engine
The application identifies **which financial factors positively or negatively affected the user’s score**, and presents them in a readable explanation section.

Examples:
- high savings ratio improves score
- high EMI burden reduces flexibility
- overspending in discretionary categories lowers financial health

This makes the output more understandable instead of showing only a score.

---

## 6. Advice and Investment Recommendation
Based on the user profile and model outputs, the app generates:

- personalized spending-control advice
- savings improvement suggestions
- emergency fund guidance
- investment allocation suggestions
- profile-based financial recommendations

---

## 7. What-If Simulation
The user can change certain values and simulate a different financial scenario to see how their score, class, savings, or forecast would change under modified conditions.

---

## 8. History and Reporting
Each prediction can be stored in a **SQLite database**, allowing the app to maintain a history of previous runs.  
The app can also generate a **PDF financial report** summarizing the analysis and recommendations.

---

# Features

## Core Financial Intelligence
- Financial health score prediction out of 100
- Health category classification (`Excellent`, `Good`, `Moderate`, `Risky`)
- Spender class prediction (`Safe`, `Moderate`, `Risky`)
- Overspending detection
- Estimated monthly savings calculation

## Forecasting & Planning
- 1-month, 6-month, and 12-month savings forecast
- Expense trend forecasting
- Future financial planning insights

## Budgeting & Recommendations
- ML-informed monthly budget allocation
- Category-wise budget recommendations
- Suggested investment plan
- Personalized advice engine

## Explainability & Analysis
- Explainable AI section showing score drivers
- Positive and negative financial contributors
- What-if simulation panel for scenario testing

## Dashboard & Reporting
- Flask-based finance dashboard UI
- Chart.js visualizations for financial insights
- PDF financial report generation
- SQLite-based history and comparison tracking

---

# Tech Stack

## Frontend
- **HTML5**
- **CSS3**
- **JavaScript**
- **Chart.js**

## Backend
- **Python**
- **Flask**

## Machine Learning / Data
- **scikit-learn**
- **NumPy**
- **Pandas**
- **Joblib**

## Deep Learning / Forecasting
- **TensorFlow / Keras** *(when available)*
- fallback neural / MLP forecasting approach for local execution

## Database & Reporting
- **SQLite**
- **ReportLab** *(for PDF generation)*

---

# Project Structure

```text
AI-Finance-Assistant/
│
├── app.py                      # Main Flask application
├── generate_dataset.py         # Generates synthetic finance dataset
├── train_ml_model.py           # Trains ML models for scoring/classification/budgeting
├── train_dl_model.py           # Trains forecasting / deep learning model
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
│
├── dataset/
│   └── finance_assistant_dataset.csv   # Generated dataset used for training
│
├── models/
│   ├── budget_recommender.joblib       # Budget recommendation model
│   ├── health_score_model.joblib       # Financial health score model
│   ├── spender_classifier.joblib       # Spender classification model
│   ├── savings_forecaster.joblib       # Savings forecasting model / fallback
│   ├── ml_metrics.csv                  # ML evaluation metrics
│   ├── dl_metrics.txt                  # DL / forecasting metrics
│   └── dl_training_history.png         # Training history visualization
│
├── notebooks/                  # Optional notebooks / experiments
│
├── static/
│   ├── style.css               # Dashboard styling
│   └── app.js                  # Frontend logic / charts / simulation handling
│
├── templates/
│   └── index.html              # Main Flask Jinja dashboard template
│
└── utils/
    ├── finance_features.py     # Shared feature engineering logic
    ├── budget_engine.py        # Budget recommendation logic
    ├── forecast_engine.py      # Savings and expense forecasting logic
    ├── explain_engine.py       # Score explanation logic
    ├── advice_engine.py        # Personalized advice generator
    ├── investment_engine.py    # Investment recommendation logic
    ├── history_manager.py      # SQLite history handling
    └── pdf_report.py           # PDF report generation
```

---

# Model Pipeline

The project uses a multi-stage model pipeline.

## 1) Dataset Generation

`generate_dataset.py` creates a synthetic household finance dataset containing realistic financial profiles and engineered finance features.

The dataset includes:

- income and expense details
- EMI / debt burden
- savings goal and current savings
- demographic and lifestyle factors
- engineered ratios and financial indicators
- target labels for score, class, and forecast

---

## 2) ML Training

`train_ml_model.py` trains models for:

### Spender Classification
Predicts whether the user is:
- `Safe`
- `Moderate`
- `Risky`

### Financial Health Score Regression
Predicts a numerical financial health score out of 100.

### Budget Recommendation Support
Helps guide category-wise monthly allocation suggestions.

Saved outputs:
- `models/budget_recommender.joblib`
- `models/health_score_model.joblib`
- `models/spender_classifier.joblib`
- `models/ml_metrics.csv`

---

## 3) Forecast Model Training

`train_dl_model.py` trains the savings forecasting model.

Primary objective:
- forecast future savings / financial trend behavior

Implementation note:
- Uses **TensorFlow / Keras** when available
- Falls back to a neural / MLP-based model if TensorFlow is unavailable
- Keeps the project runnable on local systems even without a heavy DL environment

Saved outputs may include:
- forecasting model files
- metrics
- training history graph

---

## 4) Inference in the Flask App

When the user submits the form:
1. raw values are collected
2. engineered finance features are generated
3. trained models are loaded
4. score / class / forecast predictions are computed
5. budget, advice, explanation, and investment logic are applied
6. result is displayed in the dashboard
7. history is stored in SQLite
8. report can be exported to PDF

---

# Installation & Run

## 1. Clone the repository
```bash
git clone https://github.com/your-username/AI-Finance-Assistant.git
cd AI-Finance-Assistant
```

## 2. Create a virtual environment (recommended)

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 4. Generate the dataset
```bash
python generate_dataset.py
```

---

## 5. Train the machine learning models
```bash
python train_ml_model.py
```

---

## 6. Train the forecasting / deep learning model
```bash
python train_dl_model.py
```

---

## 7. Run the Flask app
```bash
python app.py
```

---

## 8. Open in browser
```text
http://127.0.0.1:5000
```

---

# Screenshots

> Add screenshots of your project here after pushing to GitHub.

Suggested screenshots to include:
- Home dashboard / input form
- Prediction result dashboard
- Explainability section
- Budget recommendation cards
- What-if simulation section
- PDF report output
- User history table

---

# Future Scope

This project can be extended further in multiple ways:

- integrate real bank statement / CSV upload support
- support multiple user accounts and authentication
- connect to live stock / mutual fund / SIP APIs
- add more advanced time-series forecasting models
- integrate NLP-based financial chatbot support
- add anomaly detection for suspicious spending
- add monthly reminder / alert system
- deploy as a cloud-hosted finance assistant web app
- build a mobile-friendly or React-based frontend version
- improve dataset realism with real-world anonymized financial data

---

# Author

**Sai Deva Harsha**  
AI / ML / Full-Stack Learner  
GitHub: **saidevaharsha07-art**

---

# Disclaimer

This project is created for **educational, academic, and portfolio purposes**.  
It is **not certified financial, tax, or investment advice**.
