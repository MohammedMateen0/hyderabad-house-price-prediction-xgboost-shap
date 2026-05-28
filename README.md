# Hyderabad House Price Prediction using XGBoost + SHAP

## Overview

This project is an end-to-end Machine Learning application for predicting apartment prices in Hyderabad using XGBoost Regression and SHAP Explainability.

The application combines:
- machine learning
- explainable AI
- interactive web deployment

using Streamlit.

The system allows users to:
- enter apartment details
- predict apartment prices
- understand why the model made a prediction using SHAP explanations

---

# Project Features

## Machine Learning
- XGBoost Regressor
- Log-transformed target variable
- Feature engineering
- One-hot encoding

## Explainable AI (XAI)
- SHAP TreeExplainer
- Global feature importance
- Local prediction explanations
- Waterfall plots

## Streamlit Web Application
- Interactive sidebar inputs
- Real-time predictions
- Dynamic SHAP visualizations
- User-friendly dashboard

---

# Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Matplotlib
- Joblib

---

# Dataset Information

Dataset contains:
- 2276 apartment records
- 44 original features
- location and geo-coordinate data
- apartment amenities
- pricing information

### Important Features
- area
- no_of_bedrooms
- location
- city
- latitude
- longitude
- gymnasium
- swimmingpool
- clubhouse
- powerbackup

Target Variable:
- `log_price`

---

# Machine Learning Pipeline

## 1. Data Preprocessing
- removed target leakage
- categorical encoding using `pd.get_dummies()`
- train-test split
- feature alignment for deployment consistency

## 2. Model Training
XGBoost Regressor used for:
- nonlinear learning
- handling feature interactions
- high tabular-data performance

## 3. Explainability
SHAP TreeExplainer used for:
- feature attribution
- global explainability
- local explainability

---

# SHAP Explainability

## Global Explainability
SHAP summary plots showed:
- apartment area is the strongest driver of price
- latitude and longitude strongly affect pricing
- premium locations increase predictions
- amenities have smaller but meaningful effects

## Local Explainability
Waterfall plots explain:
- baseline prediction
- positive feature contributions
- negative feature contributions
- final prediction value

SHAP additive property:

prediction = baseline + sum(feature contributions)

---

# Streamlit Application

The Streamlit application allows users to:
- select apartment features
- generate price predictions
- visualize SHAP explanations interactively

### Features Included
- sidebar input widgets
- model prediction
- price conversion from log scale
- SHAP waterfall plot
- responsive layout using columns

Run locally:

```bash
streamlit run app/streamlit_app.py
```

---

# Project Structure

```bash
hyderabad-house-price-prediction-xgboost-shap/
│
├── app/
│   └── streamlit_app.py
│
├── models/
│   ├── xgb_model.pkl
│   └── training_columns.pkl
│
├── notebooks/
│   └── week8_day1_shap.ipynb
│
├── plots/
│   ├── shap_summary.png
│   ├── shap_bar.png
│   └── waterfall_plot.png
│
├── src/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Example Workflow

1. User enters apartment details
2. Streamlit app preprocesses inputs
3. XGBoost predicts log price
4. Prediction converted back to actual price
5. SHAP explains prediction contributions
6. Waterfall plot displayed to user

---

# Key Learning Outcomes

This project demonstrates:
- regression modeling
- feature engineering
- explainable AI (XAI)
- Streamlit deployment
- model serialization
- ML application architecture
- prediction interpretability

---

# Future Improvements

- Streamlit Cloud deployment
- FastAPI integration
- MLflow experiment tracking
- Docker containerization
- drift detection
- CI/CD pipeline
- automated retraining

---

# Installation

Clone repository:

```bash
git clone <your-repo-url>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run application:

```bash
streamlit run app/streamlit_app.py
```

---

# Author

Mohammed Mateen

Aspiring Machine Learning & Data Science Engineer focused on:
- explainable AI
- production ML systems
- deployment-ready machine learning applications