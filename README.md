# Hyderabad House Price Prediction using XGBoost + SHAP

## Overview

This project is an end-to-end machine learning system for predicting apartment prices in Hyderabad using XGBoost regression and SHAP explainability.

The project focuses not only on prediction accuracy but also on model interpretability, feature attribution, and business-level insights.

The model learns pricing patterns from:
- apartment area
- location
- geographic coordinates
- amenities
- resale information

SHAP (SHapley Additive exPlanations) is used to explain both:
- global model behavior
- local prediction decisions

---

# Problem Statement

Real-estate pricing depends on multiple interacting factors such as:
- property size
- premium locations
- amenities
- geographic positioning

The goal of this project is to build an explainable regression model capable of learning these pricing relationships while providing transparent feature-level explanations.

---

# Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib

---

# Dataset Information

Dataset contains:
- 2276 apartment records
- 44 original features
- pricing and amenity information
- location and geo-coordinate features

### Important Features
- area
- no_of_bedrooms
- latitude
- longitude
- location
- swimmingpool
- gymnasium
- powerbackup
- clubhouse
- liftavailable

Target Variable:
- `log_price`

---

# Machine Learning Pipeline

## 1. Data Preprocessing
- handled categorical variables using one-hot encoding
- separated features and target
- train-test split

## 2. Model Training
XGBoost Regressor used for:
- nonlinear learning
- feature interaction handling
- strong tabular data performance

## 3. Explainability
SHAP TreeExplainer used for:
- global explainability
- local explainability
- feature attribution analysis

---

# SHAP Explainability

## Global Explainability
SHAP summary plots revealed that:
- apartment area is the strongest price driver
- latitude and longitude significantly influence pricing
- premium locations such as Hitech City and Jubilee Hills increase predictions
- luxury amenities contribute smaller but meaningful effects

## Local Explainability
Waterfall plots explain individual apartment predictions by decomposing:
- baseline prediction
- positive feature contributions
- negative feature contributions

SHAP additive property:

prediction = baseline + sum(feature contributions)

---

# Results

## Key Insights
- larger apartments strongly increase predicted prices
- premium Hyderabad locations significantly affect valuation
- geo-coordinates successfully capture spatial pricing behavior
- amenities have secondary influence compared to location and area

---

# Project Structure

```bash
hyderabad-house-price-prediction-xgboost-shap/
│
├── notebooks/
│   ├── week8_day1_shap.ipynb
│
├── plots/
│   ├── shap_summary.png
│   ├── shap_dependence_plot.png
│   ├── waterfall_plot.png
│
├── src/
│
├── models/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Example SHAP Visualizations

## SHAP Summary Plot
Shows:
- feature importance
- feature impact direction
- feature value distribution

## SHAP Waterfall Plot
Explains:
- individual apartment prediction
- positive and negative feature pushes

---

# Future Improvements

- Streamlit deployment
- FastAPI inference API
- MLflow experiment tracking
- Docker containerization
- drift detection monitoring
- CI/CD pipeline integration

---

# Learning Outcomes

This project demonstrates:
- regression modeling
- feature engineering
- explainable AI (XAI)
- SHAP interpretation
- business-focused ML storytelling
- end-to-end ML workflow

---

# Author

Mohammed Mateen

Aspiring Machine Learning & Data Science Engineer focused on:
- explainable AI
- production ML systems
- deployment-ready machine learning projects