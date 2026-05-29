import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI

app=FastAPI()

model=joblib.load("models/xgb_model.pkl")
training_columns=joblib.load("models/training_columns.pkl")
@app.get("/")
def home():
    return {
        "message":"Hyderabad House Price Prediction API Running"
    }

@app.post("/predict")
def predict(
    area:float,
    bedrooms:int,
    resale:int,
    latitude:float,
    longitude:float
):
    input_df=pd.DataFrame([{
        "area":area,
        "no_of_bedrooms":bedrooms,
        "resale":resale,
        "latitude":latitude,
        "longitude":longitude
    }])
    input_df=input_df.reindex(
        columns=training_columns,
        fill_value=0
    )
    prediction=model.predict(input_df)[0]
    price=float(np.exp(prediction))
    return {
        "Predicted_Price":round(price,2)
    }