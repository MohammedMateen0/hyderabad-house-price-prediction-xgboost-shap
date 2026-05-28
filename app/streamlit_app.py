import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import shap

import os

st.write(os.getcwd())
st.set_page_config(
    page_title="Hyderabad House Price Predictior",
    page_icon="🏠",
    layout='wide'
)
st.title("🏠 Hyderabad House Price Prediction")
st.write("Predictt apartment prices using XGBoost + SHAP")


@st.cache_resource
def load_model():
    return joblib.load("models/xgb_model.pkl")

@st.cache_resource
def load_columns():
    return joblib.load("models/training_columns.pkl")

model=load_model()

tarining_columns=load_columns()

st.sidebar.header("Apartment Features")

area = st.sidebar.number_input(
    "Area (sqft)",
    min_value=500,
    max_value=10000,
    value=1500
)

bedrooms = st.sidebar.number_input(
    "Bedrooms",
    min_value=1,
    max_value=10,
    value=3
)

resale=st.sidebar.selectbox(
    "Resale",
    [0,1]
)

location = st.sidebar.selectbox(
    "Location",
    [
        "Hitech City",
        "Gachibowli",
        "Jubilee Hills",
        "Banjara Hills",
        "Madhapur",
        "Kondapur"
    ]
)
city=st.sidebar.selectbox(
    'City',
    ['Hyderabad']
)
latitude = st.sidebar.number_input(
    "Latitude",
    value=17.3850,
    format="%.6f"
)

longitude = st.sidebar.number_input(
    "Longitude",
    value=78.4867,
    format="%.6f"
)
gymnasium = st.sidebar.selectbox("Gymnasium", [0, 1])
swimmingpool = st.sidebar.selectbox("Swimming Pool", [0, 1])
clubhouse = st.sidebar.selectbox("Clubhouse", [0, 1])
liftavailable = st.sidebar.selectbox("Lift Available", [0, 1])
powerbackup = st.sidebar.selectbox("Power Backup", [0, 1])
carparking = st.sidebar.selectbox("Car Parking", [0, 1])

input_data = {
    "area": area,
    "location": location,
    "city": city,
    "no_of_bedrooms": bedrooms,
    "resale": resale,
    "latitude": latitude,
    "longitude": longitude,
    "gymnasium": gymnasium,
    "swimmingpool": swimmingpool,
    "clubhouse": clubhouse,
    "liftavailable": liftavailable,
    "powerbackup": powerbackup,
    "carparking": carparking
}

input_df = pd.DataFrame([input_data])

input_df = pd.get_dummies(
    input_df,
    columns=["location", "city"]
)

input_df=input_df.reindex(
    columns=tarining_columns,
    fill_value=0
)

if st.button("Predict Price"):
    prediction=model.predict(input_df)

    log_price=prediction[0]
    actual_price=np.exp(log_price)
    col1,col2=st.columns(2)

    with col1:
        st.subheader("Prediction Result")
        st.success(
            f"Predicted Apartment Price: ₹{actual_price:,.2f}Lakhs "
        )
        st.metric(
            label="Log Price",
            value=f"{log_price:.3f}"
        )
        st.write("### Input Features")
        st.dataframe(input_df)
    
    with col2:
        st.subheader("SHAP Explainability")

        explainer = shap.TreeExplainer(model)

        shap_values = explainer(input_df)

        fig, ax = plt.subplots(figsize=(10, 6))

        shap.plots.waterfall(
            shap_values[0],
            show=False
        )

        st.pyplot(fig)
    st.markdown("---")

st.write(
    """
    Built using:
    - XGBoost
    - SHAP Explainability
    - Streamlit
    """
)