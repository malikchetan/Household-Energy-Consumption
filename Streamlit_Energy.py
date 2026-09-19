
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import pickle
from datetime import datetime, timedelta
import xgboost as xgb

st.set_page_config(page_title="Energy Forecasting App", layout="centered")

st.markdown(
    """
    <style>
    /* Overall App Background */
    .stApp {
        background: linear-gradient(to bottom right, #0f2027, #203a43, #2c5364); /* deep blue-gray gradient */
        color: #e0e0e0;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Main Title Styling */
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        color: #00ffcc; /* bright teal accent */
        margin-bottom: 20px;
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(to right, #00b4db, #0083b0);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }

    /* Info Boxes (predictions or explanations) */
    .info-box {
        background-color: #1f2a38dd; /* semi-transparent dark gray */
        padding: 18px;
        border-radius: 16px;
        box-shadow: 0px 6px 12px rgba(0,0,0,0.5);
        margin-bottom: 20px;
    }

    /* Streamlit charts container */
    .stChart {
        background: #1a1f28bb; /* slightly lighter dark container for charts */
        border-radius: 12px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


#STEP 1 :
st.title("🏡 Home Energy Forecast")

st.markdown("Forecast your home's energy consumption for the next 3 hours using ML models.")

st.sidebar.header("📊 User Input")

selected_hour = st.sidebar.selectbox("Select Hour", list(range(24)))

selected_weekday = st.sidebar.selectbox("Select Weekday (0 = Monday)", list(range(7)))

selected_month = st.sidebar.selectbox("Select Month", list(range(1, 13)))

model_name = st.sidebar.radio("Choose Model", ["Random Forest", "Gradient Boosting" , "XGBoost"])

model_scores = {
    "Random Forest": 0.88,
    "Gradient Boosting":0.90,
    "XGBoost": 0.91,
    }

selected_score = model_scores.get(model_name)
st.markdown(f"  {model_name} (R² = {selected_score:.2f})")

st.sidebar.markdown("---")


# LOAD THE MODELS

@st.cache_resource
def load_resources():
    rf = joblib.load("Models/random_forest.pkl")
    
    # Correct XGBoost loading
    
    xgb_model = joblib.load("Models/xgb_model.pkl")
    
    gb = joblib.load("Models/gradient_boosting.pkl")

    # Load datasets
    df_rf = pd.read_csv("Dataframes/df_rf.csv")
    df_xgb = pd.read_csv("Dataframes/df_xgb.csv")
    df_gb = pd.read_csv("Dataframes/df_gb.csv")

    return rf, xgb_model, gb, df_rf, df_xgb, df_gb

# STEP 3:
rf, xgb, gb, df_rf, df_xgb, df_gb = load_resources()




# STEP 4 :
def fetch_latest_row(df, hour, weekday, month):
    subset = df[(df['Hour'] == hour) & (df['Weekday'] == weekday) & (df['Month'] == month)]
    if subset.empty:
        st.warning("⚠️ No matching data found. Try different Hour/Weekday/Month.")
        return None
    return subset.head(6)


# --- Prediction functions ---

def predict_gb(model, df, hour, weekday, month):
    features = [
        'Hour', 'Weekday', 'Month', 'Is_weekend', 'Power_Wasted', 
        'Laundry_kwh', 'Kitchen_kwh', 'Appliances_kwh',
        'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
        'lag_6', 'Voltage_lag1', 'Voltage_lag2', 'Current_lag1',
        'Current_lag2'
        ]

    data = fetch_latest_row(df, hour, weekday, month)
    return model.predict(data[features]) if data is not None else None


def predict_rf(model, df, hour, weekday, month):
    features = [
        'Hour', 'Weekday', 'Month', 'Is_weekend', 'Power_Wasted', 
        'Laundry_kwh', 'Kitchen_kwh', 'Appliances_kwh',
        'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
        'lag_6', 'Voltage_lag1', 'Voltage_lag2', 'Current_lag1',
        'Current_lag2'
        ]

    data = fetch_latest_row(df, hour, weekday, month)
    return model.predict(data[features]) if data is not None else None


def predict_xgb(model, df, hour, weekday, month):
    features = [
       'Hour', 'Weekday', 'Month', 'Is_weekend', 'Power_Wasted', 
        'Laundry_kwh', 'Kitchen_kwh', 'Appliances_kwh',
        'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
       'lag_6', 'Voltage_lag1', 'Voltage_lag2', 'Current_lag1', 'Current_lag2',
       'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12', 'lag_7', 'lag_8',
       'lag_9', 'lag_10', 'lag_11', 'lag_12']
    data = fetch_latest_row(df, hour, weekday, month)
    return model.predict(data[features]) if data is not None else None

    
  


# STEP --5 :

if st.button("🔮 Predict Energy Usage"):
    with st.spinner("Running prediction..."):

        predictions = None
        if model_name == "Random Forest":
            predictions = predict_rf(rf, df_rf, selected_hour, selected_weekday, selected_month)

        elif model_name == "XGBoost":
            predictions = predict_xgb(xgb, df_xgb, selected_hour, selected_weekday, selected_month)

        elif model_name == "Gradient Boosting":
            predictions = predict_gb(gb, df_gb, selected_hour, selected_weekday, selected_month)


        if predictions is not None:
            st.success(f"✅ Prediction Complete using {model_name}")
            st.subheader("📈 Forecast (Next 3 Hours):")

            # Generate future times starting from entered hour
            base_time = datetime(2025, 1, 1, selected_hour, 0, 0)  # fixed dummy date
            time_stamps = [(base_time + timedelta(minutes=30 * (i+1))) for i in range(6)]
            formatted_times = [ts.strftime("%I:%M %p") for ts in time_stamps]

            # Create DataFrame
            prediction_df = pd.DataFrame({
                "Time": formatted_times,
                "Predicted Energy (kWh)": predictions[:6]
            })

            # Display Table
            st.dataframe(prediction_df.style.format({"Predicted Energy (kWh)": "{:.2f}"}))
            st.markdown("📆 This is the amount of energy consumed.")

            # Display Line Chart
            st.line_chart(
                data=prediction_df.set_index("Time"),
                use_container_width=True
            )




