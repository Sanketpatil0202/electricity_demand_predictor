"""
app.py
------
Streamlit app for the Electricity Demand Predictor.

Run locally:
    streamlit run app.py

The app loads a pre-trained RandomForestRegressor pipeline (trained via
train_model.py) and lets users interactively predict electricity demand
from four inputs: Temperature, Humidity, Wind Speed and Hour of Day.
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join("model", "electricity_demand_model.pkl")

st.set_page_config(
    page_title="Electricity Demand Predictor",
    page_icon="⚡",
    layout="centered",
)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    bundle = joblib.load(MODEL_PATH)
    return bundle


bundle = load_model()

st.title("⚡ Electricity Demand Predictor")
st.write(
    "A Random Forest regression model that predicts electricity demand "
    "from weather and time-of-day features. Built with scikit-learn and "
    "served with Streamlit."
)

if bundle is None:
    st.error(
        "No trained model found. Run the following in your terminal, "
        "then reload this app:\n\n"
        "```bash\n"
        "python generate_sample_data.py\n"
        "python train_model.py\n"
        "```"
    )
    st.stop()

pipeline = bundle["pipeline"]
metrics = bundle.get("metrics", {})

st.divider()
st.subheader("🔧 Input features")

st.caption(
    "This model was trained on min-max normalized data (0 = lowest observed "
    "value, 1 = highest observed value in the training set). Move the "
    "sliders to explore how each feature affects predicted demand."
)

col1, col2 = st.columns(2)
with col1:
    temperature = st.slider("Temperature", 0.0, 1.0, 0.5, 0.01)
    humidity = st.slider("Humidity", 0.0, 1.0, 0.5, 0.01)
with col2:
    wind_speed = st.slider("Wind Speed", 0.0, 1.0, 0.5, 0.01)
    hour_of_day = st.slider("Hour of Day", 0.0, 1.0, 0.5, 0.01)

input_df = pd.DataFrame(
    [[temperature, humidity, wind_speed, hour_of_day]],
    columns=["Temperature", "Humidity", "WindSpeed", "HourOfDay"],
)

if st.button("Predict Electricity Demand", type="primary", use_container_width=True):
    prediction = min(max(pipeline.predict(input_df)[0], 0.0), 1.0)
    percentage = prediction * 100

    if prediction < 0.33:
        level, color = "Low", "🟢"
    elif prediction < 0.66:
        level, color = "Moderate", "🟡"
    else:
        level, color = "High", "🔴"

    c1, c2 = st.columns(2)
    c1.metric("Predicted Demand", f"{percentage:.1f}%")
    c2.metric("Demand Level", f"{color} {level}")

    st.progress(prediction)
    st.caption(
        f"Raw model output: {prediction:.4f} (0 = lowest observed demand in "
        "training data, 1 = highest)"
    )

st.divider()

with st.expander("📊 Model performance (on held-out test data)"):
    if metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R²", f"{metrics.get('R2', 0):.3f}")
        m2.metric("MAE", f"{metrics.get('MAE', 0):.4f}")
        m3.metric("RMSE", f"{metrics.get('RMSE', 0):.4f}")
        m4.metric("MSE", f"{metrics.get('MSE', 0):.4f}")
    else:
        st.write("No metrics available for this model.")

with st.expander("🌲 What's under the hood"):
    st.markdown(
        """
- **Model**: `RandomForestRegressor` (scikit-learn), wrapped in a `Pipeline`
- **Features**: Temperature, Humidity, Wind Speed, Hour of Day
- **Target**: Electricity Demand
- **Split**: 80% train / 20% test, `random_state=42`
- **Serialization**: `joblib`
        """
    )

with st.expander("📈 Try a batch of random scenarios"):
    if st.button("Generate 5 random scenarios"):
        rng = np.random.default_rng()
        sample = pd.DataFrame(
            rng.random((5, 4)),
            columns=["Temperature", "Humidity", "WindSpeed", "HourOfDay"],
        )
        sample["Predicted Demand"] = pipeline.predict(sample)
        st.dataframe(sample, use_container_width=True)

st.divider()
st.caption(
    "Built as a portfolio project • Model trained on synthetic/sample data "
    "for demonstration purposes • Swap in your own dataset to retrain."
)
