import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Delivery Outcome Predictor",
    page_icon="🚚",
    layout="wide",
)

MODEL_PATH = Path(__file__).parent / "decision_tree_model.sav"

FEATURES = [
    "Delivery_Distance",
    "Traffic_Congestion",
    "Weather_Condition",
    "Delivery_Slot",
    "Driver_Experience",
    "Num_Stops",
    "Vehicle_Age",
    "Road_Condition_Score",
    "Package_Weight",
    "Fuel_Efficiency",
    "Warehouse_Processing_Time",
]

@st.cache_resource
def load_model():
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)

st.title("🚚 Delivery Outcome Predictor")
st.write(
    "Enter the delivery and operating conditions below, then run the model "
    "to get its predicted class and class probabilities."
)

if not MODEL_PATH.exists():
    st.error("Model file not found. Put `decision_tree_model.sav` in the same folder as `app.py`.")
    st.stop()

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

st.info(
    "**Important:** The uploaded model contains a classifier, but the meaning of class "
    "`0` and class `1` and the original category encodings were not provided. "
    "This app therefore shows the model's raw class labels. For categorical-looking "
    "features, enter the same numeric encoding used when the model was trained."
)

with st.form("prediction_form"):
    st.subheader("Delivery details")
    c1, c2, c3 = st.columns(3)

    with c1:
        delivery_distance = st.number_input(
            "Delivery distance", min_value=0.0, value=10.0, step=1.0,
            help="Use the same distance unit as the training data."
        )
        traffic_congestion = st.number_input(
            "Traffic congestion (encoded value)", min_value=0.0, value=2.0, step=1.0,
            help="Numeric value/code from your original training dataset."
        )
        weather_condition = st.number_input(
            "Weather condition (encoded value)", min_value=0.0, value=1.0, step=1.0,
            help="Numeric value/code from your original training dataset."
        )
        delivery_slot = st.number_input(
            "Delivery slot (encoded value)", min_value=0.0, value=1.0, step=1.0,
            help="Numeric value/code from your original training dataset."
        )

    with c2:
        driver_experience = st.number_input(
            "Driver experience", min_value=0.0, value=3.0, step=1.0,
            help="Use the same unit as the training data (for example, years)."
        )
        num_stops = st.number_input(
            "Number of stops", min_value=0, value=5, step=1
        )
        vehicle_age = st.number_input(
            "Vehicle age", min_value=0.0, value=3.0, step=1.0,
            help="Use the same unit as the training data (for example, years)."
        )
        road_condition_score = st.number_input(
            "Road condition score", min_value=0.0, value=5.0, step=0.5,
            help="Use the score scale used in the training dataset."
        )

    with c3:
        package_weight = st.number_input(
            "Package weight", min_value=0.0, value=5.0, step=0.5,
            help="Use the same weight unit as the training data."
        )
        fuel_efficiency = st.number_input(
            "Fuel efficiency", min_value=0.0, value=10.0, step=0.5,
            help="Use the same unit as the training data."
        )
        warehouse_processing_time = st.number_input(
            "Warehouse processing time", min_value=0.0, value=60.0, step=5.0,
            help="Use the same time unit as the training data."
        )

    submitted = st.form_submit_button("Predict delivery outcome", type="primary")

if submitted:
    values = [
        delivery_distance,
        traffic_congestion,
        weather_condition,
        delivery_slot,
        driver_experience,
        num_stops,
        vehicle_age,
        road_condition_score,
        package_weight,
        fuel_efficiency,
        warehouse_processing_time,
    ]
    input_df = pd.DataFrame([values], columns=FEATURES)

    try:
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.classes_)
        confidence = float(np.max(probabilities))

        st.divider()
        st.subheader("Prediction")
        r1, r2 = st.columns(2)
        r1.metric("Predicted class", str(prediction))
        r2.metric("Model probability for predicted class", f"{confidence:.1%}")

        prob_df = pd.DataFrame({
            "Class": [str(c) for c in classes],
            "Probability": probabilities,
        })
        st.caption("These are the model's class probabilities, not a guarantee of the real-world outcome.")
        st.bar_chart(prob_df.set_index("Class"))

        with st.expander("Review submitted input"):
            st.dataframe(input_df, use_container_width=True, hide_index=True)

        with st.expander("Model details"):
            st.write(f"Model type: `{type(model).__name__}`")
            st.write(f"Number of expected features: `{getattr(model, 'n_features_in_', 'unknown')}`")
            st.write("Features used by the model:")
            st.code("\n".join(FEATURES))
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.divider()
st.caption("Prototype built with Streamlit and scikit-learn. Validate the feature units, category mappings, and class meanings before using for operational decisions.")
