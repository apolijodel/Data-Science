"""Educational Streamlit interface for the persisted health insurance cross-sell model."""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).with_name("health_insurance_model.joblib")
METADATA_PATH = Path(__file__).with_name("model_metadata.json")
DEFAULT_THRESHOLD = 0.5

st.set_page_config(page_title="Health Insurance Cross-Sell Demo", page_icon="🚗")
st.title("Health Insurance Cross-Sell Prediction Demo")
st.warning(
    "Educational demonstration only. This model estimates how likely an existing health insurance customer "
    "is to be interested in vehicle insurance. It is not a business decision on its own."
)

if not MODEL_PATH.exists():
    st.error(
        "No trained model found. Run all cells in `notebook/health_insurance_model.ipynb`, "
        "including the export cell, to create `app/health_insurance_model.joblib`."
    )
    st.stop()

@st.cache_resource
def load_model(path: str, modified_time: float):
    # modified_time is only part of the cache key: re-exporting the model from the notebook
    # changes it, so the new file is loaded without restarting the app.
    return joblib.load(path)


model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)

# The export cell in the model notebook writes the F1-optimal decision threshold and the
# validation metrics here. Only ~12% of customers respond, so the default 0.5 cutoff almost
# never predicts "interested" (F1 ~ 0); the tuned threshold is the realistic operating point.
metadata = {}
if METADATA_PATH.exists():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
recommended_threshold = float(metadata.get("f1_optimal_threshold", DEFAULT_THRESHOLD))

with st.sidebar:
    st.header("Decision threshold")
    threshold = st.slider(
        "Classify as interested when probability >=",
        min_value=0.05, max_value=0.95, value=round(recommended_threshold, 3), step=0.005,
        help="The notebook's F1-optimal threshold is the default. Lower it to catch more responders "
             "(more calls, more false positives); raise it to be more selective.",
    )
    if metadata:
        st.caption(
            f"Model: {metadata.get('model', 'unknown')} · "
            f"Validation ROC-AUC {metadata.get('validation_roc_auc', float('nan')):.3f} · "
            f"PR-AUC {metadata.get('validation_pr_auc', float('nan')):.3f} · "
            f"F1 {metadata.get('validation_f1_at_threshold', float('nan')):.3f} at threshold {recommended_threshold:.3f}"
        )
    else:
        st.caption("`model_metadata.json` not found; using 0.5 as the starting threshold.")

st.write("Enter the customer's details, then select **Estimate interest**.")

with st.form("prediction_form"):
    st.subheader("Customer information")
    gender = st.selectbox("Gender", ["Male", "Female"], help="Customer's recorded gender in the training data.")
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1, help="Customer age in years (training data: 20-85).")
    driving_license = st.selectbox("Driving license", ["Yes", "No"], help="Whether the customer holds a driving license.")
    region_code = st.number_input("Region code", min_value=0, max_value=52, value=28, step=1, help="Anonymised region code (0-52). Region 28 is the most common in the training data.")
    previously_insured = st.selectbox("Previously insured", ["No", "Yes"], help="Whether the customer already has vehicle insurance. Already-insured customers almost never respond.")
    vehicle_age = st.selectbox("Vehicle age", ["< 1 Year", "1-2 Year", "> 2 Years"], help="Age of the customer's vehicle.")
    vehicle_damage = st.selectbox("Vehicle damage", ["Yes", "No"], help="Whether the vehicle was damaged in the past. One of the two strongest predictors.")
    annual_premium = st.number_input("Annual premium", min_value=0.0, value=30000.0, step=500.0, help="Amount the customer pays per year for health insurance (training median ~31,700).")
    policy_sales_channel = st.number_input("Policy sales channel", min_value=0, max_value=200, value=26, step=1, help="Anonymised code of the channel used to reach the customer (agents, mail, phone, ...). Channels 26, 124 and 152 are the most common.")
    vintage = st.number_input("Vintage (days)", min_value=0, max_value=400, value=150, step=1, help="Number of days the customer has been with the company (training data: 10-299).")

    submitted = st.form_submit_button("Estimate interest")

if submitted:
    # Column names and types must match what the pipeline was trained on (Kaggle train.csv columns minus id/Response).
    customer = pd.DataFrame(
        [
            {
                "Gender": gender,
                "Age": int(age),
                "Driving_License": 1 if driving_license == "Yes" else 0,
                "Region_Code": int(region_code),
                "Previously_Insured": 1 if previously_insured == "Yes" else 0,
                "Vehicle_Age": vehicle_age,
                "Vehicle_Damage": vehicle_damage,
                "Annual_Premium": float(annual_premium),
                "Policy_Sales_Channel": int(policy_sales_channel),
                "Vintage": int(vintage),
            }
        ]
    )
    # Every model the notebook can export has predict_proba except Linear Regression
    # (included there only as a baseline comparison, see health_insurance_model.ipynb
    # Section 4) -- fall back to its raw output rather than crashing if it was exported.
    if hasattr(model.named_steps["model"], "predict_proba"):
        probability = float(model.predict_proba(customer)[0, 1])
    else:
        probability = float(model.predict(customer)[0])
    prediction = int(probability >= threshold)

    st.subheader("Model output")
    st.metric("Probability of interest in vehicle insurance", f"{probability:.1%}")
    if prediction == 1:
        st.error(f"The model classified this customer as Response=1 (interested) at threshold {threshold:.3f}.")
    else:
        st.success(f"The model classified this customer as Response=0 (not interested) at threshold {threshold:.3f}.")
    st.caption("This is a model estimate for demonstration. Use it to prioritise outreach, not to make final decisions.")
