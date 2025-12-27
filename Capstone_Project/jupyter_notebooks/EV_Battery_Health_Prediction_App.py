import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="EV Battery Health Predictor",
    page_icon="🔋",
    layout="wide"
)

# --- Title and Description ---
st.title("🔋 EV Battery Health Prediction System")
st.markdown("""
This application predicts the **State of Health (SoH)** of an EV battery using various machine learning models.
- **Supervised Models**: Random Forest and XGBoost predict SoH based on a single cycle's average metrics.
- **Deep Learning Models**: LSTM and GRU predict SoH based on a sequence of the last 10 cycles.
""")

# --- Load Models and Scaler ---
@st.cache_resource
def load_all_models():
    """Loads all models and the scaler from disk."""
    models = {}
    # Assuming models are in the same directory as app.py. 
    # If running from a different dir, adjust path accordingly.
    model_path = "." 

    # Load ML models
    try:
        models['rf'] = joblib.load(os.path.join(model_path, "rf_soh_model.joblib"))
        models['xgb'] = joblib.load(os.path.join(model_path, "xgb_model.joblib"))
    except FileNotFoundError as e:
        st.error(f"Could not find ML model files: {e}. Please run the notebook to generate them.")

    # Load DL models
    try:
        models['lstm'] = load_model(os.path.join(model_path, "lstm_soh_model.keras"), compile=False)
        models['gru'] = load_model(os.path.join(model_path, "gru_soh_model.keras"), compile=False)
    except (FileNotFoundError, OSError) as e:
        st.warning(f"Could not find DL model files (.h5): {e}. LSTM/GRU models will be unavailable.")

    # Load Scaler
    try:
        scaler = joblib.load(os.path.join(model_path, "soh_scaler.joblib"))
    except FileNotFoundError:
        st.error("Scaler file (soh_scaler.joblib) not found. Cannot make predictions.")
        return None, None # Scaler is essential

    return models, scaler

models, scaler = load_all_models()

# --- Sidebar for User Input ---
st.sidebar.header("Input Battery Parameters")
model_choice = st.sidebar.selectbox(
    "Choose a Prediction Model",
    ("Random Forest", "XGBoost", "LSTM", "GRU")
)

st.sidebar.markdown("---")

# --- Input Form ---
is_deep_learning = model_choice in ["LSTM", "GRU"]

if not is_deep_learning:
    st.sidebar.subheader("Single Cycle Input (for RF/XGBoost)")
    cycle = st.sidebar.number_input("Cycle Number", min_value=1, value=100, step=1)
    avg_voltage = st.sidebar.slider("Average Voltage (V)", 3.0, 4.2, 3.5, 0.01)
    avg_current = st.sidebar.slider("Average Current (A)", -2.0, 2.0, -1.8, 0.01)
    avg_temp = st.sidebar.slider("Average Temperature (°C)", 20.0, 40.0, 32.0, 0.1)
else:
    st.sidebar.subheader("Sequential Input (for LSTM/GRU)")
    st.sidebar.info("Please upload a CSV with data for the last 10 cycles.")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# --- Prediction Logic ---
if st.sidebar.button("Predict SoH"):
    if models and scaler:
        prediction = None
        model_key = model_choice.lower().replace(" ", "")
        if model_key.startswith("random"): model_key = "rf"
        elif model_key.startswith("xgb"): model_key = "xgb"

        if not is_deep_learning:
            if models.get(model_key):
                # Create a DataFrame from the inputs
                input_data = pd.DataFrame({
                    'cycle': [cycle],
                    'avg_voltage_measured': [avg_voltage],
                    'avg_current_measured': [avg_current],
                    'avg_temp_measured': [avg_temp]
                })

                # Scale the input data
                input_scaled = scaler.transform(input_data)

                # Select model and predict
                model = models.get(model_key)
                prediction = model.predict(input_scaled)[0]
            else:
                st.error(f"{model_choice} model is not available.")

        else: # Deep Learning models
            if uploaded_file is not None:
                if models.get(model_key):
                    try:
                        input_df = pd.read_csv(uploaded_file)
                        # Validate CSV
                        required_cols = ['cycle', 'avg_voltage_measured', 'avg_current_measured', 'avg_temp_measured']
                        if not all(col in input_df.columns for col in required_cols):
                            st.error(f"CSV must contain the columns: {', '.join(required_cols)}")
                        elif len(input_df) != 10:
                            st.error("CSV must contain exactly 10 rows (timesteps).")
                        else:
                            # Prepare data for prediction
                            input_data = input_df[required_cols]
                            input_scaled = scaler.transform(input_data)
                            input_sequence = np.array([input_scaled]) # Shape: (1, 10, 4)

                            # Select model and predict
                            model = models.get(model_key)
                            prediction = model.predict(input_sequence)[0][0]

                    except Exception as e:
                        st.error(f"An error occurred while processing the file: {e}")
                else:
                    st.error(f"{model_choice} model is not available.")
            else:
                st.warning("Please upload a CSV file for LSTM/GRU prediction.")

        if prediction is not None:
            st.subheader(f"Prediction using {model_choice}")
            st.metric(label="Predicted State of Health (SoH)", value=f"{prediction:.2%}")
            if prediction < 0.7:
                st.warning("Battery health is critical. Consider replacement.")
            else:
                st.success("Battery health is good.")

    else:
        st.error("Models or scaler could not be loaded. Prediction unavailable.")

# --- Instructions ---
st.markdown("---")
st.header("How to Use")
st.markdown("""
1.  **Select a Model**: Choose from Random Forest, XGBoost, LSTM, or GRU in the sidebar.
2.  **Provide Input**:
    *   For **Random Forest/XGBoost**, use the sliders and input fields to enter the metrics for a single battery cycle.
    *   For **LSTM/GRU**, upload a CSV file containing the data for the last 10 consecutive cycles. The CSV must have the following columns: `cycle`, `avg_voltage_measured`, `avg_current_measured`, `avg_temp_measured`.
3.  **Predict**: Click the "Predict SoH" button.
4.  **View Result**: The predicted State of Health will be displayed.
""")
