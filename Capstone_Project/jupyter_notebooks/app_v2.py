# ============================================================
# EV Battery Health Prediction System (ML + DL + SHAP)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os
import matplotlib.pyplot as plt
import boto3
from io import BytesIO
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

# ============================================================
# Environment & Page Config
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="EV Battery Health Predictor",
    page_icon="🔋",
    layout="wide"
)

st.title("🔋 EV Battery Health Prediction System")
st.markdown("""
Predict **EV Battery State of Health (SoH)** using:
- **ML Models**: Random Forest / XGBoost (single-cycle)
- **DL Models**: LSTM / GRU (full degradation history)
""")

# ============================================================
# Model & Scaler Loading
# ============================================================

@st.cache_resource
def load_all_models():
    models = {}
    model_path = os.getcwd()

    try:
        models["rf"] = joblib.load(f"{model_path}/rf_soh_model.joblib")
        models["xgb"] = joblib.load(f"{model_path}/xgb_model.joblib")
    except FileNotFoundError:
        st.warning("RF / XGBoost models not found.")

    try:
        models["lstm"] = load_model(f"{model_path}/lstm_soh_model.h5", compile=False)
        models["gru"] = load_model(f"{model_path}/gru_soh_model.h5", compile=False)
    except (FileNotFoundError, OSError):
        st.warning("LSTM / GRU models not found.")

    try:
        scaler = joblib.load(f"{model_path}/soh_scaler.joblib")
    except FileNotFoundError:
        st.error("Scaler not found.")
        return None, None

    return models, scaler

models, scaler = load_all_models()

# ============================================================
# SHAP Explainability Utility (ML ONLY)
# ============================================================

def plot_shap_explainability(model, input_scaled, feature_names, model_name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values[0]
    }).sort_values(by="SHAP Value", key=abs, ascending=False)

    st.subheader(f"Model Explainability – {model_name}")

    # --- SHAP Bar Plot ---
    fig_bar, ax = plt.subplots(figsize=(5, 3))
    ax.barh(shap_df["Feature"], shap_df["SHAP Value"])
    ax.set_title("Feature Impact on SoH Prediction")
    ax.invert_yaxis()
    st.pyplot(fig_bar)

    # --- SHAP Waterfall Plot ---
    fig_wf = plt.figure()
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_scaled[0],
            feature_names=feature_names
        ),
        show=False
    )
    st.pyplot(fig_wf)

# ============================================================
# S3 Data Loader (DL Models)
# ============================================================

@st.cache_data
def load_data_from_s3(bucket, key):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION")
    )

    obj = s3.get_object(Bucket=bucket, Key=key)
    raw_df = pd.read_parquet(BytesIO(obj["Body"].read()))

    df = raw_df[raw_df["type"] == "discharge"]
    df = df.groupby(["battery_name", "cycle"]).agg({
        "Voltage_measured": "mean",
        "Current_measured": "mean",
        "Temperature_measured": "mean"
    }).reset_index()
    df = df.sort_values(["battery_name", "cycle"])

    return df.rename(columns={
        "Voltage_measured": "avg_voltage_measured",
        "Current_measured": "avg_current_measured",
        "Temperature_measured": "avg_temp_measured"
    })

# ============================================================
# Sequence Generator (DL)
# ============================================================

def create_sequences(data, time_steps=10):
    return np.array([
        data[i:i + time_steps]
        for i in range(len(data) - time_steps + 1)
    ])

# ============================================================
# Sidebar Configuration
# ============================================================

st.sidebar.header("Configuration")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ("Random Forest", "XGBoost", "LSTM", "GRU")
)
model_key_map = {
    "Random Forest": "rf",
    "XGBoost": "xgb",
    "LSTM": "lstm",
    "GRU": "gru"
}

model_key = model_key_map[model_choice]
is_dl = model_key in ["lstm", "gru"]

# ============================================================
# Input Handling
# ============================================================

input_df = None

if is_dl:
    input_mode = "Batch"
else:
    input_mode = st.sidebar.radio("Input Mode", ("Single Cycle", "Batch"))

if input_mode == "Batch":
    source = st.sidebar.radio("Data Source", ("Amazon S3", "Upload CSV"))

    if source == "Amazon S3":
        input_df = load_data_from_s3(
            "s3aravindh973515031797",
            "EV_Battery_Health_Source/EV_Battery_Health_Source.parquet"
        )
    else:
        file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if file:
            input_df = pd.read_csv(file)

    # Filter for specific battery to prevent sequence mixing
    if input_df is not None and "battery_name" in input_df.columns:
        battery_list = input_df["battery_name"].unique()
        selected_battery = st.sidebar.selectbox("Select Battery", battery_list)
        input_df = input_df[input_df["battery_name"] == selected_battery].copy()
        # Sort by cycle to ensure correct sequence order
        input_df = input_df.sort_values("cycle")

if input_mode == "Single Cycle":
    cycle = st.sidebar.number_input("Cycle", min_value=1, value=100)
    avg_voltage = st.sidebar.slider("Avg Voltage (V)", 3.0, 4.2, 3.6)
    avg_current = st.sidebar.slider("Avg Current (A)", -2.0, 0.0, -1.0)
    avg_temp = st.sidebar.slider("Avg Temp (°C)", 20.0, 45.0, 25.0)

# ============================================================
# Run Prediction
# ============================================================

if st.button("Run Prediction"):

    if not models or not scaler:
        st.error("Models or scaler unavailable.")
        st.stop()

    # -----------------------------
    # ML MODELS (RF / XGB)
    # -----------------------------
    if not is_dl and models.get(model_key):

        if input_mode == "Single Cycle":
            input_data = pd.DataFrame({
                "cycle": [cycle],
                "avg_voltage_measured": [avg_voltage],
                "avg_current_measured": [avg_current],
                "avg_temp_measured": [avg_temp]
            })

            input_scaled = scaler.transform(input_data)
            pred = models[model_key].predict(input_scaled)[0]

            tab_info, tab_soh, tab_rul = st.tabs(["App Info", "State of Health (SoH)", "Remaining Useful Life (RUL)"])

            with tab_info:
                st.markdown("### ℹ️ App Functionalities")
                st.markdown("""
                This application provides an intelligent interface for EV battery health monitoring:
                *   **State of Health (SoH) Prediction**: Uses Machine Learning (Random Forest/XGBoost) to estimate battery health from single-cycle metrics.
                *   **Remaining Useful Life (RUL)**: Estimates remaining cycles based on current SoH and a nominal 1000-cycle life.
                *   **Explainability**: Visualizes feature contributions using SHAP values to understand model decisions.
                """)

            with tab_soh:
                st.metric("Predicted SoH", f"{pred:.2%}")

                if pred < 0.70:
                    st.error("⚠️ Battery Health Critical")
                else:
                    st.success("✅ Battery Health Acceptable")

                with st.expander("🔍 Model Explainability (SHAP)"):
                    plot_shap_explainability(
                        models[model_key],
                        input_scaled,
                        input_data.columns.tolist(),
                        model_choice
                    )
            
            with tab_rul:
                st.subheader("Estimated Remaining Useful Life")
                # Heuristic: RUL = (SoH - 0.70) / 0.30 * 1000 (Assuming 1000 cycle life to 70% SoH)
                rul_cycles = max(0, int((pred - 0.70) / 0.30 * 1000))
                rul_pct = min(100.0, max(0.0, (rul_cycles / 1000) * 100))
                
                c1, c2 = st.columns(2)
                c1.metric("Estimated RUL", f"{rul_cycles} Cycles")
                c2.metric("RUL Percentage", f"{rul_pct:.1f}%")
                
                st.progress(min(1.0, max(0.0, rul_cycles / 1000)))
                
                st.info("""
                **Methodology Note:** RUL is estimated using a physics-based heuristic assuming a linear decay 
                from **100% to 70% SoH** (End-of-Life) over a nominal lifespan of **1000 cycles**.
                \n$$ \\text{RUL} = \\frac{\\text{Current SoH} - 0.70}{0.30} \\times 1000 $$
                """)
        
        elif input_mode == "Batch" and input_df is not None:
            features = input_df[
                ["cycle", "avg_voltage_measured", "avg_current_measured", "avg_temp_measured"]
            ].values
            features_scaled = scaler.transform(features)
            
            preds = models[model_key].predict(features_scaled)
            
            input_df["Predicted SoH"] = preds
            
            tab_info, tab_soh, tab_rul = st.tabs(["App Info", "State of Health (SoH)", "Remaining Useful Life (RUL)"])
            
            with tab_info:
                st.markdown("### ℹ️ App Functionalities")
                st.markdown("""
                This application provides an intelligent interface for EV battery health monitoring:
                *   **Batch SoH Prediction**: Processes historical cycle data to map degradation trends using ML models.
                *   **RUL Trajectory**: Projects the Remaining Useful Life curve over the battery's history.
                *   **Visual Analysis**: Interactive charts to track health decline across cycles.
                """)

            with tab_soh:
                st.subheader(f"Batch Prediction Results ({model_choice})")
                st.line_chart(input_df.set_index("cycle")["Predicted SoH"])
                st.metric("Latest Predicted SoH", f"{preds[-1]:.2%}")
            
            with tab_rul:
                st.subheader("RUL Trajectory")
                # Vectorized heuristic calculation
                input_df["Predicted RUL"] = np.maximum(0, (input_df["Predicted SoH"] - 0.70) / 0.30 * 1000)
                latest_rul = int(input_df['Predicted RUL'].iloc[-1])
                latest_rul_pct = min(100.0, max(0.0, (latest_rul / 1000) * 100))
                
                st.line_chart(input_df.set_index("cycle")["Predicted RUL"])
                
                c1, c2 = st.columns(2)
                c1.metric("Latest Estimated RUL", f"{latest_rul} Cycles")
                c2.metric("RUL Percentage", f"{latest_rul_pct:.1f}%")
                
                st.info("""
                **Methodology Note:** RUL is estimated using a physics-based heuristic assuming a linear decay 
                from **100% to 70% SoH** (End-of-Life) over a nominal lifespan of **1000 cycles**.
                \n$$ \\text{RUL} = \\frac{\\text{Current SoH} - 0.70}{0.30} \\times 1000 $$
                """)

    # -----------------------------
    # DL MODELS (LSTM / GRU)
    # -----------------------------
    elif is_dl and input_df is not None and models.get(model_key):

        features = input_df[
            ["cycle", "avg_voltage_measured", "avg_current_measured", "avg_temp_measured"]
        ].values

        features_scaled = scaler.transform(features)
        X_seq = create_sequences(features_scaled, 10)

        preds = models[model_key].predict(X_seq).flatten()
        cycles = input_df["cycle"].values[9:]

        chart_df = pd.DataFrame({
            "Cycle": cycles,
            "Predicted SoH": preds
        }).set_index("Cycle")

        tab_info, tab_soh, tab_rul = st.tabs(["App Info", "State of Health (SoH)", "Remaining Useful Life (RUL)"])
        
        with tab_info:
             st.markdown("### ℹ️ App Functionalities")
             st.markdown("""
             This application provides an intelligent interface for EV battery health monitoring:
             *   **Deep Learning Prediction**: Uses LSTM/GRU networks to capture temporal degradation patterns from cycle sequences.
             *   **RUL Forecasting**: Estimates remaining useful life based on the trajectory learned by the neural network.
             *   **Trend Visualization**: Displays the complete degradation curve and RUL projection.
             """)

        with tab_soh:
            st.line_chart(
                chart_df.rename(
                    columns={"Predicted SoH": "Predicted SoH"}
                ),
                height=350,
                use_container_width=True
            )
            st.caption("**X-axis:** Cycle   **Y-axis:** Predicted SoH")
            # Add matplotlib chart with labeled axes for DL predictions
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(cycles, preds, label='Predicted SoH')
            ax.set_xlabel("Cycle")
            ax.set_ylabel("Predicted SoH")
            ax.set_title("Predicted Battery State of Health (SoH) Over Cycles")
            ax.legend()
            st.pyplot(fig)
            st.metric("Latest Predicted SoH", f"{preds[-1]:.2%}")
        
        with tab_rul:
            st.subheader("RUL Trajectory")
            rul_preds = np.maximum(0, (preds - 0.70) / 0.30 * 1000)
            rul_df = pd.DataFrame({
                "Cycle": cycles,
                "Predicted RUL": rul_preds
            }).set_index("Cycle")
            
            st.line_chart(rul_df, height=350, use_container_width=True)
            
            latest_rul = int(rul_preds[-1])
            latest_rul_pct = min(100.0, max(0.0, (latest_rul / 1000) * 100))
            
            c1, c2 = st.columns(2)
            c1.metric("Latest Estimated RUL", f"{latest_rul} Cycles")
            c2.metric("RUL Percentage", f"{latest_rul_pct:.1f}%")
            
            st.info("""
            **Methodology Note:** RUL is estimated using a physics-based heuristic assuming a linear decay 
            from **100% to 70% SoH** (End-of-Life) over a nominal lifespan of **1000 cycles**.
            \n$$ \\text{RUL} = \\frac{\\text{Current SoH} - 0.70}{0.30} \\times 1000 $$
            """)

    else:
        st.warning("Required inputs or model missing.")
