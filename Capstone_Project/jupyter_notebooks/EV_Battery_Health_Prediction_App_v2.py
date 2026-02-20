# ============================================================
# EV Battery Health Prediction System (ML + DL + SHAP)
# ============================================================

import os
import warnings

# Suppress TensorFlow oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress scikit-learn version warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Suppress TensorFlow deprecation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='keras')

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import boto3
from io import BytesIO
import tempfile
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
    bucket_name = "s3aravindh973515031797"
    prefix = "ML_DL_Models"

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION")
    )

    def load_s3_joblib(key):
        with BytesIO() as f:
            s3.download_fileobj(bucket_name, key, f)
            f.seek(0)
            return joblib.load(f)

    def load_s3_keras(key):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
            s3.download_fileobj(bucket_name, key, tmp)
            tmp_path = tmp.name
        try:
            return load_model(tmp_path, compile=False)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    try:
        models["rf"] = load_s3_joblib(f"{prefix}/rf_soh_model.joblib")
        models["xgb"] = load_s3_joblib(f"{prefix}/xgb_model.joblib")
    except Exception as e:
        st.warning(f"RF / XGBoost models not found in S3: {e}")

    try:
        models["lstm"] = load_s3_keras(f"{prefix}/lstm_soh_model.h5")
        models["gru"] = load_s3_keras(f"{prefix}/gru_soh_model.h5")
    except Exception as e:
        st.warning(f"LSTM / GRU models not found in S3: {e}")

    try:
        scaler = load_s3_joblib(f"{prefix}/soh_scaler.joblib")
    except Exception as e:
        st.error(f"Scaler not found in S3: {e}")
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
# Main Layout & Tabs
# ============================================================

tab_home, tab_ml, tab_dl = st.tabs(["🏠 Home", "🤖 Machine Learning", "🧠 Deep Learning"])

# ============================================================
# Tab 1: Home / App Info
# ============================================================

with tab_home:
    st.markdown("### Welcome to the EV Battery Health Prediction System")
    st.markdown("""
    This application serves as a comprehensive tool for monitoring and predicting the health of Electric Vehicle (EV) batteries. 
    It leverages both traditional Machine Learning and advanced Deep Learning techniques to estimate the **State of Health (SoH)** 
    and **Remaining Useful Life (RUL)**.
    """)
    
    st.divider()
    
    st.subheader("🔧 Configuration Options Explained")
    st.markdown("""
    Below is a guide to the settings available in the **Machine Learning** and **Deep Learning** tabs:

    *   **Model Selection**:
        *   **Random Forest / XGBoost (ML)**: Best for estimating SoH based on a snapshot of battery metrics (Voltage, Current, Temp) from a single cycle.
        *   **LSTM / GRU (DL)**: Best for analyzing degradation trends over time. These models require a sequence of historical data (last 10 cycles) to make a prediction.
    
    *   **Input Modes**:
        *   **Single Cycle**: Allows you to manually input parameters using sliders. Ideal for quick "what-if" scenarios.
        *   **Batch Processing**: Allows you to process entire datasets to visualize the health trajectory over the battery's life.
    
    *   **Data Sources**:
        *   **Amazon S3**: Connects directly to the project's cloud storage to retrieve standardized test data.
        *   **Upload CSV**: Provides flexibility to analyze your own local battery datasets.
    """)
    
    st.info("👆 Select a tab above to start your analysis.")

# ============================================================
# Tab 2: Machine Learning Operations
# ============================================================

with tab_ml:
    st.subheader("Machine Learning Analysis (RF / XGBoost)")
    
    col_ml_1, col_ml_2 = st.columns([1, 2])
    
    with col_ml_1:
        ml_model_choice = st.radio("Select Model", ("Random Forest", "XGBoost"))
        input_mode = st.radio("Input Mode", ("Single Cycle", "Batch"))
        
        model_key = "rf" if ml_model_choice == "Random Forest" else "xgb"
        
    with col_ml_2:
        if input_mode == "Single Cycle":
            st.markdown("#### Manual Input")
            cycle = st.number_input("Cycle Number", min_value=1, value=100)
            avg_voltage = st.slider("Avg Voltage (V)", 3.0, 4.2, 3.6)
            avg_current = st.slider("Avg Current (A)", -2.0, 0.0, -1.0)
            avg_temp = st.slider("Avg Temp (°C)", 20.0, 45.0, 25.0)
            input_df = None
        else:
            st.markdown("#### Batch Input")
            source = st.radio("Data Source", ("Amazon S3", "Upload CSV"), horizontal=True)
            input_df = None
            
            if source == "Amazon S3":
                input_df = load_data_from_s3(
                    "s3aravindh973515031797",
                    "EV_Battery_Health_Source/EV_Battery_Health_Source.parquet"
                )
            else:
                file = st.file_uploader("Upload CSV", type=["csv"])
                if file:
                    input_df = pd.read_csv(file)

            if input_df is not None and "battery_name" in input_df.columns:
                battery_list = input_df["battery_name"].unique()
                selected_battery = st.selectbox("Select Battery ID", battery_list)
                input_df = input_df[input_df["battery_name"] == selected_battery].copy().sort_values("cycle")

    st.divider()
    
    if st.button("Run ML Prediction", type="primary"):
        if not models or not scaler:
            st.error("Models failed to load.")
        elif models.get(model_key):
            
            # --- Single Cycle Logic ---
            if input_mode == "Single Cycle":
                input_data = pd.DataFrame({
                    "cycle": [cycle],
                    "avg_voltage_measured": [avg_voltage],
                    "avg_current_measured": [avg_current],
                    "avg_temp_measured": [avg_temp]
                })

                input_scaled = scaler.transform(input_data)
                pred = models[model_key].predict(input_scaled)[0]

                tab_res_soh, tab_res_rul = st.tabs(["State of Health (SoH)", "Remaining Useful Life (RUL)"])

                with tab_res_soh:
                    st.metric("Predicted SoH", f"{pred:.2%}")
                    if pred < 0.70:
                        st.error("⚠️ Battery Health Critical")
                    else:
                        st.success("✅ Battery Health Acceptable")
                    
                    with st.expander("🔍 Model Explainability (SHAP)"):
                        plot_shap_explainability(models[model_key], input_scaled, input_data.columns.tolist(), ml_model_choice)
                
                with tab_res_rul:
                    rul_cycles = max(0, int((pred - 0.70) / 0.30 * 1000))
                    rul_pct = min(100.0, max(0.0, (rul_cycles / 1000) * 100))
                    c1, c2 = st.columns(2)
                    c1.metric("Estimated RUL", f"{rul_cycles} Cycles")
                    c2.metric("RUL Percentage", f"{rul_pct:.1f}%")
                    st.progress(min(1.0, max(0.0, rul_cycles / 1000)))
                    st.info("RUL estimated assuming linear decay from 100% to 70% SoH over 1000 cycles.")
                    st.info("""
                    **Methodology Note:** RUL is estimated using a physics-based heuristic assuming a linear decay 
                    from **100% to 70% SoH** (End-of-Life) over a nominal lifespan of **1000 cycles**.
                    \n$$ \\text{RUL} = \\frac{\\text{Current SoH} - 0.70}{0.30} \\times 1000 $$
                    """)

            # --- Batch Logic ---
            elif input_mode == "Batch" and input_df is not None:
                features = input_df[["cycle", "avg_voltage_measured", "avg_current_measured", "avg_temp_measured"]].values
                features_scaled = scaler.transform(features)
                preds = models[model_key].predict(features_scaled)
                input_df["Predicted SoH"] = preds
                
                tab_res_soh, tab_res_rul = st.tabs(["State of Health (SoH)", "Remaining Useful Life (RUL)"])
                
                with tab_res_soh:
                    st.metric("Latest Predicted SoH", f"{preds[-1]:.2%}")
                    st.line_chart(input_df.set_index("cycle")["Predicted SoH"], height=350, use_container_width=True)
                    
                
                with tab_res_rul:
                    input_df["Predicted RUL"] = np.maximum(0, (input_df["Predicted SoH"] - 0.70) / 0.30 * 1000)
                    latest_rul = int(input_df['Predicted RUL'].iloc[-1])
                    latest_rul_pct = min(100.0, max(0.0, (latest_rul / 1000) * 100))
                    c1, c2 = st.columns(2)
                    c1.metric("Latest Estimated RUL", f"{latest_rul} Cycles")
                    c2.metric("Latest RUL Percentage", f"{latest_rul_pct:.1f}%")
                    st.line_chart(input_df.set_index("cycle")["Predicted RUL"], height=350, use_container_width=True)

                    st.info("RUL estimated assuming linear decay from 100% to 70% SoH over 1000 cycles.")
                    st.info("""
                    **Methodology Note:** RUL is estimated using a physics-based heuristic assuming a linear decay 
                    from **100% to 70% SoH** (End-of-Life) over a nominal lifespan of **1000 cycles**.
                    \n$$ \\text{RUL} = \\frac{\\text{Current SoH} - 0.70}{0.30} \\times 1000 $$
                    """)

# ============================================================
# Tab 3: Deep Learning Operations
# ============================================================

with tab_dl:
    st.subheader("Deep Learning Analysis (LSTM / GRU)")
    
    col_dl_1, col_dl_2 = st.columns([1, 2])
    
    with col_dl_1:
        dl_model_choice = st.radio("Select Model", ("LSTM", "GRU"))
        model_key = "lstm" if dl_model_choice == "LSTM" else "gru"
        
        source = st.radio("Data Source", ("Amazon S3", "Upload CSV"), horizontal=True, key="dl_source")
        
    with col_dl_2:
        input_df = None
        if source == "Amazon S3":
            input_df = load_data_from_s3(
                "s3aravindh973515031797",
                "EV_Battery_Health_Source/EV_Battery_Health_Source.parquet"
            )
        else:
            file = st.file_uploader("Upload CSV", type=["csv"], key="dl_file")
            if file:
                input_df = pd.read_csv(file)
        
        if input_df is not None and "battery_name" in input_df.columns:
            battery_list = input_df["battery_name"].unique()
            selected_battery = st.selectbox("Select Battery ID", battery_list, key="dl_battery")
            input_df = input_df[input_df["battery_name"] == selected_battery].copy().sort_values("cycle")
            
    st.divider()
    
    if st.button("Run DL Prediction", type="primary"):
        if not models or not scaler:
            st.error("Models failed to load.")
        elif input_df is not None and models.get(model_key):
            
            features = input_df[["cycle", "avg_voltage_measured", "avg_current_measured", "avg_temp_measured"]].values
            features_scaled = scaler.transform(features)
            X_seq = create_sequences(features_scaled, 10)
            
            if len(X_seq) == 0:
                st.error("Not enough data points to create sequences (Minimum 10 cycles required).")
            else:
                preds = models[model_key].predict(X_seq).flatten()
                cycles = input_df["cycle"].values[9:]
                
                tab_res_soh, tab_res_rul = st.tabs(["State of Health (SoH)", "Remaining Useful Life (RUL)"])
                
                with tab_res_soh:
                    
                    st.metric("Latest Predicted SoH", f"{preds[-1]:.2%}")
                    
                    chart_df = pd.DataFrame({"Cycle": cycles, "Predicted SoH": preds}).set_index("Cycle")
                    st.line_chart(chart_df, height=350, use_container_width=True)
                    
                    
            
                with tab_res_rul:
                    rul_preds = np.maximum(0, (preds - 0.70) / 0.30 * 1000)
                    rul_df = pd.DataFrame({"Cycle": cycles, "Predicted RUL": rul_preds}).set_index("Cycle")
                    
                    latest_rul = int(rul_preds[-1])
                    c1, c2 = st.columns(2)
                    c1.metric("Latest Estimated RUL", f"{latest_rul} Cycles")
                    c2.metric("Latest RUL Percentage", f"{min(100.0, max(0.0, (latest_rul / 1000) * 100)):.1f}%")

                    st.line_chart(rul_df, height=350, use_container_width=True)

                    st.info("RUL estimated assuming linear decay from 100% to 70% SoH over 1000 cycles.")
                    st.info("""
                    **Methodology Note:** RUL is estimated using a physics-based heuristic assuming a linear decay 
                    from **100% to 70% SoH** (End-of-Life) over a nominal lifespan of **1000 cycles**.
                    \n$$ \\text{RUL} = \\frac{\\text{Current SoH} - 0.70}{0.30} \\times 1000 $$
                    """)
