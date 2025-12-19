import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt
import boto3
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="EV Battery Health Predictor",
    page_icon="🔋",
    layout="wide"
)

# --- Title and Description ---
st.title("🔋 EV Battery Health Prediction System")
st.markdown("""
This application predicts the **State of Health (SoH)** of an EV battery.
- **Single Point**: Predict SoH for a specific cycle condition (RF/XGBoost).
- **Full History**: Analyze degradation trends over time (LSTM/GRU).
""")

# --- Load Models and Scaler ---
@st.cache_resource
def load_all_models():
    """Loads all models and the scaler from disk."""
    models = {}
    model_path = "." 

    # Load ML models
    try:
        models['rf'] = joblib.load(os.path.join(model_path, "rf_soh_model.joblib"))
        models['xgb'] = joblib.load(os.path.join(model_path, "xgb_model.joblib"))
    except FileNotFoundError:
        st.warning("ML models (RF/XGB) not found. Please run the notebook to generate them.")

    # Load DL models
    try:
        models['lstm'] = load_model(os.path.join(model_path, "lstm_soh_model.h5"), compile=False)
        models['gru'] = load_model(os.path.join(model_path, "gru_soh_model.h5"), compile=False)
    except (FileNotFoundError, OSError):
        st.warning("DL models (LSTM/GRU) not found. Please run the notebook to generate them.")

    # Load Scaler
    try:
        scaler = joblib.load(os.path.join(model_path, "soh_scaler.joblib"))
    except FileNotFoundError:
        st.error("Scaler file (soh_scaler.joblib) not found. Cannot make predictions.")
        return None, None

    return models, scaler

models, scaler = load_all_models()

# --- S3 Data Loading ---
@st.cache_data
def load_data_from_s3(bucket, key):
    """Loads and aggregates data from S3."""
    try:
        s3 = boto3.client("s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_DEFAULT_REGION")
        )
        obj = s3.get_object(Bucket=bucket, Key=key)
        raw_df = pd.read_parquet(BytesIO(obj['Body'].read()))
        
        # Aggregate per cycle (as per notebook logic for features)
        # Filter for discharge cycles
        df_discharge = raw_df[raw_df['type'] == 'discharge']
        
        # Group and aggregate
        df_agg = df_discharge.groupby(['battery_name', 'cycle']).agg({
            'Voltage_measured': 'mean',
            'Current_measured': 'mean',
            'Temperature_measured': 'mean'
        }).reset_index()
        
        # Rename to match model features
        df_agg = df_agg.rename(columns={
            'Voltage_measured': 'avg_voltage_measured',
            'Current_measured': 'avg_current_measured',
            'Temperature_measured': 'avg_temp_measured'
        })
        
        return df_agg
    except Exception as e:
        st.error(f"Error loading data from S3: {e}")
        return None

# --- Helper Function for Sequences ---
def create_sequences(data, time_steps=10):
    """Creates sliding window sequences from the data."""
    Xs = []
    # Ensure we have enough data
    if len(data) < time_steps:
        return np.array([])
    
    for i in range(len(data) - time_steps + 1):
        Xs.append(data[i:(i + time_steps)])
    return np.array(Xs)

# --- Sidebar for User Input ---
st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ("Random Forest", "XGBoost", "LSTM", "GRU")
)

is_deep_learning = model_choice in ["LSTM", "GRU"]

# --- Input Handling ---
input_df = None
use_demo_data = False

if is_deep_learning:
    st.sidebar.subheader("Data Input")
    data_source = st.sidebar.radio("Source", ("Upload CSV", "Use Demo Data (B0005)", "Amazon S3"))
    
    if data_source == "Amazon S3":
        bucket = 's3aravindh973515031797'
        key = 'EV_Battery_Health_Source/EV_Battery_Health_Source.parquet'
        
        df_s3 = load_data_from_s3(bucket, key)
        
        if df_s3 is not None:
            # Battery Selection
            batteries = sorted(df_s3['battery_name'].unique())
            
            select_mode = st.sidebar.radio("Select Batteries:", ("Use All", "Select Individually"))
            
            if select_mode == "Select Individually":
                selected_batteries = st.sidebar.multiselect("Choose batteries:", batteries, default=batteries[:1])
            else:
                selected_batteries = batteries
            
            # Filter by battery
            if selected_batteries:
                input_df = df_s3[df_s3['battery_name'].isin(selected_batteries)].copy()
                
                # Cycle Limiting
                if not input_df.empty:
                    max_cycle = int(input_df['cycle'].max())
                    limit_cycles = st.sidebar.checkbox("Limit Cycles")
                    if limit_cycles:
                        cycle_limit = st.sidebar.slider("Max Cycle", 1, max_cycle, max_cycle)
                        input_df = input_df[input_df['cycle'] <= cycle_limit]
                    
                    st.info(f"Selected {len(selected_batteries)} batteries with {len(input_df)} total cycles.")
            else:
                st.warning("Please select at least one battery.")

    elif data_source == "Use Demo Data (B0005)":
        demo_path = "final_df_export.csv"
        if os.path.exists(demo_path):
            # Load and filter for just one battery to make it a clean time series
            full_df = pd.read_csv(demo_path)
            input_df = full_df[full_df['battery_name'] == 'b0005'].copy()
            st.success(f"Loaded demo data for Battery B0005 ({len(input_df)} cycles)")
        else:
            st.error("Demo file 'final_df_export.csv' not found.")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV (History)", type=["csv"])
        if uploaded_file:
            input_df = pd.read_csv(uploaded_file)

else:
    # Manual input for ML models
    st.sidebar.subheader("Single Cycle Parameters")
    cycle = st.sidebar.number_input("Cycle Number", min_value=1, value=100)
    avg_voltage = st.sidebar.slider("Avg Voltage (V)", 3.0, 4.2, 3.6)
    avg_current = st.sidebar.slider("Avg Current (A)", -2.0, 0.0, -1.0) # Discharge is usually negative
    avg_temp = st.sidebar.slider("Avg Temp (°C)", 20.0, 45.0, 25.0)

# --- Main Execution ---
if st.button("Run Prediction"):
    if not models or not scaler:
        st.error("Models or scaler not loaded.")
        
    else:
        model_key = model_choice.lower().replace(" ", "")
        if model_key.startswith("random"): model_key = "rf"
        
        # 1. Machine Learning (Single Point)
        if not is_deep_learning:
            if models.get(model_key):
                # Prepare single row dataframe
                input_data = pd.DataFrame({
                    'cycle': [cycle],
                    'avg_voltage_measured': [avg_voltage],
                    'avg_current_measured': [avg_current],
                    'avg_temp_measured': [avg_temp]
                })
                
                # Scale
                input_scaled = scaler.transform(input_data)
                
                # Predict
                pred = models[model_key].predict(input_scaled)[0]
                
                st.subheader(f"Result ({model_choice})")
                st.metric("Predicted SoH", f"{pred:.2%}")
                
                # Gauge chart logic could go here
                if pred < 0.70:
                    st.error("⚠️ Battery Health Critical (Replace)")
                else:
                    st.success("✅ Battery Health Good")
            else:
                st.error(f"Model {model_choice} not loaded.")

        # 2. Deep Learning (Time Series / Full History)
        else:
            if input_df is not None:
                if models.get(model_key):
                    required_cols = ['cycle', 'avg_voltage_measured', 'avg_current_measured', 'avg_temp_measured']
                    
                    # Validation
                    if not all(col in input_df.columns for col in required_cols):
                        st.error(f"CSV missing columns. Required: {required_cols}")
                    else:
                        try:
                            # Prepare features
                            features = input_df[required_cols].values
                            
                            # Scale
                            features_scaled = scaler.transform(features)
                            
                            # Create Sequences (Sliding Window)
                            TIME_STEPS = 10
                            X_seq = create_sequences(features_scaled, TIME_STEPS)
                            
                            if len(X_seq) == 0:
                                st.error(f"Not enough data. Need at least {TIME_STEPS} cycles.")
                            else:
                                # Batch Prediction
                                with st.spinner(f"Running {model_choice} on {len(X_seq)} sequences..."):
                                    predictions = models[model_key].predict(X_seq).flatten()
                                
                                # Align predictions with cycles
                                # Predictions start from cycle index `TIME_STEPS`
                                pred_cycles = input_df['cycle'].values[TIME_STEPS-1:]
                                
                                # Visualization
                                st.subheader(f"Degradation Trend ({model_choice})")
                                
                                chart_data = pd.DataFrame({
                                    'Cycle': pred_cycles,
                                    'Predicted SoH': predictions
                                })
                                
                                # If actual SoH is present (e.g. in demo data), plot it too
                                if 'capacity_Ah' in input_df.columns:
                                    # Calculate actual SoH (assuming initial capacity is approx 2.0Ah or max of dataset)
                                    # For demo data, we can try to derive it or just plot capacity
                                    max_cap = input_df['capacity_Ah'].max()
                                    actual_soh = input_df['capacity_Ah'].values[TIME_STEPS-1:] / max_cap
                                    chart_data['Actual SoH'] = actual_soh
                                
                                st.line_chart(chart_data.set_index('Cycle'))
                                
                                # Latest Status
                                latest_soh = predictions[-1]
                                st.metric("Latest Predicted SoH", f"{latest_soh:.2%}")
                                
                        except Exception as e:
                            st.error(f"Error during processing: {e}")
                else:
                    st.error(f"Model {model_choice} not loaded.")
            else:
                st.warning("Please upload data or select demo data to proceed.")
