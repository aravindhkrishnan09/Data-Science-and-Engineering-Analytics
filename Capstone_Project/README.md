# Capstone Project: EV Battery Health Prediction

## 1. Objective

To develop an **Intelligent Predictive Maintenance System** that estimates the **State of Health (SoH)** and forecasts the **Remaining Useful Life (RUL)** of Lithium-ion batteries using machine learning and deep learning techniques.

**Primary Goals**:
- Predict battery SoH within ±5% absolute error on unseen data.
- Estimate RUL with >90% recall for failure detection.
- Build a scalable, deployable ML pipeline.
- Deploy as a web application on free hosting platforms.

**Technical Requirements**:
- Implement supervised learning (Random Forest, XGBoost, LSTM/GRU).
- Apply unsupervised learning (K-means clustering for degradation patterns).
- Integrate reinforcement learning (Q-learning for charging optimization).
- Achieve engineering-grade data quality (<5% missing data).

[EV Battery Health Prediction Website](https://aravindhgk-z4pcfbk.gamma.site/)

## 2. Source Data

### Primary Dataset: NASA PCoE Battery Data Set
- **Dataset Name**: NASA Prognostics Center of Excellence (PCoE) Battery Data Set
- **Battery Types**: Li-ion 18650 batteries
- **Battery IDs**: B0005, B0006, B0007, B0018
- **Format**: Raw MATLAB (`.mat`) files containing nested hierarchical structures.
- **Location**: `Capstone_Project/source_data/`

### Data Structure
Each `.mat` file contains:
- **Cycle-level data**: Charge, discharge, and impedance operations.
- **Measured Parameters**:
  - `Voltage_measured`: Battery terminal voltage (Volts)
  - `Current_measured`: Battery output current (Amps)
  - `Temperature_measured`: Battery temperature (°C)
  - `Capacity`: Battery capacity (Ahr) for discharge cycles
  - `Current_charge`: Current at charger/load (Amps)
  - `Voltage_charge`: Voltage at charger/load (Volts)
  - `Time`: Time vector for each cycle (seconds)
  - `Current_load`: Current at load (Amps) (for discharge cycles)
  - `Voltage_load`: Voltage at load (Volts) (for discharge cycles)
  - `Battery_current`, `Battery_impedance`, `Current_ratio`, `Rct`, `Re`, `Rectified_Impedance`, `Sense_current`: Parameters for impedance cycles.
- **Operational Profiles**:
  - **Charge**: Constant Current (CC) at 1.5A until 4.2V, then Constant Voltage (CV) until 20mA.
  - **Discharge**: Constant Current (CC) at 2A until voltage thresholds (2.7V, 2.5V, 2.2V, 2.5V for batteries 5, 6, 7, 18 respectively).

### Processed Data Files
- `Flattened_b0005.csv`, `Flattened_b0006.csv`, `Flattened_b0007.csv`, `Flattened_b0018.csv`: Flattened hierarchical data from the `.mat` files.
- `final_df_export.csv`: Complete aggregated dataset used for modeling.

### Model Artifacts
- `lstm_soh_model.keras`, `gru_soh_model.keras`: Deep learning models (Keras/TensorFlow).
- `rf_soh_model.joblib`, `xgb_model.joblib`: Serialized machine learning models.
- `soh_scaler.joblib`: Feature scaler for preprocessing.

## 3. Key Operations & Tasks

### 1. Data Engineering & Pipeline
- **Extraction**: Parsed complex nested structures from raw `.mat` files using `scipy.io.loadmat`.
- **Transformation**:
  - Flattened hierarchical data into structured Pandas DataFrames.
  - Aggregated cycle-level metrics (mean voltage, current, temperature).
  - Derived physics-based features (Voltage Drop `Delta_V`, Temperature Increase `Delta_T`, Discharge Time).
  - Engineered features like State of Health (SoH).
- **Storage**:
  - Implemented an automated pipeline to upload processed data to **AWS S3** using `boto3`.
  - Data stored in both CSV and Parquet formats for efficient access.
  - Cloud integration for scalable deployment.

### 2. Exploratory Data Analysis (EDA) & Preprocessing
- **Degradation Analysis**:
  - Visualized capacity fade over cycle life to identify degradation patterns.
  - Identified "knee points" where rapid degradation begins.
- **Correlation Analysis**:
  - Generated heatmaps to find relationships between voltage, current, temperature, and capacity.
- **Outlier Detection**:
  - Applied **Z-Score** analysis to identify and handle anomalies in sensor data.
  - Implemented robust data quality checks.
- **Feature Engineering**:
  - Derived time-series features (rolling averages, trends).
  - Created cycle-based aggregations for model input.
  - Engineered domain-specific metrics.

### 3. Predictive Modeling

#### Supervised Learning (SoH Estimation)
- **Random Forest Regressor**: Achieved high accuracy (R² ~0.94) in mapping cycle parameters to SoH.
- **XGBoost**: Utilized gradient boosting for robust baseline performance.
  - R²: 0.9276, MAE: 1.76%.
  - Strong alternative to Random Forest.

#### Deep Learning (Time-Series Forecasting)
- **LSTM (Long Short-Term Memory)**: Captured long-term dependencies in degradation trends.
  - Requires sequence of 10 cycles for prediction.
  - Performance: R² 0.29–0.81 (varies with architecture tuning).
- **GRU (Gated Recurrent Unit)**: An efficient alternative to LSTM, which proved to be the most effective time-series model with a low Mean Absolute Error (MAE) of **1.50%**.
  - Suitable for production deployment.

#### Unsupervised Learning
- **K-Means Clustering**:
  - Segmented battery life into distinct aging stages based on operational data.
  - Identified degradation patterns and failure modes.

#### Reinforcement Learning
- **Q-Learning**:
  - Developed an agent to optimize charging profiles (current/voltage control).
  - Aims to extend battery life through intelligent charging strategies.

### 4. Model Explainability
 - **SHAP (SHapley Additive exPlanations)**:
  - Used to interpret model decisions.
  - Quantified the impact of features like voltage and temperature on SoH predictions.
  - Enhanced model transparency for engineering decisions.

### 4. Deployment
- **Web Application**:
  - Built an interactive dashboard using **Streamlit** (`EV_Battery_Health_Prediction_App.py`, `EV_Battery_Health_Prediction_App_v2.py`).
  - Features:
    - Real-time SoH and RUL predictions.
    - Support for single-cycle (RF/XGBoost) and sequence-based (LSTM/GRU) predictions.
    - Interactive visualizations and model comparison.
    - CSV file upload and AWS S3 integration.

## 4. Project Files

### Core Analysis Notebook
- `Capstone_Project/jupyter_notebooks/EV_Battery_Health_Prediction.ipynb`:
  - Comprehensive notebook with the entire workflow from data extraction to modeling.

### Deployment Applications
- `Capstone_Project/jupyter_notebooks/EV_Battery_Health_Prediction_App.py`: Streamlit application for model deployment.
- `Capstone_Project/jupyter_notebooks/EV_Battery_Health_Prediction_App_v2.py`: Enhanced version with AWS S3 integration.
```
### Documentation
- `Capstone_Project/jupyter_notebooks/EV_Battery_Health_Prediction.html`: Exported HTML documentation of the analysis.
- `Capstone_Project/jupyter_notebooks/EV_Battery_Health_Prediction.docx`: Project report.

### Project Requirements
- `Capstone_Project/requirement/EV_Battery_SoH_RUL_Capstone_Guide.pdf`: Project guidelines.
- `Capstone_Project/requirement/Intelligent Predictive Maintenance and Battery Health Forecasting System for Electric Vehicles.docx`: Detailed project specification.
