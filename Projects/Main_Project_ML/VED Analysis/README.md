# Vehicle Energy Dataset (VED) Analysis & Machine Learning

## Overview

This project analyzes the **Vehicle Energy Dataset (VED)**, a comprehensive collection of real-world driving data from Ann Arbor, Michigan. The dataset encompasses four distinct vehicle types: Internal Combustion Engine (ICE), Hybrid Electric Vehicle (HEV), Plug-in Hybrid Electric Vehicle (PHEV), and Battery Electric Vehicle (EV).

## Objective

The primary objective is to develop and evaluate machine learning models capable of predicting **Fuel Consumption Rate (FCR)** for conventional and hybrid vehicles, and **HV Battery Power** for electric vehicles, using standard on-board diagnostics (OBD-II) and vehicle telemetry data.

## Challenge

The core challenge lies in integrating static vehicle specifications with high-frequency dynamic sensor data. This involves managing missing values, standardizing data types, harmonizing column names, and deriving physics-based features for accurate energy estimation.

## Analysis Pipeline

The project follows a structured analysis pipeline:

1.  **Static Data Integration**: Loading and merging static vehicle specifications (e.g., weight, engine configuration, transmission type, drive wheels).
2.  **Dynamic Data Aggregation**: Processing high-frequency time-series data, including vehicle speed, GPS coordinates, engine RPM, and environmental sensor readings.
3.  **Feature Engineering**: Calculating derived features such as distance traveled, Fuel Consumption Rate (FCR) using VED algorithms, and categorizing outside air temperature (OAT) to understand environmental impacts.
4.  **Exploratory Data Analysis (EDA)**: Identifying correlations, visualizing trip patterns, and analyzing energy consumption trends across different vehicle types.
5.  **Predictive Modeling**: Building and evaluating linear regression models for each powertrain architecture with iterative feature refinement.
6.  **Clustering Analysis**: Applying K-means clustering to categorize vehicle behavior patterns and validate energy consumption segments.

## Data Sources

The project utilizes data stored in an AWS S3 bucket (`s3aravindh973515031797`).

*   **Static Data**:
    *   `VED_Static_Data_ICE&HEV.xlsx`
    *   `VED_Static_Data_PHEV&EV.xlsx`
    *   Processed and stored as `df_ICE_HEV.parquet`, `df_PHEV_EV.parquet`, `df_static.parquet` in S3.
*   **Dynamic Data**:
    *   Weekly CSV files from `VED_DynamicData_Part1/` (and `VED_DynamicData_Part2/`).
    *   Strategically sampled (50% random selection) to manage computational constraints, resulting in over 5.1 million time-series records.
    *   Stored as `df_dynamic_sample.parquet` in S3.
*   **Combined Data**: The merged static and dynamic data is stored as `df_VED.parquet` and `df_combined.parquet` in S3.

## Data Preprocessing & Feature Engineering

Key steps included:

*   **Missing Value Handling**: Replaced 'NO DATA' strings with NaN. Imputation strategies varied by vehicle type (e.g., using mean values for ICE/HEV, or 0 for EV-specific engine parameters).
*   **Data Type Conversion**: Converted `DayNum` to `DateTime`, `Date`, and `Time` columns.
*   **Distance Calculation**: `Distance[km]` was calculated using `Vehicle Speed[km/h]` and `Timestamp(ms)`.
*   **Fuel Consumption Rate (FCR) Estimation**: Implemented a hierarchical algorithm from the IEEE VED paper, utilizing `MAF[g/sec]`, `Absolute Load[%]`, `Fuel Trim Banks`, `Engine RPM[RPM]`, and `Displacement_L`.
*   **HV Battery Power Calculation**: `HV Battery Power[Watts]` was derived from `HV Battery Voltage[V]` and `HV Battery Current[A]`.
*   **Outside Air Temperature Categorization**: `OAT[DegC]` was binned into categories like 'Extremely Cold', 'Cold', 'Cool', 'Mild', 'Warm', 'Hot'.

## Exploratory Data Analysis (EDA) Insights

*   **Average Distance Traveled**: PHEV vehicles showed the longest average trip distances (5.9 km), while EVs had the shortest (4.35 km). ICE and HEV clustered between 4.4-5.0 km.
*   **HV Battery Voltage Over Time**: Exhibited seasonal patterns, with a notable decline during winter months and stabilization around 300V in temperate periods.
*   **Fuel Consumption Rate Over Time**: Showed an increasing trend in winter/spring, peaking at 1.1 L/hr in March, correlating with cold-start enrichment.
*   **Trip Density Heatmaps**:
    *   **ICE**: Broad geographic dispersion.
    *   **HEV**: Corridor-focused with hotspots.
    *   **PHEV**: Clustered near amenities.
    *   **EV**: Tight high-density clusters, indicating charging infrastructure dependency.
*   **Energy Consumption Comparison (Distance vs FCR/HV Power)**:
    *   **ICE & HEV**: ICE generally showed higher FCR than HEV, especially at shorter distances. HEV demonstrated better fuel efficiency.
    *   **EV & PHEV**: PHEV exhibited both positive and negative HV Battery Power (regenerative braking), while EV power usage was more concentrated.

## Machine Learning Models

### Supervised Learning (Linear Regression)

Linear Regression models were developed to predict FCR and HV Battery Power for different vehicle types.

*   **ICE FCR Prediction**:
    *   **Features**: `Vehicle Speed[km/h]`, `Distance[km]`, `Generalized_Weight`, `MAF[g/sec]`, `Absolute Load[%]`, `Long Term Fuel Trim Bank 1[%]`.
    *   **Performance**: R² score of 0.689 (from PDF). The model predicts FCR reasonably well, but not as accurately as HEVs. SHAP analysis was used for feature importance.
*   **HEV FCR Prediction**:
    *   **Features**: `MAF[g/sec]`.
    *   **Performance**: R² score of 0.944. High accuracy, indicating `MAF[g/sec]` is a strong predictor.
*   **EV Battery Power Prediction**:
    *   **Features**: `Air Conditioning Power[Watts]`, `Heater Power[Watts]`, `Vehicle Speed[km/h]`.
    *   **Performance**: R² score of -0.150. The model failed to capture the relationship, suggesting the need for better features or a different modeling approach due to the inherent nonlinearity of EV power consumption.
*   **PHEV Battery Power Prediction**:
    *   **Features**: `Engine RPM[RPM]`, `Air Conditioning Power[Watts]`, `Vehicle Speed[km/h]`, `OAT[DegC]`.
    *   **Performance**: R² score of 0.712. Effective prediction, capturing the relationship well.

### Unsupervised Learning (K-Means Clustering)

K-Means clustering was applied to identify natural groupings in vehicle behavior patterns.

*   **Elbow Method**: Used to determine the optimal number of clusters (typically 2-3).
*   **Speed vs FCR by Vehicle Type**:
    *   **Features**: `Vehicle Type`, `Vehicle Speed[km/h]`, `FCR`.
    *   **Results**: Distinct clusters (2-3) were observed for each vehicle type, indicating different driving or consumption patterns. EV FCR was near zero, while PHEV showed a wider spread.
*   **Speed vs Battery Power by Vehicle Type**:
    *   **Features**: `Vehicle Type`, `Vehicle Speed[km/h]`, `HV Battery Power[Watts]`.
    *   **Results**: ICE and HEV showed a single cluster (negligible battery power), while PHEV and EV exhibited multiple clusters reflecting diverse battery power usage.
*   **Outside Air Temperature vs FCR**:
    *   **Features**: `OAT[DegC]`, `FCR`.
    *   **Results**: 5 distinct clusters were identified, with higher FCR variability at very low temperatures.
*   **Outside Air Temperature vs Battery Power**:
    *   **Features**: `OAT[DegC]`, `HV Battery Power[Watts]`.
    *   **Results**: 5 clusters, showing varying battery output and regenerative braking scenarios across temperature bands.

## Key Findings and Conclusions

*   **Feasibility Confirmed**: Predicting energy consumption from OBD-II data is highly viable for HEV and ICE vehicles with comprehensive powertrain sensors.
*   **Model Robustness**: HEVs achieved 94% accuracy using single-feature models, demonstrating high predictability due to sophisticated power management algorithms.
*   **Feature Importance**: Mass Air Flow (MAF) and Absolute Load emerged as critical predictors for fuel consumption. For battery power, simple kinematic variables were insufficient, highlighting the need for road gradient and high-frequency acceleration data.
*   **Temporal Analysis**: Time-based data splitting consistently outperformed random trip splitting, emphasizing the importance of temporal continuity.
*   **EV Challenges**: EV power consumption is inherently nonlinear, dominated by transient dynamics that linear regression struggles to capture from aggregate speed measurements. Future work should explore gradient boosting or LSTM architectures.
*   **Clustering Validation**: K-Means clustering effectively segments vehicle operational patterns by type, revealing characteristic speed and fuel/energy consumption behaviors for each vehicle class.

## How to Run the Streamlit Application

The Streamlit application `VED_ML_Streamlit_v2.py` provides an interactive interface for exploring the analysis and models.

1.  **Environment Setup**: Ensure you have Python installed and the necessary libraries (listed in the `VED_ML_Streamlit_v2.py` imports) are installed. You can typically install them using `pip install -r requirements.txt` if a `requirements.txt` file is available, or `pip install streamlit pandas numpy boto3 scikit-learn matplotlib python-dotenv shap`.
2.  **AWS Credentials**: The application requires AWS S3 credentials to access the data.
   *   If running locally, create a `.env` file in the same directory as the script with your `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_DEFAULT_REGION`.
   *   If deploying to Streamlit Cloud, configure these as Streamlit secrets.
3.  **Run the Application**: Navigate to the directory containing `VED_ML_Streamlit_v2.py` in your terminal and run:
   ```bash
   streamlit run VED_ML_Streamlit_v2.py
   ```
   This will open the application in your web browser.