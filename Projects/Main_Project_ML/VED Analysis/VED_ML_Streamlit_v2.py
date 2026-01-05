import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import warnings
import shap
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')
import os
import boto3
from io import BytesIO
import random
from io import StringIO
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

st.set_page_config(page_title="VED ML Data Modelling", layout="centered", initial_sidebar_state="expanded")
st.title('VED ML Data Modelling')

st.markdown("""
    <style>
        /* Custom styling for tab container with horizontal scroll */
        .stTabs [role="tablist"] {
            overflow-x: auto;
            white-space: nowrap;
            scrollbar-width: thin;
            display: flex;
            gap: 4px;
        }

        /* Webkit scrollbar styling for better visual appearance */
        .stTabs [role="tablist"]::-webkit-scrollbar {
            height: 8px;
            background-color: #f0f0f0;
        }

        .stTabs [role="tablist"]::-webkit-scrollbar-track {
            background-color: #f0f0f0;
            border-radius: 4px;
        }

        .stTabs [role="tablist"]::-webkit-scrollbar-thumb {
            background-color: #888;
            border-radius: 4px;
            transition: background-color 0.3s ease;
        }

        .stTabs [role="tablist"]::-webkit-scrollbar-thumb:hover {
            background-color: #555;
        }

        /* Individual tab styling for better spacing and appearance */
        .stTabs [role="tab"] {
            flex-shrink: 0;
            padding: 12px 20px;
            margin-right: 2px;
            border-radius: 6px 6px 0 0;
            transition: all 0.3s ease;
        }

        /* Hover effect for tabs */
        .stTabs [role="tab"]:hover {
            background-color: rgba(255, 255, 255, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Load environment variables only if running locally
if not st.secrets:  # st.secrets is empty in local
    load_dotenv()

# Use st.secrets if on Streamlit Cloud, else use os.getenv
AWS_ACCESS_KEY_ID = st.secrets.get("AWS_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID"))
AWS_SECRET_ACCESS_KEY = st.secrets.get("AWS_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY"))
AWS_DEFAULT_REGION = st.secrets.get("AWS_DEFAULT_REGION", os.getenv("AWS_DEFAULT_REGION"))

s3 = boto3.client("s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION
    )

bucket_name = 's3aravindh973515031797'
ICE_HEV = 'Cleaned up VED Source Data/df_ICE_HEV.parquet'
PHEV_EV = 'Cleaned up VED Source Data/df_PHEV_EV.parquet'
df_static = 'Cleaned up VED Source Data/df_static.parquet'
df_dynamic_sample = 'Cleaned up VED Source Data/df_dynamic_sample.parquet'
df = 'Cleaned up VED Source Data/df_VED.parquet'
df_combined = 'Cleaned up VED Source Data/df_combined.parquet'

@st.cache_data
def read_parquet_from_s3(bucket_name, object_key):
        """
        Reads a Parquet file from an AWS S3 bucket using the global s3 client.

        Args:
            bucket_name: Name of the S3 bucket.
            object_key: Key (path) to the Parquet file in the S3 bucket.

        Returns:
            DataFrame containing the Parquet data.
        """
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        file_content = response['Body'].read()
        df = pd.read_parquet(BytesIO(file_content))
        return df

tabs = st.tabs(["ICE, HEV, EV, and PHEV Analysis",
                 "Supervised Learning",
                 "Unsupervised Learning"])

with tabs[0]:
    st.markdown("""
    ### ICE, HEV, EV, and PHEV Analysis: Distance vs FCR and HV Battery Power

    The tables below show a comparison of key metrics for different vehicle types:
    - **ICE & HEV**: Internal Combustion Engine and Hybrid Electric Vehicles
    - **EV & PHEV**: Electric Vehicles and Plug-in Hybrid Electric Vehicles

    The data is grouped by trip, time and vehicle type, and displays the top records for each group.
    """)

    df = read_parquet_from_s3(bucket_name, df)

    # Grouping and aggregating the data
    df_SOC = df.groupby(['OAT_Category', 'Vehicle Type'])[
        ['HV Battery Power[Watts]', 'Air Conditioning Power[Watts]', 'Heater Power[Watts]', 'HV Battery SOC[%]', 'FCR']
    ].mean().reset_index().sort_values(by='HV Battery SOC[%]', ascending=False)

    df_EC_trip = df.groupby(['Trip', 'Vehicle Type'])[
        [
            'Latitude[deg]',
            'Longitude[deg]',
            'Air Conditioning Power[Watts]',
            'Heater Power[Watts]',
            'Vehicle Speed[km/h]',
            'Distance[km]',
            'Engine RPM[RPM]',
            'OAT[DegC]',
            'Generalized_Weight',
            'FCR',
            'HV Battery Power[Watts]',
            'MAF[g/sec]',
            'Absolute Load[%]',
            'Short Term Fuel Trim Bank 1[%]',
            'Short Term Fuel Trim Bank 2[%]',
            'Long Term Fuel Trim Bank 1[%]',
            'Long Term Fuel Trim Bank 2[%]'
        ]
    ].mean().reset_index().sort_values(by=['FCR', 'HV Battery Power[Watts]'], ascending=False)

    df_EC_time = (
        df.groupby([df['Date'].dt.to_period('M'), 'Vehicle Type'])[
            [
                'Latitude[deg]',
                'Longitude[deg]',
                'Air Conditioning Power[Watts]',
                'Heater Power[Watts]',
                'Vehicle Speed[km/h]',
                'Distance[km]',
                'Engine RPM[RPM]',
                'OAT[DegC]',
                'Generalized_Weight',
                'FCR',
                'HV Battery Power[Watts]',
                'MAF[g/sec]',
                'Absolute Load[%]',
                'Short Term Fuel Trim Bank 1[%]',
                'Short Term Fuel Trim Bank 2[%]',
                'Long Term Fuel Trim Bank 1[%]',
                'Long Term Fuel Trim Bank 2[%]'
            ]
        ]
        .mean()
        .reset_index()
        .sort_values(by=['Date', 'FCR', 'HV Battery Power[Watts]'])
    )

    df_EC_time = (
        df.groupby([df['Date'].dt.to_period('D'), 'Vehicle Type'])[
            [
                'Latitude[deg]',
                'Longitude[deg]',
                'Air Conditioning Power[Watts]',
                'Heater Power[Watts]',
                'Vehicle Speed[km/h]',
                'Distance[km]',
                'Engine RPM[RPM]',
                'OAT[DegC]',
                'Generalized_Weight',
                'FCR',
                'HV Battery Power[Watts]',
                'MAF[g/sec]',
                'Absolute Load[%]',
                'Short Term Fuel Trim Bank 1[%]',
                'Short Term Fuel Trim Bank 2[%]',
                'Long Term Fuel Trim Bank 1[%]',
                'Long Term Fuel Trim Bank 2[%]'
            ]
        ]
        .mean()
        .reset_index()
        .sort_values(by=['Date', 'FCR', 'HV Battery Power[Watts]'])
    )

    df_la_lo = (
        df.groupby(['Latitude[deg]', 'Longitude[deg]', 'Vehicle Type'])[
            [
                'Vehicle Speed[km/h]',
                'Absolute Load[%]',
                'Engine RPM[RPM]',
                'OAT[DegC]',
                'Generalized_Weight',
                'FCR',
                'HV Battery Power[Watts]'
            ]
        ]
        .mean()
        .reset_index()
        .sort_values(by=['FCR', 'HV Battery Power[Watts]'], ascending=False)
    )

    df_EC_trip_ICE_HEV = df_EC_trip[df_EC_trip['Vehicle Type'].isin(['ICE', 'HEV'])]
    df_EC_trip_EV_PHEV = df_EC_trip[df_EC_trip['Vehicle Type'].isin(['EV', 'PHEV'])]

    df_EC_time_ICE_HEV = df_EC_time[df_EC_time['Vehicle Type'].isin(['ICE','HEV'])]
    df_EC_time_EV_PHEV = df_EC_time[df_EC_time['Vehicle Type'].isin(['EV','PHEV'])]

    st.subheader("ICE & HEV Vehicles")
    st.dataframe(df_EC_trip_ICE_HEV.head(10), use_container_width=True)

    st.subheader("EV & PHEV Vehicles")
    st.dataframe(df_EC_trip_EV_PHEV.head(10), use_container_width=True)

    st.markdown("""
    **Comments:**  
    - The tables above provide a summary of distance, fuel consumption rate (FCR), and HV battery power for ICE, HEV, EV, and PHEV vehicle types.
    - This allows for a direct comparison of energy consumption and operational characteristics across different powertrain technologies.
    """)

    # Description:
    st.markdown("""
    #### Energy Consumption Comparison by Vehicle Type over Trip

    The following scatter plots visualize the relationship between trip distance and energy consumption metrics for different vehicle types:
    - **Left Plot:** Distance vs Fuel Consumption Rate (FCR) for Internal Combustion Engine (ICE) and Hybrid Electric Vehicles (HEV).
    - **Right Plot:** Distance vs HV Battery Power for Electric Vehicles (EV) and Plug-in Hybrid Electric Vehicles (PHEV).

    These visualizations help compare operational efficiency and energy usage patterns across various powertrain technologies.
    """)

    # Create a 1x2 subplot for the two comparisons
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=False)

    # Subplot 1: ICE vs HEV (Distance vs FCR)
    ax1 = axes[0]
    ax1.scatter(
        df_EC_trip_ICE_HEV[df_EC_trip_ICE_HEV['Vehicle Type'] == 'ICE']['Distance[km]'],
        df_EC_trip_ICE_HEV[df_EC_trip_ICE_HEV['Vehicle Type'] == 'ICE']['FCR'],
        alpha=0.5,
        c='blue',
        label='ICE'
    )
    ax1.scatter(
        df_EC_trip_ICE_HEV[df_EC_trip_ICE_HEV['Vehicle Type'] == 'HEV']['Distance[km]'],
        df_EC_trip_ICE_HEV[df_EC_trip_ICE_HEV['Vehicle Type'] == 'HEV']['FCR'],
        alpha=0.5,
        c='green',
        label='HEV'
    )
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Fuel Consumption Rate (L/hr)')
    ax1.set_title('ICE vs HEV')
    ax1.legend()
    ax1.grid(True)

    # Subplot 2: EV vs PHEV (Distance vs HV Battery Power)
    ax2 = axes[1]
    ax2.scatter(
        df_EC_trip_EV_PHEV[df_EC_trip_EV_PHEV['Vehicle Type'] == 'PHEV']['Distance[km]'],
        df_EC_trip_EV_PHEV[df_EC_trip_EV_PHEV['Vehicle Type'] == 'PHEV']['HV Battery Power[Watts]'],
        alpha=0.5,
        c='red',
        label='PHEV'
    )
    ax2.scatter(
        df_EC_trip_EV_PHEV[df_EC_trip_EV_PHEV['Vehicle Type'] == 'EV']['Distance[km]'],
        df_EC_trip_EV_PHEV[df_EC_trip_EV_PHEV['Vehicle Type'] == 'EV']['HV Battery Power[Watts]'],
        alpha=0.5,
        c='orange',
        label='EV'
    )
    ax2.set_xlabel('Distance (km)')
    ax2.set_ylabel('HV Battery Power (Watts)')
    ax2.set_title('EV vs PHEV')
    ax2.legend()
    ax2.grid(True)

    # Shared Title and Layout
    fig.suptitle('Energy Consumption Comparison by Vehicle Type over Trip', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 1])  # Leave space for suptitle

    # Display the figure in Streamlit
    st.pyplot(fig)

    st.markdown("""
    - **ICE vs HEV Plot (Left):** Shows a negative correlation between distance and fuel consumption rate, where fuel consumption tends to be higher (1.5-2.0 L/hr) at shorter distances (0-10 km) and decreases (below 1.0 L/hr) as trip distance increases. ICE vehicles (blue) generally show higher fuel consumption compared to HEV vehicles (green) across all distances
    - **EV vs PHEV Plot (Right):** Shows the battery power usage pattern, where PHEV vehicles (red) exhibit both positive and negative power values (-20000 to +5000 Watts) indicating both battery discharge and regenerative charging, while EV vehicles (yellow) appear to have a more concentrated power usage pattern. The spread of power values is most diverse in the 0-10 km range and becomes more sparse at longer distances.
    - These plots enable a visual comparison of energy consumption patterns across different vehicle technologies.
    """)

    st.markdown("""
    #### Energy Consumption Comparison by Vehicle Type over Time

    The following scatter plots visualize the relationship between trip distance and energy consumption metrics for different vehicle types over time:
    - **Left Plot:** Distance vs Fuel Consumption Rate (FCR) for Internal Combustion Engine (ICE) and Hybrid Electric Vehicles (HEV).
    - **Right Plot:** Distance vs HV Battery Power for Electric Vehicles (EV) and Plug-in Hybrid Electric Vehicles (PHEV).

    These visualizations help compare operational efficiency and energy usage patterns across various powertrain technologies over time periods.
    """)

    # Create a 1x2 subplot for the two comparisons
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=False)

    # Subplot 1: ICE vs HEV (Distance vs FCR)
    ax1 = axes[0]
    ax1.scatter(
        df_EC_time_ICE_HEV[df_EC_time_ICE_HEV['Vehicle Type'] == 'ICE']['Distance[km]'],
        df_EC_time_ICE_HEV[df_EC_time_ICE_HEV['Vehicle Type'] == 'ICE']['FCR'],
        alpha=0.5,
        c='blue',
        label='ICE'
    )
    ax1.scatter(
        df_EC_time_ICE_HEV[df_EC_time_ICE_HEV['Vehicle Type'] == 'HEV']['Distance[km]'],
        df_EC_time_ICE_HEV[df_EC_time_ICE_HEV['Vehicle Type'] == 'HEV']['FCR'],
        alpha=0.5,
        c='green',
        label='HEV'
    )
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Fuel Consumption Rate (L/hr)')
    ax1.set_title('ICE vs HEV')
    ax1.legend()
    ax1.grid(True)

    # Subplot 2: EV vs PHEV (Distance vs HV Battery Power)
    ax2 = axes[1]
    ax2.scatter(
        df_EC_time_EV_PHEV[df_EC_time_EV_PHEV['Vehicle Type'] == 'PHEV']['Distance[km]'],
        df_EC_time_EV_PHEV[df_EC_time_EV_PHEV['Vehicle Type'] == 'PHEV']['HV Battery Power[Watts]'],
        alpha=0.5,
        c='red',
        label='PHEV'
    )
    ax2.scatter(
        df_EC_time_EV_PHEV[df_EC_time_EV_PHEV['Vehicle Type'] == 'EV']['Distance[km]'],
        df_EC_time_EV_PHEV[df_EC_time_EV_PHEV['Vehicle Type'] == 'EV']['HV Battery Power[Watts]'],
        alpha=0.5,
        c='orange',
        label='EV'
    )
    ax2.set_xlabel('Distance (km)')
    ax2.set_ylabel('HV Battery Power (Watts)')
    ax2.set_title('EV vs PHEV')
    ax2.legend()
    ax2.grid(True)

    # Shared Title and Layout
    fig.suptitle('Energy Consumption Comparison by Vehicle Type over Time', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 1])  # Leave space for suptitle

    # Display the figure in Streamlit
    st.pyplot(fig)

    st.markdown("""
    - **ICE vs HEV Plot (Left):** Shows that ICE vehicles (blue) have higher fuel consumption rates (0.8-0.9 L/hr) compared to HEV vehicles (green) (0.5-0.6 L/hr) over time. The data points are concentrated around 4 km distance, with a few outliers extending up to 16 km, showing HEV's consistently better fuel efficiency across different time periods.
    - **EV vs PHEV Plot (Right):** Displays battery power consumption between -2000 to -14000 Watts for both vehicle types, with PHEV (red) showing slightly more scattered power usage compared to EV (yellow). The majority of data points fall within 0-7.5 km range, with occasional trips extending to 17.5 km, suggesting these represent typical daily driving patterns over time.
    - These plots enable a visual comparison of energy consumption patterns across different vehicle technologies and their temporal variations.
    """)

    st.markdown("""Individual Vehicle Type analysis by Trip: ICE, HEV, EV, PHEV
    """)

    df_ICE = df_EC_trip_ICE_HEV[df_EC_trip_ICE_HEV['Vehicle Type'] == 'ICE']
    df_HEV = df_EC_trip_ICE_HEV[df_EC_trip_ICE_HEV['Vehicle Type'] == 'HEV']
    df_EV = df_EC_trip_EV_PHEV[df_EC_trip_EV_PHEV['Vehicle Type'] == 'EV']
    df_PHEV = df_EC_trip_EV_PHEV[df_EC_trip_EV_PHEV['Vehicle Type'] == 'PHEV']

    df_ICE.fillna({
    'OAT[DegC]': 15,
    'Generalized_Weight': df_ICE['Generalized_Weight'].mean(),
    'FCR': df_ICE['FCR'].mean(),
    'HV Battery Power[Watts]': 0.0,  # ICE vehicles typically do not have HV Battery Power
    'Air Conditioning Power[Watts]' : 0,
    'Heater Power[Watts]': df_ICE['Heater Power[Watts]'].mean(),
    'MAF[g/sec]': df_ICE['MAF[g/sec]'].mean(),
    'Absolute Load[%]': df_ICE['Absolute Load[%]'].mean(),
    'Short Term Fuel Trim Bank 1[%]': df_ICE['Short Term Fuel Trim Bank 1[%]'].mean(),
    'Short Term Fuel Trim Bank 2[%]': df_ICE['Short Term Fuel Trim Bank 2[%]'].mean(),
    'Long Term Fuel Trim Bank 1[%]': df_ICE['Long Term Fuel Trim Bank 1[%]'].mean(),
    'Long Term Fuel Trim Bank 2[%]': df_ICE['Long Term Fuel Trim Bank 2[%]'].mean(),
    'Vehicle Speed[km/h]': df_ICE['Vehicle Speed[km/h]'].mean(),
    'Distance[km]': df_ICE['Distance[km]'].mean(),
    'Engine RPM[RPM]': df_ICE['Engine RPM[RPM]'].mean()
    }, inplace=True)

    df_HEV.fillna({
        'OAT[DegC]': 15,
        'FCR': df_HEV['FCR'].mean(),
        'HV Battery Power[Watts]': 0,
        'Air Conditioning Power[Watts]': 0,
        'Heater Power[Watts]': 0,
        'MAF[g/sec]': df_HEV['MAF[g/sec]'].mean(),
        'Absolute Load[%]': df_HEV['Absolute Load[%]'].mean(),
        'Short Term Fuel Trim Bank 1[%]': df_HEV['Short Term Fuel Trim Bank 1[%]'].mean(),
        'Short Term Fuel Trim Bank 2[%]': df_HEV['Short Term Fuel Trim Bank 2[%]'].mean(),
        'Long Term Fuel Trim Bank 1[%]': df_HEV['Long Term Fuel Trim Bank 1[%]'].mean(),
        'Long Term Fuel Trim Bank 2[%]': df_HEV['Long Term Fuel Trim Bank 2[%]'].mean(),'Vehicle Speed[km/h]': df_ICE['Vehicle Speed[km/h]'].mean(),
        'Distance[km]': df_HEV['Distance[km]'].mean(),
        'Engine RPM[RPM]': df_HEV['Engine RPM[RPM]'].mean()
    }, inplace=True)

    df_EV.fillna({
        'Engine RPM[RPM]': 0,  # EVs typically do not have engine RPM
        'FCR': 0,  # EVs typically do not have fuel consumption rate  
        'MAF[g/sec]': 0,
        'Absolute Load[%]': 0,
        'Short Term Fuel Trim Bank 1[%]': 0,
        'Short Term Fuel Trim Bank 2[%]': 0,
        'Long Term Fuel Trim Bank 1[%]': 0,
        'Long Term Fuel Trim Bank 2[%]': 0
    }, inplace=True)

    df_PHEV.fillna({
        'OAT[DegC]': 15,
        'FCR': df_PHEV['FCR'].mean(),
        'HV Battery Power[Watts]': 0.0,  # PHEVs typically do not have HV Battery Power
        'Air Conditioning Power[Watts]': 0,
        'Heater Power[Watts]': 0,
        'MAF[g/sec]': df_PHEV['MAF[g/sec]'].mean(),
        'Absolute Load[%]': df_PHEV['Absolute Load[%]'].mean(),
        'Short Term Fuel Trim Bank 1[%]': df_PHEV['Short Term Fuel Trim Bank 1[%]'].mean(),
        'Short Term Fuel Trim Bank 2[%]': 0,
        'Long Term Fuel Trim Bank 1[%]': df_PHEV['Long Term Fuel Trim Bank 1[%]'].mean(),
        'Long Term Fuel Trim Bank 2[%]': 0
    }, inplace=True)

    st.subheader("Data Overview: ICE Vehicles")
    st.dataframe(df_ICE.describe(), use_container_width=True)

    st.subheader("Data Overview: HEV Vehicles")
    st.dataframe(df_HEV.describe(), use_container_width=True)

    st.subheader("Data Overview: EV Vehicles")
    st.dataframe(df_EV.describe(), use_container_width=True)

    st.subheader("Data Overview: PHEV Vehicles")
    st.dataframe(df_PHEV.describe(), use_container_width=True)

with tabs[1]:

    st.markdown("""
        ### Supervised Learning - Regression     
        - **Model:** Linear Regression.
        - **X_test, y_test:** Test data for further analysis.
        - **y_pred:** Predictions on test set.
        - **Regression Line:** Model for plotting Actual vs Predicted regression line.
    """)

    @st.cache_data
    def linear_regression_analysis(features, target, X, y):

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Fit linear model
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Print model coefficients
        print("Model Coefficients:")
        for feature, coef in zip(features, model.coef_):
            print(f"  {feature}: {coef:.4f}")

        # Print regression equation
        equation = " + ".join([f"{coef:.4f}*{feature}" for feature, coef in zip(features, model.coef_)])
        print(f"\nRegression Equation:")
        print(f"  Slope of the regression line: {model.coef_}")
        print(f"  Intercept: {model.intercept_:.4f}")
        print(f"  Target Variable: {target}")
        print(f"  {target} = {equation} + {model.intercept_:.4f}\n")

        # Evaluation metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        print("Evaluation Metrics:")
        print(f"  R² Score  : {r2:.4f}")
        print(f"  MAE       : {mae:.4f}")
        print(f"  MSE       : {mse:.4f}")
        print(f"  RMSE      : {rmse:.4f}\n")

        # Regression line for plotting (optional)
        regression_line_model = LinearRegression()
        regression_line_model.fit(y_test.values.reshape(-1, 1), y_pred)

        return model, X_test, y_test, y_pred, regression_line_model, r2, mae, mse, rmse
    
    st.markdown("""
        ### Linear Regression to Predict Energy Consumption in ICE, HEV, EV and PHEV Vehicles
    """)

    st.success("""### Predict FCR for ICE vehicles""")

    features = ['Vehicle Speed[km/h]', 'Distance[km]', 'Generalized_Weight','MAF[g/sec]',
                'Absolute Load[%]', 'Short Term Fuel Trim Bank 1[%]',
                'Short Term Fuel Trim Bank 2[%]', 'Long Term Fuel Trim Bank 1[%]',
                'Long Term Fuel Trim Bank 2[%]']
    target = 'FCR'

    # Split and train
    X = df_ICE[features]
    y = df_ICE[target]
    model, X_test, y_test, y_pred, regression_line_model, r2, mae, mse, rmse = linear_regression_analysis(features, target, X, y)

    # Top description
    st.subheader("***Actual vs Predicted Fuel Consumption Rate (FCR) for ICE Vehicles***")
    st.markdown("This scatter plot compares the **actual FCR** values with those predicted by the linear regression model. A red dashed line represents the fitted regression line.")

    st.markdown("""
        - **Used features:** Vehicle Speed[km/h], Distance[km], Generalized_Weight, MAF[g/sec], Absolute Load[%], Short Term Fuel Trim Bank 1[%], Short Term Fuel Trim Bank 2[%], Long Term Fuel Trim Bank 1[%], Long Term Fuel Trim Bank 2[%].
        - Plotted Actual vs Predicted FCR values and regression line.
        - Displayed evaluation metrics: R², MAE, MSE, and RMSE on the plot.
    """)

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 3))
    y_test_sorted = np.sort(y_test)
    y_line = regression_line_model.predict(y_test_sorted.reshape(-1, 1))

    ax.scatter(y_test, y_pred, alpha=0.5, color='blue')
    ax.plot(y_test_sorted, y_line, color='red', linestyle='--', linewidth=2, label='Regression Line')
    ax.set_title('Actual vs Predicted FCR for ICE Vehicles')
    ax.set_xlabel('Actual FCR (L/hr)')
    ax.set_ylabel('Predicted FCR (L/hr)')
    ax.grid(True)
    ax.legend()

    # Display metrics inside plot
    textstr = f'R²: {r2_score(y_test, y_pred):.4f}\nMAE: {mean_absolute_error(y_test, y_pred):.4f}\nMSE: {mean_squared_error(y_test, y_pred):.4f}\nRMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}'
    ax.text(1.20, 0.5, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    st.pyplot(fig)

    # --- Show Coefficients ---
    st.markdown("**Model Coefficients:**")
    coef_lines = [f"  {feature}: {coef:.4f}" for feature, coef in zip(features, model.coef_)]
    st.code("\n".join(coef_lines))

    # --- Regression Equation ---
    equation = " + ".join([f"{coef:.4f}*{feature}" for feature, coef in zip(features, model.coef_)])
    st.markdown("**Regression Equation:**")
    st.code(f"Slope of the regression line: {model.coef_}\nIntercept: {model.intercept_:.4f}\nTarget Variable: {target}\n{target} = {equation} + {model.intercept_:.4f}")

    # --- Evaluation Metrics ---
    st.markdown("**Evaluation Metrics:**")
    metric_text = f"""
    R² Score  : {r2:.4f}
    MAE       : {mae:.4f}
    MSE       : {mse:.4f}
    RMSE      : {rmse:.4f}
    """
    st.code(metric_text) 

    st.markdown("""
    ### SHAP Feature Importance for ICE Vehicle FCR Prediction
    The following visualizations use SHAP (SHapley Additive exPlanations) to interpret the linear regression model predicting Fuel Consumption Rate (FCR) for ICE vehicles.
    - **Bar Plot:** Shows mean absolute SHAP value for each feature (global importance).
    - **Beeswarm Plot:** Shows the distribution and impact of each feature on model output.
    - **Waterfall Plot:** Explains the SHAP values for a single prediction.
    - **SHAP Values Table:** Displays the SHAP values for the test set.
    """)

    # Standardize features for SHAP and model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # SHAP explanation
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    # Bar plot (global feature importance)
    st.markdown("#### SHAP Bar Plot (Global Feature Importance)")
    import matplotlib.pyplot as plt
    fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
    shap.plots.bar(shap_values, show=False, ax=ax_bar)
    st.pyplot(fig_bar)
    plt.close(fig_bar)

    # Beeswarm plot (feature impact distribution)
    st.markdown("#### SHAP Beeswarm Plot (Feature Impact Distribution)")
    plt.figure(figsize=(8, 4))
    shap.plots.beeswarm(shap_values, show=False)
    fig_beeswarm = plt.gcf()
    st.pyplot(fig_beeswarm)
    plt.close(fig_beeswarm)

    # Waterfall plot (single prediction explanation)
    st.markdown("#### SHAP Waterfall Plot (First Test Sample)")
    plt.figure(figsize=(8, 4))
    shap.plots.waterfall(shap_values[0], show=False)
    fig_waterfall = plt.gcf()
    st.pyplot(fig_waterfall)
    plt.close(fig_waterfall)

    # SHAP values as DataFrame
    shap_df = pd.DataFrame(shap_values.values, columns=features)
    st.markdown("#### SHAP Values Table (First 10 Test Samples)")
    st.dataframe(shap_df.head(10), use_container_width=True)

    st.markdown("""
    **Comments:**
    - The bar plot shows which features have the largest average impact on FCR prediction.
    - The beeswarm plot visualizes how each feature's value (high/low) pushes the prediction higher or lower.
    - The waterfall plot explains the SHAP value breakdown for a single prediction.
    - The SHAP values table provides the actual SHAP values for each feature and test sample.
    """)

    features = ['Vehicle Speed[km/h]'
                , 'Generalized_Weight'
                ,'MAF[g/sec]'
                ,'Absolute Load[%]'
                ,'Long Term Fuel Trim Bank 1[%]']
    target = 'FCR'

    # Split and train
    X = df_ICE[features]
    y = df_ICE[target]
    model, X_test, y_test, y_pred, regression_line_model, r2, mae, mse, rmse = linear_regression_analysis(features, target, X, y)

    # Top description
    st.subheader("***Actual vs Predicted Fuel Consumption Rate (FCR) for ICE Vehicles after SHAP Analysis***")

    st.markdown("""
        - **Used features:** Vehicle Speed[km/h], Generalized_Weight, MAF[g/sec], Absolute Load[%], Long Term Fuel Trim Bank 1[%]
        - Plotted Actual vs Predicted FCR values and regression line.
        - Displayed evaluation metrics: R², MAE, MSE, and RMSE on the plot.
    """)


    # Plotting
    fig, ax = plt.subplots(figsize=(6, 3))
    y_test_sorted = np.sort(y_test)
    y_line = regression_line_model.predict(y_test_sorted.reshape(-1, 1))

    ax.scatter(y_test, y_pred, alpha=0.5, color='blue')
    ax.plot(y_test_sorted, y_line, color='red', linestyle='--', linewidth=2, label='Regression Line')
    ax.set_title('Actual vs Predicted FCR for ICE Vehicles')
    ax.set_xlabel('Actual FCR (L/hr)')
    ax.set_ylabel('Predicted FCR (L/hr)')
    ax.grid(True)
    ax.legend()

    # Display metrics inside plot
    textstr = f'R²: {r2_score(y_test, y_pred):.4f}\nMAE: {mean_absolute_error(y_test, y_pred):.4f}\nMSE: {mean_squared_error(y_test, y_pred):.4f}\nRMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}'
    ax.text(1.20, 0.5, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    st.pyplot(fig)

    # --- Show Coefficients ---
    st.markdown("**Model Coefficients:**")
    coef_lines = [f"  {feature}: {coef:.4f}" for feature, coef in zip(features, model.coef_)]
    st.code("\n".join(coef_lines))

    # --- Regression Equation ---
    equation = " + ".join([f"{coef:.4f}*{feature}" for feature, coef in zip(features, model.coef_)])
    st.markdown("**Regression Equation:**")
    st.code(f"Slope of the regression line: {model.coef_}\nIntercept: {model.intercept_:.4f}\nTarget Variable: {target}\n{target} = {equation} + {model.intercept_:.4f}")

    # --- Evaluation Metrics ---
    st.markdown("**Evaluation Metrics:**")
    metric_text = f"""
    R² Score  : {r2:.4f}
    MAE       : {mae:.4f}
    MSE       : {mse:.4f}
    RMSE      : {rmse:.4f}
    """
    st.code(metric_text) 

    st.markdown("***The model can predict FCR for ICE vehicles reasonably well, but not as accurately as for HEVs. There is a moderate linear relationship, but other factors may influence FCR in ICE vehicles, or the selected features may not capture all the variability. Further feature engineering or model improvement could enhance prediction accuracy.***")

    st.success("""### Predict FCR for HEV vehicles""")

    features = ['MAF[g/sec]']
    target = 'FCR'

    # Split and train
    X = df_HEV[features]
    y = df_HEV[target]
    model, X_test, y_test, y_pred, regression_line_model, r2, mae, mse, rmse = linear_regression_analysis(features, target, X, y)

    # Top description
    st.subheader("***Actual vs Predicted Fuel Consumption Rate (FCR) for HEV Vehicles***")
    st.markdown("This scatter plot compares the **actual FCR** values with those predicted by the linear regression model. A red dashed line represents the fitted regression line.")

    st.markdown("""
        - **Used features:** MAF[g/sec]
        - Plotted Actual vs Predicted FCR values and regression line.
        - Displayed evaluation metrics: R², MAE, MSE, and RMSE on the plot.
    """)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(6, 3))
    y_test_sorted = np.sort(y_test)
    y_line = regression_line_model.predict(y_test_sorted.reshape(-1, 1))

    ax.scatter(y_test, y_pred, alpha=0.5, color='blue')
    ax.plot(y_test_sorted, y_line, color='red', linestyle='--', linewidth=2, label='Regression Line')
    ax.set_title('Actual vs Predicted FCR for HEV Vehicles')
    ax.set_xlabel('Actual FCR (L/hr)')
    ax.set_ylabel('Predicted FCR (L/hr)')
    ax.grid(True)
    ax.legend()

    # Display metrics inside plot
    textstr = f'R²: {r2_score(y_test, y_pred):.4f}\nMAE: {mean_absolute_error(y_test, y_pred):.4f}\nMSE: {mean_squared_error(y_test, y_pred):.4f}\nRMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}'
    ax.text(1.20, 0.5, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5))
    plt.tight_layout()
    st.pyplot(fig)

    # --- Show Coefficients ---
    st.markdown("**Model Coefficients:**")
    coef_lines = [f"  {feature}: {coef:.4f}" for feature, coef in zip(features, model.coef_)]
    st.code("\n".join(coef_lines))

    # --- Regression Equation ---
    equation = " + ".join([f"{coef:.4f}*{feature}" for feature, coef in zip(features, model.coef_)])
    st.markdown("**Regression Equation:**")
    st.code(f"Slope of the regression line: {model.coef_}\nIntercept: {model.intercept_:.4f}\nTarget Variable: {target}\n{target} = {equation} + {model.intercept_:.4f}")

    # --- Evaluation Metrics ---
    st.markdown("**Evaluation Metrics:**")
    metric_text = f"""
    R² Score  : {r2:.4f}
    MAE       : {mae:.4f}
    MSE       : {mse:.4f}
    RMSE      : {rmse:.4f}
    """
    st.code(metric_text) 

    st.markdown("***The linear regression model predicts FCR for HEV vehicles with high accuracy using the selected feature(s). The model's predictions are very close to the actual values, as shown by the high R² and low error metrics. This suggests that the chosen feature(s) (here, 'MAF[g/sec]') are strong predictors of FCR for HEVs in this dataset.***")

    st.success("""### Predict Battery Power for EV vehicles""")

    features = [
    'Air Conditioning Power[Watts]',
    'Heater Power[Watts]',
    'Vehicle Speed[km/h]',
    ]

    target = 'HV Battery Power[Watts]'

    # Split and train
    X = df_EV[features]
    y = df_EV[target]
    model, X_test, y_test, y_pred, regression_line_model, r2, mae, mse, rmse = linear_regression_analysis(features, target, X, y)

    # Top description
    st.subheader("***Actual vs Predicted HV Battery Power for EVs***")
    st.markdown("This scatter plot compares the **actual Battery Power** values with those predicted by the linear regression model. A red dashed line represents the fitted regression line.")

    st.markdown("""
        - **Used features:** Air Conditioning Power[Watts], Heater Power[Watts], Vehicle Speed[km/h]
        - Plotted Actual vs Predicted Battery Power values and regression line.
        - Displayed evaluation metrics: R², MAE, MSE, and RMSE on the plot.
    """)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(6, 3))
    y_test_sorted = np.sort(y_test)
    y_line = regression_line_model.predict(y_test_sorted.reshape(-1, 1))

    ax.scatter(y_test, y_pred, alpha=0.5, color='blue')
    ax.plot(y_test_sorted, y_line, color='red', linestyle='--', linewidth=2, label='Regression Line')
    ax.set_title('Actual vs Predicted HV Battery Power for EVs')
    ax.set_xlabel('Actual Power (Watts)')
    ax.set_ylabel('Predicted Power (Watts)')
    ax.grid(True)
    ax.legend()

    # Display metrics inside plot
    textstr = f'R²: {r2_score(y_test, y_pred):.4f}\nMAE: {mean_absolute_error(y_test, y_pred):.4f}\nMSE: {mean_squared_error(y_test, y_pred):.4f}\nRMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}'
    ax.text(1.20, 0.5, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5))
    plt.tight_layout()
    st.pyplot(fig)

    # --- Show Coefficients ---
    st.markdown("**Model Coefficients:**")
    coef_lines = [f"  {feature}: {coef:.4f}" for feature, coef in zip(features, model.coef_)]
    st.code("\n".join(coef_lines))

    # --- Regression Equation ---
    equation = " + ".join([f"{coef:.4f}*{feature}" for feature, coef in zip(features, model.coef_)])
    st.markdown("**Regression Equation:**")
    st.code(f"Slope of the regression line: {model.coef_}\nIntercept: {model.intercept_:.4f}\nTarget Variable: {target}\n{target} = {equation} + {model.intercept_:.4f}")

    # --- Evaluation Metrics ---
    st.markdown("**Evaluation Metrics:**")
    metric_text = f"""
    R² Score  : {r2:.4f}
    MAE       : {mae:.4f}
    MSE       : {mse:.4f}
    RMSE      : {rmse:.4f}
    """
    st.code(metric_text)

    st.markdown("***The selected features do not adequately explain or predict the HV Battery Power for EVs in this dataset. The model fails to capture the relationship, suggesting the need for better feature selection, more data, or a different modeling approach.***")

    st.success("""### Predict Battery Power for PHEV vehicles""")

    features = [
    'Engine RPM[RPM]',
    'Air Conditioning Power[Watts]',
    'Vehicle Speed[km/h]',
    'OAT[DegC]',
    ]

    target = 'HV Battery Power[Watts]'

    # Split and train
    X = df_PHEV[features]
    y = df_PHEV[target]
    model, X_test, y_test, y_pred, regression_line_model, r2, mae, mse, rmse = linear_regression_analysis(features, target, X, y)

    # Top description
    st.subheader("***Actual vs Predicted HV Battery Power for PHEVs***")
    st.markdown("This scatter plot compares the **actual Battery Power** values with those predicted by the linear regression model. A red dashed line represents the fitted regression line.")

    st.markdown("""
        - **Used features:** Air Conditioning Power[Watts], Heater Power[Watts], Vehicle Speed[km/h]
        - Plotted Actual vs Predicted Battery Power values and regression line.
        - Displayed evaluation metrics: R², MAE, MSE, and RMSE on the plot.
    """)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(6, 3))
    y_test_sorted = np.sort(y_test)
    y_line = regression_line_model.predict(y_test_sorted.reshape(-1, 1))

    ax.scatter(y_test, y_pred, alpha=0.5, color='blue')
    ax.plot(y_test_sorted, y_line, color='red', linestyle='--', linewidth=2, label='Regression Line')
    ax.set_title('Actual vs Predicted HV Battery Power for PHEVs')
    ax.set_xlabel('Actual Power (Watts)')
    ax.set_ylabel('Predicted Power (Watts)')
    ax.grid(True)
    ax.legend()

    # Display metrics inside plot
    textstr = f'R²: {r2_score(y_test, y_pred):.4f}\nMAE: {mean_absolute_error(y_test, y_pred):.4f}\nMSE: {mean_squared_error(y_test, y_pred):.4f}\nRMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}'
    ax.text(1.20, 0.5, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5))
    plt.tight_layout()
    st.pyplot(fig)

    # --- Show Coefficients ---
    st.markdown("**Model Coefficients:**")
    coef_lines = [f"  {feature}: {coef:.4f}" for feature, coef in zip(features, model.coef_)]
    st.code("\n".join(coef_lines))

    # --- Regression Equation ---
    equation = " + ".join([f"{coef:.4f}*{feature}" for feature, coef in zip(features, model.coef_)])
    st.markdown("**Regression Equation:**")
    st.code(f"Slope of the regression line: {model.coef_}\nIntercept: {model.intercept_:.4f}\nTarget Variable: {target}\n{target} = {equation} + {model.intercept_:.4f}")

    # --- Evaluation Metrics ---
    st.markdown("**Evaluation Metrics:**")
    metric_text = f"""
    R² Score  : {r2:.4f}
    MAE       : {mae:.4f}
    MSE       : {mse:.4f}
    RMSE      : {rmse:.4f}
    """
    st.code(metric_text)

    st.markdown("***The selected features are effective in predicting HV Battery Power for PHEVs. The model captures the relationship well, making it suitable for estimating power consumption in PHEV vehicles.***")

with tabs[2]:
    st.markdown("""
        ### Unsupervised Learning - Clustering    
        - **Model:** K-means
        - **Inertia:** Calculates the difference in inertia to suggest an optimal number of clusters (elbow point).
        - **Clusters:** Predicts cluster labels for the input data.
    """)

    # --- Streamlit UI and Descriptions ---
    st.markdown("""
    ### K-Means Clustering: Elbow Method and Cluster Assignment

    This section demonstrates unsupervised learning using K-Means clustering:
    - **Elbow Method:** Visualizes the inertia (within-cluster sum of squares) for different cluster counts to help select the optimal number of clusters.
    - **Cluster Assignment:** Assigns cluster labels to each data point and displays the resulting cluster assignments.

    The elbow plot helps determine the best value for K, and the resulting clusters are shown in a table.
    """)

    # --- Data Preparation and Clustering ---
    df_combined = read_parquet_from_s3(bucket_name, df_combined)
    df_combined_sf = df_combined[['Vehicle Type', 'Vehicle Speed[km/h]', 'FCR']]

    # Map vehicle type strings to numeric codes
    df_combined_sf['Vehicle Type'] = df_combined_sf['Vehicle Type'].map({'ICE': 0, 'HEV': 1, 'EV': 2, 'PHEV': 3})

    # --- Elbow Plot Function ---
    @st.cache_data
    def plot_kmeans_elbow(df):
        """
        Plots the elbow curve for KMeans clustering to help select the optimal number of clusters.
        Returns the inertia list and suggested optimal K.
        """
        inertia = []
        for i in range(1, 5):
            kmeans = KMeans(n_clusters=i, init='random', random_state=42)
            kmeans.fit(df)
            inertia.append(kmeans.inertia_)

        # Find elbow point (simple method: where the decrease in inertia slows down the most)
        diff = np.diff(inertia)
        elbow_k = np.argmin(diff) + 2  # +2 because diff is one less and we start from k=1
        print("Suggested optimal K:", elbow_k)   

        # Plot the elbow curve using Streamlit
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(range(1, 5), inertia, 'bo-')
        ax.set_xlabel('Number of Clusters (K)')
        ax.set_ylabel('Inertia')
        ax.set_title('Elbow Method for Optimal K Selection')
        ax.grid(True)
        st.pyplot(fig)

    # Elbow plot
    plot_kmeans_elbow(df_combined_sf)

    # --- Fit and Predict KMeans Function ---
    @st.cache_data
    def fit_predict_kmeans(df, n_clusters, random_state=42):
        """
        Fits KMeans clustering on the given DataFrame and returns a copy with a new 'Cluster' column.
        """

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        kmeans.fit(df)
        pred = kmeans.predict(df)
        df['Cluster'] = kmeans.labels_
        return df, pred, kmeans

    # --- Streamlit Descriptions ---
    st.markdown("""
    ### K-Means Clustering (Speed vs FCR by Vehicle Type)
    - Loads the combined dataset and selects relevant features.
    - Maps vehicle type strings to numeric codes for clustering.
    - Runs the elbow method to suggest optimal cluster count.
    - Fits KMeans (K=3) and assigns cluster labels.
    - Visualizes clusters for each vehicle type (Speed vs FCR).
    - Shows the cluster assignment table.
    """)

    # Fit KMeans and assign clusters
    df_combined_sf, pred, kmeans = fit_predict_kmeans(df_combined_sf, 3)

    # --- Visualization: Clusters by Vehicle Type ---
    vehicle_types = df_combined_sf['Vehicle Type'].unique()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=True)
    axes = axes.flatten()

    for i, vtype in enumerate(vehicle_types):
        ax = axes[i]
        subset = df_combined_sf[df_combined_sf['Vehicle Type'] == vtype]
        scatter = ax.scatter(
            subset['Vehicle Speed[km/h]'],
            subset['FCR'],
            c=subset['Cluster'],
            cmap='viridis',
            s=50
        )
        # Map numeric vehicle type back to string for title
        vtype_str = {0: "ICE", 1: "HEV", 2: "EV", 3: "PHEV"}.get(vtype, str(vtype))
        ax.set_title(f'Vehicle Type: {vtype_str}')
        ax.set_xlabel('Speed')
        ax.set_ylabel('FCR')
        handles, labels = scatter.legend_elements(prop="colors")
        ax.legend(handles, [f"Cluster {i}" for i in range(len(handles))])

    fig.suptitle('KMeans Clusters: Speed vs FCR by Vehicle Type', fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
    **Outcome & Understanding:**

    - The plots show KMeans clustering results for Speed vs FCR across four vehicle types (ICE, HEV, PHEV, EV).
    - Each subplot represents a vehicle type, with points colored by cluster.
    - **Distinct Clusters:** Each vehicle type forms 2–3 clusters, indicating different driving or consumption patterns within each type.
    - **ICE & HEV:** Most data points are at lower FCR values, with clusters separated by speed.
    - **PHEV:** Shows a wider spread in FCR, with clusters capturing both low and high FCR at various speeds.
    - **EV:** FCR is near zero (as expected for electric vehicles), with clusters mainly reflecting speed differences.

    **Conclusion:**  
    KMeans clustering effectively segments vehicle operational patterns by type, revealing characteristic speed and fuel/energy consumption behaviors for each vehicle class.
    """)

    # --- Cluster Assignment Table ---
    st.markdown("**Cluster Assignment Table (first 10 rows):**")
    st.dataframe(df_combined_sf.head(10), use_container_width=True)

    st.markdown("""
    ### Silhouette Score for KMeans Clustering
    
    This visualization helps identify the optimal number of clusters using the **Silhouette Score** metric.
    - **Data Used**: 'Vehicle Speed[km/h]' and 'HV Battery Power[Watts]'
    - **Range Tested**: Clusters from 2 to 4
    - The **higher** the silhouette score, the **better** the clustering quality.
    - Iterates over cluster numbers 2 to 4.
    - Applies KMeans clustering on selected features.
    - Computes **silhouette score** for each model to evaluate clustering quality.
    - Displays both a **line chart** of scores and a **data table** for reference.
    """)

    df_combined_sb = df_combined[['Vehicle Speed[km/h]','HV Battery Power[Watts]']]

    # Compute silhouette scores
    silhouette_scores = []
    for n_clusters in range(2, 5):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(df_combined_sb[['Vehicle Speed[km/h]', 'HV Battery Power[Watts]']])
        labels = kmeans.labels_
        score = silhouette_score(df_combined_sb[['Vehicle Speed[km/h]', 'HV Battery Power[Watts]']], labels)
        silhouette_scores.append((n_clusters, score))

    # Convert to DataFrame for display
    score_df = pd.DataFrame(silhouette_scores, columns=['Number of Clusters', 'Silhouette Score'])
    st.dataframe(score_df)

    # Plot the silhouette scores
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(score_df['Number of Clusters'], score_df['Silhouette Score'], marker='o', color='blue')
    ax.set_title('Silhouette Scores for KMeans Clustering')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Silhouette Score')
    st.pyplot(fig)

    st.markdown("""
    ### K-Means Clustering (Speed vs Battery Power by Vehicle Type)
    - 2x2 grid of scatter plots, one for each vehicle type, showing the relationship between 'Vehicle Speed[km/h]' and 'HV Battery Power[Watts]' for each type..
    - It iterates over unique vehicle types, selects the corresponding subset of data, and plots the points colored by their KMeans cluster assignment.
    - Visualizes clusters for each vehicle type (Speed vs Battery Power).
    - Shows the cluster assignment table.
    """)

    df_combined_sb = df_combined[['Vehicle Type','Vehicle Speed[km/h]','HV Battery Power[Watts]']]
    # map vehicle type to 0,1,2,3
    df_combined_sb['Vehicle Type'] = df_combined_sb['Vehicle Type'].map({'ICE': 0, 'HEV': 1, 'EV': 2, 'PHEV': 3})

    # Fit KMeans and assign clusters
    df_combined_sb, pred, kmeans = fit_predict_kmeans(df_combined_sb, 3)

    # --- Visualization: Clusters by Vehicle Type ---
    vehicle_types = df_combined_sf['Vehicle Type'].unique()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=True)
    axes = axes.flatten()

    for i, vtype in enumerate(vehicle_types):
        ax = axes[i]
        subset = df_combined_sb[df_combined_sb['Vehicle Type'] == vtype]
        scatter = ax.scatter(
        subset['Vehicle Speed[km/h]'],
        subset['HV Battery Power[Watts]'],
        c=subset['Cluster'],
        cmap='viridis',
        s=50
    )
        # Map numeric vehicle type back to string for title
        vtype_str = {0: "ICE", 1: "HEV", 2: "EV", 3: "PHEV"}.get(vtype, str(vtype))
        ax.set_title(f'Vehicle Type: {vtype_str}')
        ax.set_xlabel('Speed')
        ax.set_ylabel('Battery Power')
        handles, labels = scatter.legend_elements(prop="colors")
        ax.legend(handles, [f"Cluster {i}" for i in range(len(handles))])

    fig.suptitle('KMeans Clusters: Speed vs Battery Power by Vehicle Type', fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
    **Outcome & Understanding (Battery Power Clustering):**

    - **ICE & HEV:** All data points are assigned to a single cluster (Cluster 0), indicating almost no variation in battery power—expected, as these vehicles have little or no high-voltage battery activity.
    - **PHEV & EV:** Multiple clusters are identified, reflecting more diverse battery power usage patterns. PHEVs show a wide range of battery power (including negative values, likely due to regenerative braking or charging), while EVs also display clustering based on battery power and speed.

    **Conclusion:**  
    Clustering reveals that battery power is only a meaningful differentiator for PHEV and EV types. ICE and HEV vehicles show negligible battery power variation, while PHEV and EV vehicles exhibit distinct operational patterns based on speed and battery power.
    """)

    # --- Cluster Assignment Table ---
    st.markdown("**Cluster Assignment Table (first 10 rows):**")
    st.dataframe(df_combined_sb.head(10), use_container_width=True)

    st.markdown("""
    ### KMeans Clustering: Outside Air Temperature vs FCR

    This scatter plot visualizes KMeans clustering on:
    - **X-axis**: Outside Air Temperature (`OAT[DegC]`)
    - **Y-axis**: Fuel Consumption Rate (`FCR`)
    - Points are **colored by cluster assignment** using the 'viridis' colormap.
    - Useful for identifying patterns in vehicle behavior based on environmental conditions.
    """)

    df_combined_of = df_combined[['OAT[DegC]','FCR']]
    
    # Fit KMeans and assign clusters
    df_combined_of, pred, kmeans = fit_predict_kmeans(df_combined_of,5)

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(
        df_combined_of['OAT[DegC]'],
        df_combined_of['FCR'],
        c=df_combined_of['Cluster'],
        cmap='viridis',
        s=50
    )
    ax.set_xlabel('Outside Air Temperature')
    ax.set_ylabel('FCR')
    ax.set_title('KMeans Clusters: Outside Air Temperature vs FCR')

    # Cluster Legend
    handles, labels = scatter.legend_elements(prop="colors")
    ax.legend(handles, [f"Cluster {i}" for i in range(len(handles))], title="Clusters")
    st.pyplot(fig)

    st.markdown("""
    #### Cluster Analysis: Outside Air Temperature vs FCR

    The scatter plot visualizes **KMeans clustering** results for the relationship between:
    - **Outside Air Temperature (OAT[°C])**
    - **Fuel Consumption Rate (FCR)**

    #### Key Observations:
    - The data is grouped into **5 distinct clusters (Cluster 0–4)**.
    - **Cluster separation** appears to reflect patterns in how vehicles consume fuel under different outside temperatures.
    - **Cluster 4** (yellow) tends to appear at **very low temperatures** (below -10°C), showing **higher FCR variability**.
    - **Cluster 0 and 1** (dark colors) dominate the **moderate-to-warm temperature** range, where FCR is generally low.
    - **Cluster 3** shows more spread across moderate temperatures and seems to include more varied FCR.

    #### Insights:
    - Fuel consumption behavior varies with **environmental conditions**, especially temperature.
    - The model successfully segments driving patterns that may correspond to **cold-weather inefficiencies**, **efficient warm-weather driving**, or **transitional performance zones**.

    Understanding these clusters can help:
    - Optimize driving or energy usage under specific temperature conditions.
    - Inform EV or hybrid battery management strategies.
    """)

    st.markdown("""
    #### KMeans Clustering: Outside Air Temperature vs Battery Power

    This scatter plot visualizes KMeans clustering on:
    - **X-axis**: Outside Air Temperature (`OAT[DegC]`)
    - **Y-axis**: HV Battery Power (`Watts`)
    - Points are **colored by cluster assignment** using the 'viridis' colormap.
    - Useful for identifying patterns in vehicle behavior based on environmental conditions.
    """)

    df_combined_ob = df_combined[['OAT[DegC]','HV Battery Power[Watts]']]
    
    # Fit KMeans and assign clusters
    df_combined_ob, pred, kmeans = fit_predict_kmeans(df_combined_ob,5)

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(
        df_combined_ob['OAT[DegC]'],
        df_combined_ob['HV Battery Power[Watts]'],
        c=df_combined_ob['Cluster'],
        cmap='viridis',
        s=50
    )
    ax.set_xlabel('Outside Air Temperature')
    ax.set_ylabel('Battery Power')
    ax.set_title('KMeans Clusters: Outside Air Temperature vs Battery Power')

    # Cluster Legend
    handles, labels = scatter.legend_elements(prop="colors")
    ax.legend(handles, [f"Cluster {i}" for i in range(len(handles))], title="Clusters")
    st.pyplot(fig)

    st.markdown("""
    #### Cluster Analysis: Outside Air Temperature vs Battery Power

    This scatter plot visualizes **KMeans clustering** applied to:
    - **X-axis**: Outside Air Temperature (`OAT[DegC]`)
    - **Y-axis**: HV Battery Power (`[Watts]`)
    - Points are **grouped into 5 clusters (Cluster 0–4)** and **colored using the 'viridis' colormap**.

    #### Key Observations:
    - **Cluster 0** (purple) dominates the **upper power range**, indicating **higher battery output** at varying temperatures.
    - **Cluster 1 and 3** (dark blue and light green) represent data with **significant negative battery power**, suggesting **regenerative braking or power intake scenarios**, especially in **moderate temperatures**.
    - **Cluster 4** (yellow) appears consistent across low to moderate temperatures with a narrow range of battery power usage.
    - The separation between clusters shows **how battery power usage varies across temperature bands** and operating modes.

    #### Insights:
    - Clusters effectively segment **vehicle operating conditions** — such as energy consumption, regeneration, or idle states — influenced by **external temperature**.
    - This analysis can help improve:
    - **Battery management algorithms**
    - **Climate-aware energy optimization**
    - **Driving behavior insights across seasons**
    """)
