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

st.set_page_config(page_title="VED ML ICE/HEV/EV/PHEV Analysis", layout="centered", initial_sidebar_state="expanded")
st.title('VED ML Data Modelling - ICE, HEV, EV, and PHEV Analysis')

# Load environment variables only if running locally
if not st.secrets:
    load_dotenv()

AWS_ACCESS_KEY_ID = st.secrets.get("AWS_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID"))
AWS_SECRET_ACCESS_KEY = st.secrets.get("AWS_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY"))
AWS_DEFAULT_REGION = st.secrets.get("AWS_DEFAULT_REGION", os.getenv("AWS_DEFAULT_REGION"))

s3 = boto3.client("s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION
)

def read_parquet_from_s3(bucket_name, object_key):
    response = s3.get_object(Bucket=bucket_name, Key=object_key)
    file_content = response['Body'].read()
    df = pd.read_parquet(BytesIO(file_content))
    return df

with st.spinner("Loading data for ICE/HEV/EV/PHEV analysis..."):
    bucket_name = 's3aravindh973515031797'
    df = read_parquet_from_s3(bucket_name, 'Cleaned up VED Source Data/df_VED.parquet')

# --- Tab 4 Analysis ---
st.markdown("""
### ICE, HEV, EV, and PHEV Analysis: Distance vs FCR and HV Battery Power

The tables below show a comparison of key metrics for different vehicle types:
- **ICE & HEV**: Internal Combustion Engine and Hybrid Electric Vehicles
- **EV & PHEV**: Electric Vehicles and Plug-in Hybrid Electric Vehicles

The data is grouped by trip, time and vehicle type, and displays the top records for each group.
""")

df_EC_trip = df.groupby(['Trip', 'Vehicle Type'])[\
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

df['Date'] = pd.to_datetime(df['Date'])
df_EC_time = (
    df.groupby([df['Date'].dt.to_period('M'), 'Vehicle Type'])[\
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

st.markdown("""
---
### 🔗 Navigation

- [Data Loading & Preprocessing](https://share.streamlit.io/your-username/ved-ml-tab1-dataloading)
- [Data Visualization - Sample Plots](https://share.streamlit.io/your-username/ved-ml-tab2-visualization)
- [Exploratory Data Analysis](https://share.streamlit.io/your-username/ved-ml-tab3-eda)
- [ICE, HEV, EV, and PHEV Analysis](https://share.streamlit.io/your-username/ved-ml-tab4-ice-hev-ev-phev)
""")