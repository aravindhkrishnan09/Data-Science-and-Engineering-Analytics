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

st.set_page_config(page_title="VED ML EDA", layout="centered", initial_sidebar_state="expanded")
st.title('VED ML Data Modelling - Exploratory Data Analysis')

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

with st.spinner("Loading data for EDA..."):
    bucket_name = 's3aravindh973515031797'
    df = read_parquet_from_s3(bucket_name, 'Cleaned up VED Source Data/df_VED.parquet')

# --- Tab 3 EDA ---
st.markdown("""
### Battery Power, Fuel Consumption Rate (FCR), and Battery SOC
The table below shows the mean values of HV Battery Power, Air Conditioning Power, Heater Power, HV Battery SOC, and FCR, grouped by OAT_Category and Vehicle Type.
""")

# Grouping and aggregating the data
df_SOC = df.groupby(['OAT_Category', 'Vehicle Type'])[\
    ['HV Battery Power[Watts]', 'Air Conditioning Power[Watts]', 'Heater Power[Watts]', 'HV Battery SOC[%]', 'FCR']
].mean().reset_index().sort_values(by='HV Battery SOC[%]', ascending=False)

# Display the resulting dataframe in Streamlit
st.dataframe(df_SOC.head(10), use_container_width=True)

st.markdown("""**Comment:**
            The dataframe above provides a summary of key power and consumption metrics by OAT_Category and Vehicle Type.
            """)

st.markdown("""
### Battery Power and Fuel Consumption Rate (FCR) by Trip and Vehicle Type
The table below summarizes the mean values of key features such as Battery Power, Fuel Consumption Rate (FCR), Air Conditioning Power, Heater Power, and other relevant metrics, grouped by Trip and Vehicle Type. This allows for comparison of energy consumption and operational characteristics across different trips and vehicle types.
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

st.dataframe(df_EC_trip.head(10), use_container_width=True)

st.markdown("""
**Comments:**  
The table above displays the top 10 trip and vehicle type combinations with the highest Fuel Consumption Rate (FCR) and Battery Power usage.  
This helps identify trips and vehicle types with the most energy-intensive operation, supporting further analysis of efficiency and performance.
""")

st.markdown("""
### Battery Power and Fuel Consumption Rate (FCR) by Month and Vehicle Type
This table summarizes the mean values of key features such as Battery Power, Fuel Consumption Rate (FCR), Air Conditioning Power, Heater Power, and other relevant metrics, grouped by **Month** and **Vehicle Type**.  
It allows for the analysis of trends and operational characteristics over time and across different vehicle types.
""")

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

st.dataframe(df_EC_time.head(10), use_container_width=True)

st.markdown("""
**Comment:**  
The table above provides a summary of energy consumption and operational metrics by month and vehicle type, supporting the identification of seasonal or temporal trends in vehicle performance and efficiency.
""")

st.markdown("""
### Battery Power and Fuel Consumption Rate (FCR) by Date and Vehicle Type
This table summarizes the mean values of key features such as Battery Power, Fuel Consumption Rate (FCR), Air Conditioning Power, Heater Power, and other relevant metrics, grouped by **Date** and **Vehicle Type**.  
It allows for the analysis of daily trends and operational characteristics across different vehicle types.
""")

df['Date'] = pd.to_datetime(df['Date'])
df_EC_time = (
    df.groupby([df['Date'].dt.to_period('D'), 'Vehicle Type'])[\
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

st.dataframe(df_EC_time.head(10), use_container_width=True)

st.markdown("""
**Comment:**  
The table above provides a summary of energy consumption and operational metrics by date and vehicle type, supporting the identification of daily trends in vehicle performance and efficiency.
""")

st.markdown("""
### Battery Power and Fuel Consumption Rate (FCR) by Location and Vehicle Type
This table summarizes the mean values of key features such as Vehicle Speed, Absolute Load, Engine RPM, Outside Air Temperature (OAT), Generalized Weight, Fuel Consumption Rate (FCR), and HV Battery Power, grouped by **Latitude**, **Longitude**, and **Vehicle Type**.
It enables spatial analysis of vehicle operational characteristics and energy consumption patterns across different locations and vehicle types.
""")

df_la_lo = (
    df.groupby(['Latitude[deg]', 'Longitude[deg]', 'Vehicle Type'])[\
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

st.dataframe(df_la_lo.head(10), use_container_width=True)

st.markdown("""
**Comment:**  
The table above provides a spatial summary of energy consumption and operational metrics by location and vehicle type, supporting the identification of geographic trends in vehicle performance and efficiency.
""") 

st.markdown("""
---
### 🔗 Navigation

- [Data Loading & Preprocessing](https://share.streamlit.io/your-username/ved-ml-tab1-dataloading)
- [Data Visualization - Sample Plots](https://share.streamlit.io/your-username/ved-ml-tab2-visualization)
- [Exploratory Data Analysis](https://share.streamlit.io/your-username/ved-ml-tab3-eda)
- [ICE, HEV, EV, and PHEV Analysis](https://share.streamlit.io/your-username/ved-ml-tab4-ice-hev-ev-phev)
""")