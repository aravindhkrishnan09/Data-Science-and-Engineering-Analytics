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

st.set_page_config(page_title="VED ML Data Loading & Preprocessing", layout="centered", initial_sidebar_state="expanded")
st.title('VED ML Data Modelling - Data Loading and Preprocessing')

st.header("Data Loading and Preprocessing")

st.markdown("### 1. Data Loading")
st.markdown(
    """
    This section is responsible for importing both static and dynamic datasets required for further analysis. 
    Static data is loaded from Excel files, while dynamic data is aggregated from multiple CSV files within a specified directory.
    """
)

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

def read_parquet_from_s3(bucket_name, object_key):
    response = s3.get_object(Bucket=bucket_name, Key=object_key)
    file_content = response['Body'].read()
    df = pd.read_parquet(BytesIO(file_content))
    return df

with st.spinner("Loading data..."):
    
    bucket_name = 's3aravindh973515031797'
    ICE_HEV = 'Cleaned up VED Source Data/df_ICE_HEV.parquet'
    PHEV_EV = 'Cleaned up VED Source Data/df_PHEV_EV.parquet'
    df_static = 'Cleaned up VED Source Data/df_static.parquet'
    df_dynamic_sample = 'Cleaned up VED Source Data/df_dynamic_sample.parquet'
    df = 'Cleaned up VED Source Data/df_VED.parquet'

    df_ICE_HEV = read_parquet_from_s3(bucket_name, ICE_HEV)
    df_PHEV_EV = read_parquet_from_s3(bucket_name, PHEV_EV)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ICE & HEV Records", f"{len(df_ICE_HEV):,}")
    with col2:
        st.metric("PHEV & EV Records", f"{len(df_PHEV_EV):,}")           

st.markdown("### 2. Data Cleaning")
st.markdown(
    """
    The data cleaning process involves replacing placeholder values, correcting data types, renaming columns for consistency, and merging static datasets. 
    Duplicate records are also identified and reported.
    """
)

with st.spinner("Cleaning data..."):

    df_static = read_parquet_from_s3(bucket_name, df_static)
    df_dynamic_sample = read_parquet_from_s3(bucket_name, df_dynamic_sample)

    st.write("#### Data Cleaning Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Static Records", f"{len(df_static):,}")
    with col2:
        st.metric("Dynamic Data Sample Size", f"{len(df_dynamic_sample):,}")
    
    duplicates = df_static.duplicated().sum()
    if duplicates > 0:
        st.warning(f"{duplicates} duplicate records were detected in the static dataset.")
    else:
        st.success("No duplicate records were found in the static dataset.")

    duplicates = df_dynamic_sample.duplicated().sum()
    if duplicates > 0:
        st.warning(f"{duplicates} duplicate records were detected in the dynamic sample dataset.")
    else:
        st.success("No duplicate records were found in the dynamic sample dataset.")

st.markdown("### 3. Data Joining")
st.markdown(
    """
    In this step, the sampled dynamic data is merged with the consolidated static dataset using the 'VehId' key. 
    The results of the join operation, including the number of matched and unmatched records, are displayed below.
    """
)

with st.spinner("Joining datasets..."):
    df = read_parquet_from_s3(bucket_name, df)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records After Join", f"{len(df):,}")
    with col2:
        matched_records = df['VehId'].notna().sum()
        st.metric("Matched Records", f"{matched_records:,}")
    with col3:
        unmatched_records = df['VehId'].isna().sum()
        st.metric("Unmatched Records", f"{unmatched_records:,}")

st.markdown("### 4. Data Transformation")
st.markdown(
    """
    This section performs several data transformations, including:
    - Categorization of Outside Air Temperature (OAT)
    - Conversion of day numbers to datetime objects
    - Calculation of distance traveled
    - Computation of Fuel Consumption Rate (FCR) using available sensor data
    The results of these transformations are summarized and visualized below.
    """
)

with st.spinner("Transforming data..."):

    st.write("#### Transformation Results")
        
    st.write("**Distribution of Outside Air Temperature (OAT) Categories:**")
    oat_dist = df['OAT_Category'].value_counts()
    st.dataframe(oat_dist.reset_index().rename(columns={'index': 'OAT Category', 'OAT_Category': 'Count'}), use_container_width=True, hide_index=True)
        
    st.write("**Fuel Consumption Rate (FCR) Statistical Summary:**")
    fcr_stats = df['FCR'].describe()
    st.dataframe(fcr_stats.reset_index().rename(columns={'index': 'Statistic', 'FCR': 'Value'}), use_container_width=True, hide_index=True)    
            
    st.write("**HV Battery Power[Watts]:**")
    st.markdown(
                """
                The column **'HV Battery Power[Watts]'** is calculated as the product of **'HV Battery Voltage[V]'** and **'HV Battery Current[A]'** for each record in the dataset.
                This represents the instantaneous electrical power output (in Watts) of the high-voltage battery at each timestamp.
                \n
                **Formula:**  
                HV Battery Power [Watts] = HV Battery Voltage [V] × HV Battery Current [A]
                """
            )
            
    st.write("**Final Dataset Overview:**")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)
            
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    with st.container():
        st.text("Final Cleaned Data Info:")
        st.text(info_str) 


st.markdown("""
---
### 🔗 Navigation

- [Data Loading & Preprocessing](https://share.streamlit.io/your-username/ved-ml-tab1-dataloading)
- [Data Visualization - Sample Plots](https://share.streamlit.io/your-username/ved-ml-tab2-visualization)
- [Exploratory Data Analysis](https://share.streamlit.io/your-username/ved-ml-tab3-eda)
- [ICE, HEV, EV, and PHEV Analysis](https://share.streamlit.io/your-username/ved-ml-tab4-ice-hev-ev-phev)
""")