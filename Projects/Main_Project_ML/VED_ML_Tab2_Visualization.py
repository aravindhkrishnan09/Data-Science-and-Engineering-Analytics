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

st.set_page_config(page_title="VED ML Data Visualization", layout="centered", initial_sidebar_state="expanded")
st.title('VED ML Data Modelling - Data Visualization')

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

with st.spinner("Loading data for visualization..."):
    bucket_name = 's3aravindh973515031797'
    df = read_parquet_from_s3(bucket_name, 'Cleaned up VED Source Data/df_VED.parquet')

# --- Tab 2 Visualizations ---
st.markdown("""
### Average Distance Travelled by Vehicle Type
Calculate and display the average distance travelled for each vehicle type.
This is done by grouping the DataFrame by 'Vehicle Type' and computing the mean of 'Distance[km]'.
The result is then sorted in descending order of mean distance.
""")

df_distance = df.groupby(['Vehicle Type'])['Distance[km]'].mean().reset_index().sort_values(by='Distance[km]', ascending=False)

st.markdown("""
Plot a bar chart showing the average distance travelled by each vehicle type.
""")
fig, ax = plt.subplots(figsize=(5, 3))
bars = ax.bar(df_distance['Vehicle Type'], df_distance['Distance[km]'], alpha=0.7, color='blue')
ax.set_title('Average Distance Travelled by Vehicle Types')
ax.set_xlabel('Vehicle Type')
ax.set_ylabel('Mean Distance (km)')
ax.set_ylim(0, df_distance['Distance[km]'].max() * 1.1)
for index, value in enumerate(df_distance['Distance[km]']):
    ax.text(index, value, f"{value:.2f}", ha='center', va='bottom')
st.pyplot(fig)

st.markdown("""
### Average HV Battery Voltage Over Time
Calculate the average HV Battery Voltage for each day.
""")

df['Date'] = pd.to_datetime(df['Date'])
df_eot = df.groupby(df['Date'].dt.to_period('D'))['HV Battery Voltage[V]'].mean().reset_index()
df_eot['Date'] = df_eot['Date'].dt.to_timestamp()
df_eot = df_eot.sort_values(by='Date')

st.markdown("""
Plot the average HV Battery Voltage over time.
""")
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(df_eot['Date'], df_eot['HV Battery Voltage[V]'], marker='o', linestyle='-', color='blue', alpha=0.7)
ax.set_title('Average HV Battery Voltage Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Mean HV Battery Voltage (V)')
st.pyplot(fig)

st.markdown("""
### Average Fuel Consumption Rate (FCR) Over Time
Calculate the average Fuel Consumption Rate (FCR) for each day.
""")

df['Date'] = pd.to_datetime(df['Date'])
df_eof = df.groupby(df['Date'].dt.to_period('D'))['FCR'].mean().reset_index()
df_eof['Date'] = df_eof['Date'].dt.to_timestamp()
df_eof = df_eof.sort_values(by='Date')

st.markdown("""
Plot the average Fuel Consumption Rate (FCR) over time.
""")
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(df_eof['Date'], df_eof['FCR'], marker='o', linestyle='-', color='blue', alpha=0.7)
ax.set_title('Average Fuel Rate Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Mean Fuel Rate (L/hr)')
st.pyplot(fig)

st.markdown("""
### Average Trip Count by Location (Heat Map)
The following visualization shows the average trip count by location for each vehicle type using a scatter plot on subplots.
It iterates over unique vehicle types, filters the data for each type, and plots longitude and latitude with color representing the logarithm of the average trip count.
Subplot titles and axis labels are set, and a colorbar is added for each subplot.
The layout is adjusted for clarity.
""")

df_map = df.groupby(['Latitude[deg]', 'Longitude[deg]','Vehicle Type'])['Trip'].mean().reset_index()
vehicle_types = df_map['Vehicle Type'].unique()
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
axes = axes.flatten()

for idx, vtype in enumerate(vehicle_types):
    ax = axes[idx]
    data = df_map[df_map['Vehicle Type'] == vtype]
    sc = ax.scatter(
        data['Longitude[deg]'],
        data['Latitude[deg]'],
        c=np.log(data['Trip']),
        marker='o',
        s=10,
        cmap='RdYlGn_r'
    )
    ax.set_title(f'Average Trip Count: {vtype}')
    ax.set_xlabel('Longitude (deg)')
    ax.set_ylabel('Latitude (deg)')
    plt.colorbar(sc, ax=ax, label='Log(Average Trip Count)')
st.pyplot(fig)

st.markdown("""
### Average Trips Over Time
The following plot shows the average number of trips over time using a line plot.
The data is grouped by month and the average trips per month are calculated and visualized.
""")

df['Date'] = pd.to_datetime(df['Date'])
df_trip = df.groupby(df['Date'].dt.to_period('M'))['Trip'].mean().reset_index()
df_trip['Date'] = df_trip['Date'].dt.to_timestamp()
df_trip = df_trip.sort_values(by='Date')

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(df_trip['Date'], df_trip['Trip'], marker='o', linestyle='-', color='blue', alpha=0.7)
ax.set_title('Average Trips Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Average Trips')
st.pyplot(fig) 

st.markdown("""
---
### 🔗 Navigation

- [Data Loading & Preprocessing](https://share.streamlit.io/your-username/ved-ml-tab1-dataloading)
- [Data Visualization - Sample Plots](https://share.streamlit.io/your-username/ved-ml-tab2-visualization)
- [Exploratory Data Analysis](https://share.streamlit.io/your-username/ved-ml-tab3-eda)
- [ICE, HEV, EV, and PHEV Analysis](https://share.streamlit.io/your-username/ved-ml-tab4-ice-hev-ev-phev)
""")