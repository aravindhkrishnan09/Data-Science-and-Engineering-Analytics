import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------
st.set_page_config(page_title="EV Regeneration Analyzer using SVM", layout='wide')
st.title("EV Regeneration Analyzer using Support Vector Machine")

st.markdown("""
## Machine Learning Workflow & Outcome

This Streamlit app performs the following steps:

1. Loads EV driving data from multiple CSV files.
2. Preprocesses the data to focus on energy regeneration events (negative regenwh).
3. Trains a linear Support Vector Machine regression model to predict regeneration energy.
4. Shows the location with maximum predicted regeneration.
5. Identifies and plots total predicted regeneration by car ID.
""")

# -----------------------------------------------------------
# Load Data
# -----------------------------------------------------------
DATA_DIR = r"G:\DIY Guru\Data-Science-and-Engineering-Analytics\DEVRT\DEVRT\NISSAN LEAF"

@st.cache_data
def load_data(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    dfs = [pd.read_csv(os.path.join(directory, f)) for f in files]
    return pd.concat(dfs, ignore_index=True)

df = load_data(DATA_DIR)

# -----------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------
required_cols = ['longitude', 'latitude', 'altitude', 'regenwh', 'speed']
missing = [col for col in required_cols if col not in df.columns]

if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

clean_df = df.dropna(subset=required_cols)
clean_df = clean_df[clean_df['regenwh'] < 0]  # Focus on regeneration events

if clean_df.empty:
    st.warning("No rows with negative regeneration values found.")
    st.stop()

# -----------------------------------------------------------
# Feature Engineering
# -----------------------------------------------------------
features = ['longitude', 'latitude', 'altitude', 'speed']
X = clean_df[features]
y = clean_df['regenwh']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# -----------------------------------------------------------
# Train SVM Model
# -----------------------------------------------------------
model = SVR(kernel='linear')
model.fit(X_train, y_train)

clean_df['regenwh_pred'] = model.predict(X_scaled)

# -----------------------------------------------------------
# Find Location with Max Regeneration
# -----------------------------------------------------------
max_idx = clean_df['regenwh_pred'].idxmin()  # Most negative predicted regen (max recovery)
max_row = clean_df.loc[max_idx]

st.subheader("Location of Maximum Predicted Regeneration")
st.write(
    f"**Latitude:** {max_row['latitude']} | **Longitude:** {max_row['longitude']} | "
    f"**Altitude:** {max_row['altitude']} | **Speed:** {max_row['speed']} | "
    f"**Predicted Regen (Wh):** {max_row['regenwh_pred']:.2f}"
)

# -----------------------------------------------------------
# Find car_id with Maximum Total Regeneration and Plot Bar Chart
# -----------------------------------------------------------
if 'car_id' not in clean_df.columns:
    st.warning("Column `car_id` not found in data, unable to compute maximum regeneration by vehicle.")
else:
    regen_by_car = clean_df.groupby('car_id')['regenwh_pred'].sum().sort_values()
    top_car = regen_by_car.idxmin()

    st.subheader("Car with Maximum Predicted Regeneration")
    st.write(
        f"**Car ID:** {top_car} | **Total Predicted Regeneration (Wh):** {regen_by_car.loc[top_car]:.2f}"
    )

    st.subheader("Regeneration Values by Car ID (Linear SVM)")

    fig, ax = plt.subplots(figsize=(12, 6))
    regen_by_car.plot(kind='bar', ax=ax, color='green')

    # Label each bar with its value
    for i, value in enumerate(regen_by_car):
        ax.text(i, value, f"{value:.1f}", ha='center', va='bottom' if value < 0 else 'top', fontsize=8, rotation=90)

    ax.set_ylabel('Total Predicted Regeneration (Wh)')
    ax.set_xlabel('Car ID')
    ax.set_title('Total Regeneration by Car (Predicted by Linear SVM)')
    ax.axhline(0, color='black', linewidth=0.8)

    st.pyplot(fig)
