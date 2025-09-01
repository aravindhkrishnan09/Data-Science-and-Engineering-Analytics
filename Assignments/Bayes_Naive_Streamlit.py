import os                    # For directory and file operations
import streamlit as st       # For building the web app UI
import pandas as pd          # For data manipulation
import numpy as np           # For numerical operations
from dotenv import load_dotenv
import boto3
from io import BytesIO
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.naive_bayes import GaussianNB            # Naive Bayes classifier
import matplotlib.pyplot as plt                       # For plotting

# -----------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------
st.set_page_config(page_title="EV Route Classifier",layout='wide')  # Set Streamlit page title and layout
st.title("EV Route Classifier using Naive Bayes")                    # Display the main title

# -----------------------------------------------------------
# Load data from s3 bucket
# -----------------------------------------------------------

# Load environment variables from .env
### Environment and Data Loading Setup

load_dotenv()

### S3 Bucket Configuration

bucket_name = 's3aravindh973515031797'
DATA_DIR = 'NISSAN LEAF/NISSAN_LEAF.parquet'

### AWS S3 Client Configuration

s3 = boto3.client("s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION")
)

### S3 Parquet File Reader

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

### Load Dataset

df = read_parquet_from_s3(bucket_name, DATA_DIR)

# -----------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------
required_cols = ['route_code', 'speed', 'soc', 'soh', 'latitude', 'longitude', 'timestamp_data_utc']  # Columns needed
missing = [col for col in required_cols if col not in df.columns]  # Check for missing columns
if missing:
    st.error(f"Missing required columns: {missing}")  # Show error if any columns are missing
    st.stop()                                        # Stop execution if columns are missing

df['timestamp_data_utc'] = pd.to_datetime(df['timestamp_data_utc'], errors='coerce')  # Convert timestamp to datetime
df.dropna(subset=required_cols + ['timestamp_data_utc'], inplace=True)                # Drop rows with missing values

df['route_code'] = df['route_code'].astype('category')        # Convert route_code to categorical
df['route_code_cat'] = df['route_code'].cat.codes             # Encode categories as numbers

features = ['speed']                  # Use only speed as feature
X = df[features]                      # Feature matrix
y = df['route_code_cat']              # Target vector

# -----------------------------------------------------------
# Train Model
# -----------------------------------------------------------

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data
    model = GaussianNB()                  # Instantiate Gaussian Naive Bayes
    model.fit(X_train, y_train)           # Train model
    return model

model = train_model(X, y)                 # Train and cache the model

# -----------------------------------------------------------
# Tabs
# -----------------------------------------------------------
tab1, tab2 = st.tabs(["1. Data Preview", "2. Predict and Visualize"])  # Create two tabs

# -----------------------------------------------------------
# Tab 1: Preview
# -----------------------------------------------------------
with tab1:
    st.subheader("Dataset Preview")                       # Subheader
    st.dataframe(df.head(20))                             # Show first 20 rows
    st.markdown("**Route Code Frequencies**")             # Markdown for frequency table
    st.dataframe(df['route_code'].value_counts().reset_index().rename(
        columns={'index': 'Route Code', 'route_code': 'Count'}))  # Show route code counts

# -----------------------------------------------------------
# Tab 2: Prediction + Visualizations (Speed only input + SOC & SOH plots)
# -----------------------------------------------------------
with tab2:
    st.subheader("Manual Prediction")                     # Subheader

    st.markdown("Select speed range to generate a prediction:")  # Instructions

    speed_min, speed_max = float(df['speed'].min()), float(df['speed'].max())  # Get min/max speed
    speed_from, speed_to = st.slider(
        "Speed range",
        min_value=speed_min,
        max_value=speed_max,
        value=(speed_min + (speed_max - speed_min) * 0.25, speed_min + (speed_max - speed_min) * 0.75),
        step=0.1
    )  # Slider for speed range

    if st.button("Predict Route"):                        # Button to trigger prediction
        speed_mean = np.mean([speed_from, speed_to])      # Use mean of selected speed range

        input_data = {'speed': speed_mean}                # Prepare input dict
        input_df = pd.DataFrame([input_data])             # Convert to DataFrame

        pred_class = model.predict(input_df)[0]           # Predict class (numeric)
        pred_label = df['route_code'].cat.categories[pred_class]  # Map to original label

        pred_proba = model.predict_proba(input_df)[0]     # Get probabilities for all classes
        proba_df = pd.DataFrame({
            'Route Code': df['route_code'].cat.categories,
            'Probability': pred_proba
        }).sort_values(by='Probability', ascending=False).reset_index(drop=True)  # Sort by probability

        st.success(f"Predicted Route Code: {pred_label}") # Show prediction

        st.subheader("Prediction Probabilities for Each Route Code")  # Subheader
        st.dataframe(proba_df)                                       # Show probabilities

        st.subheader("How the Naive Bayes Model Predicts the Route Code")  # Explanation
        st.markdown("""
        This prediction is based on the Naive Bayes algorithm, which works as follows:

        1. **Data Preparation:**
           - The model uses historical EV data including speed only in this version.
        
        2. **Feature Independence Assumption:**
           - Naive Bayes assumes all features contribute independently to the outcome. Since only speed is used, this simplifies to likelihood based on speed.
        
        3. **Probability Calculation:**
           - For each route code, the model calculates the likelihood of the input speed assuming Gaussian distributions.
           - It multiplies these likelihoods by the prior probability of each route.
              
        4. **Prediction:**
           - The route with the highest posterior probability is chosen as the predicted route.

        5. **Visualization:**
           - After prediction, data for the predicted route is displayed on the map and visualized over time.
        """)

        route_df = df[df['route_code'] == pred_label].sort_values('timestamp_data_utc')  # Filter for predicted route

        st.subheader(f"Map for Predicted Route: {pred_label}")      # Subheader
        st.map(route_df[['latitude', 'longitude']])                 # Show route on map

        st.subheader("Feature Trends Over Time for Predicted Route")  # Subheader

        fig, axs = plt.subplots(3, 1, figsize=(8,10))   # Create 3 subplots

        # Prepare X for regression (seconds since start)
        X_time = (route_df['timestamp_data_utc'] - route_df['timestamp_data_utc'].min()).dt.total_seconds().values.reshape(-1, 1)

        # Polynomial features (degree 2)
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X_time)

        # --- SOC plot ---
        y_soc = route_df['soc'].values
        # Linear regression
        soc_lin_reg = LinearRegression().fit(X_time, y_soc)
        soc_lin_pred = soc_lin_reg.predict(X_time)
        # Polynomial regression
        soc_poly_reg = LinearRegression().fit(X_poly, y_soc)
        soc_poly_pred = soc_poly_reg.predict(X_poly)
        axs[0].plot(route_df['timestamp_data_utc'], y_soc, color='blue', label='SOC')
        axs[0].plot(route_df['timestamp_data_utc'], soc_lin_pred, color='red', linestyle='--', label='Linear Regression')
        axs[0].plot(route_df['timestamp_data_utc'], soc_poly_pred, color='orange', linestyle='-.', label='Polynomial Regression (deg 2)')
        axs[0].set_title("SOC Over Time")
        axs[0].set_xlabel("Timestamp")
        axs[0].set_ylabel("SOC")
        axs[0].tick_params(axis='x', rotation=45)
        axs[0].legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=1)

        # --- Speed plot ---
        y_speed = route_df['speed'].values
        speed_lin_reg = LinearRegression().fit(X_time, y_speed)
        speed_lin_pred = speed_lin_reg.predict(X_time)
        speed_poly_reg = LinearRegression().fit(X_poly, y_speed)
        speed_poly_pred = speed_poly_reg.predict(X_poly)
        axs[1].plot(route_df['timestamp_data_utc'], y_speed, color='green', label='Speed')
        axs[1].plot(route_df['timestamp_data_utc'], speed_lin_pred, color='red', linestyle='--', label='Linear Regression')
        axs[1].plot(route_df['timestamp_data_utc'], speed_poly_pred, color='orange', linestyle='-.', label='Polynomial Regression (deg 2)')
        axs[1].set_title("Speed Over Time")
        axs[1].set_xlabel("Timestamp")
        axs[1].set_ylabel("Speed")
        axs[1].tick_params(axis='x', rotation=45)
        axs[1].legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=1)

        # --- SOH plot ---
        y_soh = route_df['soh'].values
        soh_lin_reg = LinearRegression().fit(X_time, y_soh)
        soh_lin_pred = soh_lin_reg.predict(X_time)
        soh_poly_reg = LinearRegression().fit(X_poly, y_soh)
        soh_poly_pred = soh_poly_reg.predict(X_poly)
        axs[2].plot(route_df['timestamp_data_utc'], y_soh, color='purple', label='SOH')
        axs[2].plot(route_df['timestamp_data_utc'], soh_lin_pred, color='red', linestyle='--', label='Linear Regression')
        axs[2].plot(route_df['timestamp_data_utc'], soh_poly_pred, color='orange', linestyle='-.', label='Polynomial Regression (deg 2)')
        axs[2].set_title("SOH Over Time")
        axs[2].set_xlabel("Timestamp")
        axs[2].set_ylabel("SOH")
        axs[2].tick_params(axis='x', rotation=45)
        axs[2].legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=1)

        plt.tight_layout()                         # Adjust layout
        st.pyplot(fig)  