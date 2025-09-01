import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import boto3
from io import BytesIO

# -----------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------
st.set_page_config(page_title="EV Clustering App", layout='wide')
st.title("K-Means Clustering of EV Data (car_id vs regenwh)")

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
required_cols = ['car_id', 'regenwh']
missing = [col for col in required_cols if col not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

df.dropna(subset=required_cols, inplace=True)
df['car_id'] = df['car_id'].astype(str)  # Ensure car_id is string for grouping

# Prepare data for clustering (do NOT encode car_id)
# Use regenwh and a numeric representation of car_id for clustering, but do not show encoded values
# For clustering, we need numeric data, so use pd.factorize but do not add to df
car_id_numeric, car_id_uniques = pd.factorize(df['car_id'])
X = np.column_stack([car_id_numeric, df['regenwh'].values])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------------------------
# Sidebar Controls
# -----------------------------------------------------------
st.sidebar.header("K-Means Configuration")
k_clusters = st.sidebar.slider("Select Number of Clusters (K)", min_value=2, max_value=10, value=3)

# -----------------------------------------------------------
# Clustering
# -----------------------------------------------------------
kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# -----------------------------------------------------------
# Tabs
# -----------------------------------------------------------
tab1, tab2 = st.tabs(["1. Data Preview", "2. Cluster Visualization"])

# -----------------------------------------------------------
# Tab 1: Data Preview
# -----------------------------------------------------------
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df[['car_id', 'regenwh', 'cluster']].head(20))

    st.markdown("**Cluster Distribution**")
    st.bar_chart(df['cluster'].value_counts().sort_index())

# -----------------------------------------------------------
# Tab 2: Visualization
# -----------------------------------------------------------
with tab2:
    st.subheader("K-Means Cluster Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df['car_id'], df['regenwh'], c=df['cluster'], cmap='Set1', alpha=0.6, edgecolor='k')
    ax.set_xlabel("Car ID")
    ax.set_ylabel("Regenerated Energy (Wh)")
    ax.set_title("Clusters by Car ID and RegenWh")
    legend_labels = [f"Cluster {i}" for i in range(k_clusters)]
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          label=label, markersize=10,
                          markerfacecolor=col) 
               for label, col in zip(legend_labels, scatter.cmap(range(k_clusters)))]
    st.pyplot(fig)

    st.subheader("Cluster Grouping Details")
    # Group by cluster and aggregate
    cluster_summary = df.groupby('cluster').agg(
        Count=('regenwh', 'size'),
        Mean_RegenWh=('regenwh', 'mean'),
        Std_RegenWh=('regenwh', 'std'),
        Car_IDs=('car_id', lambda x: ', '.join(sorted(set(x))))
    ).reset_index()
    st.dataframe(cluster_summary)

    st.markdown("""
    **Interpretation Notes:**
    - Clustering is based on `car_id` and corresponding `regenwh` values.
    - Each color represents a distinct cluster determined by K-Means.
    - This view helps in discovering natural groupings among cars based on regenerative energy patterns.
    """)