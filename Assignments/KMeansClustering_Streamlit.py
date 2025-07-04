import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------
st.set_page_config(page_title="EV Clustering App", layout='wide')
st.title("K-Means Clustering of EV Data (car_id vs regenwh)")

# -----------------------------------------------------------
# Load All CSVs from Directory
# -----------------------------------------------------------
DATA_DIR = r"G:\DIY Guru\Data-Science-and-Engineering-Analytics\DEVRT\DEVRT\NISSAN LEAF"

@st.cache_data
def load_all_data(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    dfs = [pd.read_csv(os.path.join(directory, f)) for f in files]
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

df = load_all_data(DATA_DIR)

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
    ax.legend(handles=handles)
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