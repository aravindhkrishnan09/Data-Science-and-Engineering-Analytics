import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import warnings
import shap
from datetime import datetime, timedelta
import matplotlib.pyplot as plt 
import seaborn as sns
warnings.filterwarnings('ignore')

st.set_page_config(page_title="VED ML Data Loading & Preprocessing", layout="centered", initial_sidebar_state="expanded")
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

tabs = st.tabs(["Data Loading and Preprocessing",
                 "Data Visualization - Sample Plots",
                 "Exploratory Data Analysis",
                 "ICE, HEV, EV, and PHEV Analysis"])

with tabs[0]:
    st.header("Data Loading and Preprocessing")
    
    st.markdown("### 1. Data Loading")
    st.markdown(
        """
        This section is responsible for importing both static and dynamic datasets required for further analysis. 
        Static data is loaded from Excel files, while dynamic data is aggregated from multiple CSV files within a specified directory.
        """
    )
    
    @st.cache_data
    def load_data_excel(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        return pd.read_excel(file_path)

    @st.cache_data
    def load_csv_files_from_directory(directory):
        all_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        np.random.seed(42)
        sampled_files = np.random.choice(all_files, size=int(len(all_files) * 0.5), replace=False)
        df_list = []
        for file in sampled_files:
            file_path = os.path.join(directory, file)
            df = pd.read_csv(file_path)
            df_list.append(df)
        return pd.concat(df_list, ignore_index=True)


    with st.spinner("Loading data..."):
        df_ICE_HEV = load_data_excel("G:\\DIYguru\\Notes and Sample Data\\VED-master\\Data\\VED_Static_Data_ICE&HEV.xlsx")
        df_PHEV_EV = load_data_excel("G:\\DIYguru\\Notes and Sample Data\\VED-master\\Data\\VED_Static_Data_PHEV&EV.xlsx")
        df_dynamic_sample = load_csv_files_from_directory("G:\\DIYguru\\Notes and Sample Data\\VED-master\\Data\\VED_DynamicData_Part1")
        
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
        df_ICE_HEV.replace('NO DATA', np.nan, inplace=True)
        df_PHEV_EV.replace('NO DATA', np.nan, inplace=True)
        df_ICE_HEV['Drive Wheels'] = df_ICE_HEV['Drive Wheels'].astype('object')
        df_PHEV_EV.rename(columns={'EngineType': 'Vehicle Type'}, inplace=True)
        df_static = pd.concat([df_ICE_HEV, df_PHEV_EV], ignore_index=True)

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
        df = df_dynamic_sample.merge(df_static, on='VehId', how='left')

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
        def categorize_oat(value):
            if value < -20: return 'Extremely Cold'
            elif -20 <= value < 0: return 'Cold'
            elif 0 <= value < 10: return 'Cool'
            elif 10 <= value < 20: return 'Mild'
            elif 20 <= value < 30: return 'Warm'
            elif value >= 30: return 'Hot'
            else: return np.nan

    df['OAT_Category'] = df['OAT[DegC]'].apply(categorize_oat)

    reference_date = datetime(2017, 11, 1)
    df['DateTime'] = pd.to_timedelta(df['DayNum'] - 1, unit='D') + reference_date
    df['Date'] = df['DateTime'].dt.date
    df['Time'] = df['DateTime'].dt.time

    df['Distance[km]'] = df['Vehicle Speed[km/h]'] * (df['Timestamp(ms)'] / 3600000)

    df['HV Battery Power[Watts]'] = df['HV Battery Voltage[V]'] * df['HV Battery Current[A]']

    # Constants
    AFR = 14.7  # typical AFR for gasoline engines
    ρ_air = 1.184  # air density in kg/m³

    def compute_fcr(df):
    # Parse displacement in liters from 'Engine Configuration & Displacement' if format like "I4 2.0L"
        def extract_displacement(val):
            try:
                return float(val.split()[-1].replace("L", ""))
            except:
                return np.nan

        df['Displacement_L'] = df['Engine Configuration & Displacement'].apply(extract_displacement)

        # Compute correction factor
        df['correction'] = (1 + df['Short Term Fuel Trim Bank 1[%]']/100 + df['Long Term Fuel Trim Bank 1[%]']/100) / AFR

        # Step 1: Use FuelRate if available
        df['FCR'] = np.where(
            ~df['Fuel Rate[L/hr]'].isna(),
            df['Fuel Rate[L/hr]'],
            np.nan
        )

        # Step 2: Else if MAF is available
        maf_condition = df['FCR'].isna() & ~df['MAF[g/sec]'].isna()
        df.loc[maf_condition, 'FCR'] = df.loc[maf_condition, 'MAF[g/sec]'] * df.loc[maf_condition, 'correction']

        # Step 3: Else if AbsLoad and RPMeng are available
        derived_condition = df['FCR'].isna() & ~df['Absolute Load[%]'].isna() & ~df['Engine RPM[RPM]'].isna() & ~df['Displacement_L'].isna()
        maf_derived = (df['Absolute Load[%]'] / 100) * ρ_air * df['Displacement_L'] * df['Engine RPM[RPM]'] / 120
        df.loc[derived_condition, 'FCR'] = maf_derived[derived_condition] * df.loc[derived_condition, 'correction']

        return df

    df = compute_fcr(df)

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


with tabs[1]:
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

with tabs[2]:
    st.markdown("""
    ### Battery Power, Fuel Consumption Rate (FCR), and Battery SOC
    The table below shows the mean values of HV Battery Power, Air Conditioning Power, Heater Power, HV Battery SOC, and FCR, grouped by OAT_Category and Vehicle Type.
    """)

    # Grouping and aggregating the data
    df_SOC = df.groupby(['OAT_Category', 'Vehicle Type'])[
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

    st.dataframe(df_la_lo.head(10), use_container_width=True)

    st.markdown("""
    **Comment:**  
    The table above provides a spatial summary of energy consumption and operational metrics by location and vehicle type, supporting the identification of geographic trends in vehicle performance and efficiency.
    """)

with tabs[3]:
    st.markdown("""
    ### ICE, HEV, EV, and PHEV Analysis: Distance vs FCR and HV Battery Power

    The tables below show a comparison of key metrics for different vehicle types:
    - **ICE & HEV**: Internal Combustion Engine and Hybrid Electric Vehicles
    - **EV & PHEV**: Electric Vehicles and Plug-in Hybrid Electric Vehicles

    The data is grouped by trip, time and vehicle type, and displays the top records for each group.
    """)

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

