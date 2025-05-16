import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Set page config
st.set_page_config(
    page_title="Vehicle Energy Dataset (VED) Summary",
    page_icon="🚗",
)

# Title
st.title("Vehicle Energy Dataset (VED) Summary")

# Define a primary color for consistent styling
primary_color = '#1f77b4'  # Default blue color

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Abstract", "Introduction", "VED Details", 
    "Privacy Protection", "Driver Behavior", 
    "Eco-Driving", "Conclusion"
])

# Tab 1: Abstract
with tab1:
    st.header("Abstract")
    st.markdown("""
    - **VED** is a large-scale dataset of fuel and energy data from **383 personal cars** in Ann Arbor, Michigan
    - Includes:
        - GPS routes
        - Speed data
        - Fuel/energy use
        - Auxiliary power (like AC/heater usage)
    - Fleet mix:
        - 264 gasoline cars
        - 92 hybrids (HEVs)
        - 27 plug-in hybrids/EVs (PHEVs/EVs)
    - Data collected via **OBD-II loggers** from **Nov 2017–Nov 2018**
    - Covers **374,000 miles** of driving
    - Used for research on:
        - Fuel efficiency
        - Driver behavior
        - Eco-driving techniques
        - Self-driving car development
    - Dataset is **publicly available** [here](https://github.com/gsoh/VED)
    """)

    st.warning(
    """
    > **⚠️ Disclaimer:**  
    > _The actual dataset (approximately 3 GB in size) was not used to generate the visualizations due to its large volume
    > and lack of generalized columns for calculating metrics based on seasons and driving environments. 
    > Instead, a manipulated subset of data (generated in python script) was used for demonstration purposes._
    """
    )

# Tab 2: Introduction
with tab2:
    st.header("Introduction")
    st.markdown("""
    - Most existing fuel data comes from **lab tests**, but real-world driving differs due to:
        - Traffic conditions
        - Weather variations
        - Road conditions
    - Existing traffic datasets from Government lacks:
        - Individual vehicle energy data
        - Detailed driving patterns
    - VED fills this gap by providing:
        - Real-world driving patterns
        - Actual energy use
        - Data for diverse vehicle types
    """)

# Tab 3: VED Details
with tab3:
    st.header("Vehicle Energy Dataset (VED) Details")
    
    st.subheader("A. Data Collection")
    st.markdown("""
    - Data logged via **OBD-II devices**
    - Tracked:
        - GPS location
        - Speed
        - Fuel consumption
        - Battery data (for hybrids/EVs)
    - Recruited hybrid/EV owners
    - Fleet included various vehicle types:
        - Sedans
        - SUVs
        - Trucks
    """)
    
    st.subheader("B. Data Description")
    st.markdown("""
    - **Static data** (vehicle specs):
        - Model
        - Weight
        - Engine type
    - **Dynamic data** (real-time driving info):
        - GPS signals (latitude/longitude)
        - Standard OBD-II signals:
            - Speed
            - Engine stats
            - Outside temperature
        - OEM-specific signals (for hybrids/EVs):
            - Battery voltage/current
            - AC/heater power
    """)
    
    st.subheader("C. Fuel Consumption Estimation")
    st.markdown("""
    - Fuel data isn't always directly logged, so it's calculated from:
        - Airflow measurements
        - Engine load
        - RPM
        - Fuel-trim adjustments
    - For EVs, energy use calculated as:
        - Battery voltage × current
    """)

# Tab 4: Privacy Protection
with tab4:
    st.header("Privacy Protection (De-identification)")
    st.markdown("""
    - GPS data could reveal sensitive locations (homes/workplaces)
    - So it was anonymized by:
    
    1. **Random-fogging**:
        - Hide start/end points of trips
    
    2. **Geo-fencing**:
        - Only keep data from urban areas
        - Removes rural areas with fewer roads
    
    3. **Major intersections bounding**:
        - Remove data near trip starts/ends
        - Only keep data after first major intersection and before last one
    
    - Vehicle details were generalized
    
    """)

# Tab 5: Driver Behavior
with tab5:
    st.header("Driver Behavior & Fuel Consumption")
    
    st.subheader("A. Driving Environments")
    st.markdown("""
    - **Highways**:
        - Best fuel economy (steady speed, fewer stops)
        - ~2× better than city driving
    - **Urban roads**:
        - Lower fuel economy (traffic lights, speed changes)
    - **Downtown**:
        - Worst fuel economy (heavy traffic, frequent stops)
    - Key finding: 50% of fuel near intersections (which cover only 29% of distance)
    """)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
    # Data
    environments = ['Highway', 'Urban Road', 'Downtown']
    mpg = [32, 18, 12]
    stops_per_mile = [0.2, 2.5, 6.0]
    colors = ['#2ca02c', '#ff7f0e', '#d62728']
        
    # Plot 1: MPG by Environment
    bars = ax1.bar(environments, mpg, color=colors)
    ax1.bar_label(bars, padding=1)
    ax1.set_title("Fuel Economy by Driving Environment", pad=20)
    ax1.set_ylabel("Fuel Economy (MPG)")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Stops vs MPG
    ax2.scatter(stops_per_mile, mpg, s=100, c=colors)
    ax2.set_title("Fuel Economy vs Stop Frequency", pad=20)
    ax2.set_ylabel("Fuel Economy (MPG)")
    ax2.set_xlabel("Stops per Mile")
    # Add legend instead of annotate
    for i, env in enumerate(environments):
        ax2.scatter([], [], color=colors[i], label=env)
    ax2.legend(title="Environment")
    ax2.grid(True, alpha=0.3)

    st.pyplot(fig)
    st.markdown("""
    **Key Findings Confirmed:**
    - Highways show best MPG (32) with few stops
    - Urban roads moderate (18 MPG)
    - Downtown worst (12 MPG) with frequent stops
    - Strong correlation between stops and reduced MPG
    """)
    
    st.subheader("B. Speed vs. Fuel Economy")
    st.markdown("""
    - Best fuel efficiency at **35–45 mph** (16–20 m/s)
    - Efficiency drops at:
        - Very low speeds (frequent acceleration)
        - Very high speeds (air resistance increases)
    """)

    fig, ax = plt.subplots(figsize=(10, 6))
        
    # Data
    speeds = np.arange(10, 80, 2)
    mpg = 40 * np.exp(-0.5 * ((speeds - 40)/15)**2)  # Gaussian curve
        
    # Plot
    ax.plot(speeds, mpg, linewidth=3, color=primary_color)
    ax.axvspan(35, 45, color='green', alpha=0.1, label='Optimal Range')
    ax.annotate('Peak Efficiency\n(35-45 mph)', xy=(40, 38), xytext=(40, 30), ha='center',
                arrowprops=dict(arrowstyle='->'))
    ax.set_title("Fuel Economy vs Vehicle Speed", pad=20)
    ax.set_ylabel("Fuel Economy (MPG)")
    ax.set_xlabel("Speed (mph)")
    ax.grid(True, alpha=0.3)
    ax.legend()
        
    st.pyplot(fig)
    st.markdown("""
    **Key Findings Confirmed:**
    - Peak efficiency between 35-45 mph (16-20 m/s)
    - Sharp decline at higher speeds (aerodynamic drag)
    - Reduced efficiency at very low speeds (engine inefficiency)
    """)
    
    st.subheader("C. Seasons & Temperature")
    st.markdown("""
    - **Winter**:
        - 25% worse fuel economy
        - Reasons:
            - Cold engines less efficient
            - Heater use increases load
    - **Summer**:
        - Better efficiency
        - Warmer temps help engine performance
    - Extreme cold (−8°C) uses 46% more fuel than mild temps (20°C)
    """)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
    # Data
    temps = [-8, 0, 10, 20, 30]
    mpg = [18, 22, 26, 33, 31]
    seasons = ['Winter\n(Extreme)', 'Winter\n(Mild)', 'Spring/Fall', 'Summer\n(Ideal)', 'Summer\n(Hot)']
        
    # Plot 1: Temperature vs MPG
    ax1.plot(temps, mpg, marker='o', markersize=8, color=primary_color)
    ax1.annotate('46% worse\nthan 20°C', xy=(-8, 18), xytext=(0, 19),
                arrowprops=dict(arrowstyle='->'))
    ax1.set_title("Fuel Economy vs Ambient Temperature", pad=20)
    ax1.set_ylabel("Fuel Economy (MPG)")
    ax1.set_xlabel("Temperature (°C)")
    ax1.grid(True, alpha=0.3)
        
    # Plot 2: Seasonal comparison
    bars = ax2.bar(seasons[1:4], mpg[1:4], color=['#1f77b4', '#2ca02c', '#ff7f0e'])
    ax2.bar_label(bars, padding=1)
    ax2.set_title("Seasonal Comparison (Excluding Extremes)", pad=20)
    ax2.set_ylabel("Fuel Economy (MPG)")
    ax2.grid(True, alpha=0.3)
        
    st.pyplot(fig)
    st.markdown("""
    **Key Findings Confirmed:**
    - 25% worse MPG in winter vs summer
    - Extreme cold (-8°C) shows 46% reduction vs 20°C
    - Peak efficiency around mild temperatures (20°C)
    - Slight decrease in very hot weather (AC load)
    """)

    st.subheader("D. Time of Day")
    st.markdown("""
    - **Rush hour (4–6 PM)**:
        - Low fuel economy
        - Traffic jams cause stop-and-go driving
    - **Late night (9 PM–6 AM)**:
        - Better efficiency
        - Free-flowing traffic allows steady speeds
    """)

    fig, ax = plt.subplots(figsize=(10, 6))
        
    # Data
    hours = [0,3,6,9,12,15,18,21]
    mpg = [30, 32, 28, 26, 25, 24, 20, 28]
    labels = ['Late Night', 'Late Night', 'Morning', 'Day', 'Day', 'Evening', 'Rush Hour', 'Night']
        
    # Plot
    ax.plot(hours, mpg, marker='o', markersize=8, color=primary_color)
    ax.axvspan(16, 18, color='red', alpha=0.2, label='Rush Hour')
    ax.axvspan(21, 6, color='green', alpha=0.1, label='Free Flow')
        
    # Highlight rush hour
    rush_idx = 6
    ax.annotate('Lowest MPG (20)\n4-6 PM', xy=(hours[rush_idx], mpg[rush_idx]), 
                xytext=(13, 20), arrowprops=dict(arrowstyle='->'))
        
    ax.set_title("Fuel Economy by Time of Day", pad=20)
    ax.set_ylabel("Fuel Economy (MPG)")
    ax.set_xlabel("Hour of Day")
    ax.set_xticks(hours)
    ax.legend()
    ax.grid(True, alpha=0.3)
        
    st.pyplot(fig)
    st.markdown("""
    **Key Findings Confirmed:**
    - Rush hour (4-6 PM) shows lowest MPG (20)
    - Late night (9 PM-6 AM) shows best efficiency (30-32 MPG)
    - Inverse relationship with traffic density
    """)
    
    st.subheader("E. Gas vs. Hybrid Vehicles")
    st.markdown("""
    - **Hybrids (HEVs)**:
        - 2.9× better fuel economy than gas cars in cities
        - Regenerative braking recovers energy
    - **Plug-in Hybrids (PHEVs)**:
        - Can run on electric-only mode
        - Achieved 138 MPGe (miles per gallon equivalent)
        - Battery recharges during deceleration
    """)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
    # Data
    vehicles = ['Gasoline', 'Hybrid (HEV)', 'Plug-in Hybrid (PHEV)']
    city_mpg = [18, 52, 138]
    colors = ['#d62728', '#2ca02c', '#9467bd']
        
    # Plot 1: City Comparison
    bars = ax1.bar(vehicles, city_mpg, color=colors)
    ax1.bar_label(bars, padding=1)
    ax1.annotate('2.9× better\nthan gas', xy=(1, 60), xytext=(1, 65),
                arrowprops=dict(arrowstyle='->'), ha='center')
    ax1.set_title("City Fuel Economy Comparison", pad=20)
    ax1.set_ylabel("Fuel Economy (MPG)")
    ax1.grid(True, alpha=0.3)
        
    # Plot 2: PHEV electric mode
    ax2.text(0.5, 0.5, '138 MPGe', fontsize=40, ha='center', va='center', color='purple')
    ax2.text(0.5, 0.3, 'PHEV Electric Mode', fontsize=12, ha='center', va='center')
    ax2.axis('off')
    ax2.set_facecolor('lavender')
        
    st.pyplot(fig)
    st.markdown("""
    **Key Findings Confirmed:**
    - HEVs show 2.9× better city MPG than gas (52 vs 18)
    - PHEVs achieve 138 MPGe in electric mode
    - Regenerative braking significantly improves city efficiency
    """)        
    
# Tab 6: Eco-Driving
with tab6:
    st.header("Eco-Driving Applications")
    
    st.subheader("A. Eco-Routing")
    st.markdown("""
    - Choosing routes to **save fuel**:
        - Avoid traffic lights
        - Prefer steady-speed roads
    - VED helps build better fuel models for:
        - Navigation apps
        - Traffic simulators
    - Example: Some routes can save 20% fuel with similar travel time
    """)
    
    st.subheader("B. Eco-Approach at Intersections")
    st.markdown("""
    - Techniques to save fuel at traffic lights:
        - Smooth braking/acceleration
        - Coasting instead of hard stops
    - Can cut fuel use by **20%** at intersections
    - Real-world data shows benefits of:
        - Early deceleration
        - Maintaining momentum
    """)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Data
    approaches = ['Hard Braking', 'Moderate', 'Coasting', 'Eco-Approach']
    fuel_used = [320, 280, 240, 200]
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#2ca02c']
    
    # Plot 1: Intersection approaches
    bars = ax1.bar(approaches, fuel_used, color=colors)
    ax1.bar_label(bars, padding=1)
    ax1.annotate('20% savings', xy=(3, 220), xytext=(3, 240),
                arrowprops=dict(arrowstyle='->'), ha='center')
    ax1.set_title("Fuel Use by Braking Approach", pad=20)
    ax1.set_ylabel("Fuel Used (mL)")
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Route comparison
    routes = ['Standard', 'Eco-Route']
    route_mpg = [24, 29]
    bars = ax2.bar(routes, route_mpg, color=['#ff7f0e', '#2ca02c'])
    ax2.bar_label(bars, padding=1)

    ax2.set_title("Eco-Routing Benefits", pad=20)
    ax2.set_ylabel("Fuel Economy (MPG)")
    ax2.grid(True, alpha=0.3)
    
    st.pyplot(fig)

    st.markdown("""
    **Key Findings Confirmed:**
    - Eco-approach saves 20% fuel at intersections
    - Coasting beats hard braking (240mL vs 320mL)
    - Eco-routes can improve MPG by 21% with minimal time penalty
    """)

# Tab 7: Conclusion
with tab7:
    st.header("Conclusion")
    st.markdown("""
    - VED is the **first large-scale dataset** linking:
        - Real-world driving patterns
        - Actual energy consumption
    - Key findings:
        - **Hybrids/EVs** are far more efficient in cities
        - **Cold weather** significantly hurts fuel economy
        - **Intersections** are major fuel-wasting spots
    - Applications:
        - Improving eco-driving techniques
        - Calibrating traffic simulators
        - Developing better self-driving car algorithms
    - Dataset available for research at: [github.com/gsoh/VED](https://github.com/gsoh/VED)
    """)