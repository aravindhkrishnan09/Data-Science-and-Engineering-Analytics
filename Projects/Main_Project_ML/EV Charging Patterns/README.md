# EV Charging Demand Forecasting

## Project Overview

This project involves an advanced machine learning analysis of electric vehicle (EV) charging patterns across California stations from 2021-2024. The primary goal is to predict demand patterns and optimize grid infrastructure planning by leveraging neural networks, decision trees, and SHAP interpretability frameworks.

## Objective

The main objective is to develop robust predictive models for EV charging demand, understand the key factors influencing these patterns, and provide strategic recommendations for sustainable grid management.

## Dataset Overview

### Data Characteristics

*   **Observations**: 59,810 hourly observations collected over 3.5 years.
*   **Stations**: Data combined from two different California charging stations.
*   **Features**: 24 features encompassing energy metrics, pricing, infrastructure details, and environmental factors.
*   **Granularity**: Hourly measurements, including seasonal patterns.

### Key Variables

*   **Target**: EV Charging Demand (kW)
*   **Renewable Energy Production**: Includes solar and wind energy generation (kW).
*   **Grid Metrics**: Grid stability index and availability.
*   **Environmental Factors**: Weather conditions and temporal features (e.g., hour, day period, season).

## Methodology

The project utilizes a combination of data preprocessing, feature engineering, and machine learning models:

1.  **Data Loading**: Data is loaded from an AWS S3 bucket (`s3aravindh973515031797`) using `boto3` and `pandas.read_parquet`.
2.  **Preprocessing & Feature Engineering**:
    *   'Date' column is converted to datetime format.
    *   'Time' column is processed to extract the 'Hour'.
    *   A 'Day Period' column is created, categorizing hours into 'Night', 'Morning', 'Afternoon', and 'Evening'.
    *   A 'Season' column is added based on the month ('Winter', 'Spring', 'Summer', 'Autumn').
    *   Key numerical features such as 'EV Charging Efficiency (%)', 'EV Charging Demand (kW)', and 'Total Renewable Energy Production (kW)' are normalized using `MinMaxScaler` and `StandardScaler` for model compatibility.
3.  **Exploratory Data Analysis (EDA)**: Initial visualizations are generated to understand trends and relationships within the data.
4.  **Machine Learning Models**:
    *   **Neural Networks (MLP)**: Used for precise demand forecasting.
    *   **Decision Trees**: Employed for interpretable classification tasks, such as predicting day periods and seasons.
    *   **SHAP Framework**: Applied for feature importance analysis to explain model predictions.

## Key Findings & Insights

### Seasonal Demand Patterns

*   **Winter**: Highest evening demand, moderate morning usage. Lower renewable production increases grid dependency.
*   **Spring**: Balanced demand across day periods. Improved renewable energy availability supports efficient charging.
*   **Summer**: Peak afternoon demand coinciding with solar production. Lowest evening renewable output.
*   **Autumn**: Stable nighttime demand with moderate fluctuations. Transitional renewable energy patterns.

### Hourly Demand Distribution

*   **Nighttime (0-6h)**: Lowest demand (0.35-0.45 normalized). Ideal for grid-friendly charging.
*   **Morning (6-12h)**: Gradual increase (0.50-0.55 normalized) as commuters charge. Solar production ramps up.
*   **Afternoon (12-18h)**: Peak demand (0.60-0.65 normalized). Maximum solar availability.
*   **Evening (18-24h)**: Secondary peak (0.55-0.60 normalized). Grid pressure intensifies as solar declines.

### Renewable Energy Production

*   Nighttime consistently shows the highest renewable energy production across seasons, especially in summer and spring, due to strong wind generation compensating for solar absence.

### Weather Impact on Charging Efficiency

*   **Sunny Conditions**: Highest efficiency (0.53 normalized).
*   **Cloudy Weather**: Moderate efficiency (0.50 normalized).
*   **Partly Cloudy**: Moderate efficiency (0.49 normalized).
*   **Rainy Periods**: Lower efficiency (0.48 normalized).
*   **Clear Skies**: Lowest efficiency (0.46 normalized).

### SHAP Feature Importance

*   **Renewable Energy Usage (%)**: Dominant predictor, higher usage correlates with reduced grid demand.
*   **Electricity Price ($/kWh)**: Second most influential, price signals drive user behavior.
*   **Carbon Emissions (kgCO2/kWh)**: Third key driver, higher emissions indicate grid stress.
*   **Adjusted Charging Demand (kW)**: Engineered feature capturing optimized load patterns.

## Model Performance

### Neural Network (MLP)

*   **R² Score**: 0.9978 (99.78% variance explained)
*   **MAE (kW)**: 0.0032 (3.2 watts mean absolute error)
*   **RMSE (kW)**: 0.0040 (4.0 watts root mean squared error)
*   **Training Epochs**: 13 (rapid convergence)

### Decision Trees

*   **Day Period Classification**: 100% accuracy using month and hour features.
*   **Seasonal Classification**: 100% accuracy using month (dominant feature), hour, charging demand, and electricity price.
*   **Grid Availability Prediction**: 95% accuracy using electricity price, hour, month, and grid stability index.

## Strategic Recommendations

*   **Maximize Renewable Integration**: Prioritize EV charging during high renewable availability windows (nighttime and morning).
*   **Dynamic Pricing Incentives**: Implement time-of-use rates to shift demand to off-peak hours.
*   **Smart Charging Optimization**: Utilize demand response programs and adjusted charging demand metrics to optimize load distribution and minimize peak stress.
*   **Carbon Footprint Reduction**: Target low-emission charging windows and invest in cleaner technologies.

## Technical Details

### Libraries Used

*   `pandas`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `scikit-learn` (for `MinMaxScaler`, `StandardScaler`)
*   `boto3` (for S3 interaction)
*   `dotenv` (for environment variable management)

### How to Run

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd EV Charging Patterns
    ```
2.  **Set up environment variables**: Ensure your AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`) are configured, preferably using a `.env` file.
3.  **Install dependencies**:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn boto3 python-dotenv
    ```
4.  **Run the Jupyter Notebook**: Open and execute `EV_Charging_Patterns.ipynb` to load data, perform preprocessing, generate visualizations, and train models.
    ```bash
    jupyter notebook EV_Charging_Patterns.ipynb
    ```

## Visualizations

The notebook includes several visualizations to illustrate charging patterns and feature impacts:

*   Average EV Charging Efficiency over time.
*   Average EV Charging Efficiency by Weather Condition.
*   Average EV Charging Demand by Period.
*   Average Total Renewable Energy Production by Season and Period.
*   Hourly Normalized EV Charging Demand by Month.
*   Hourly Normalized EV Charging Demand by Season.
