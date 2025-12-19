# Data Science and Engineering Analytics

## 📌 Overview

This repository serves as a comprehensive portfolio and learning resource for **Data Science**, **Machine Learning**, and **Engineering Analytics**. It houses a collection of projects, assignments, and training notes, ranging from fundamental statistical analysis to advanced deep learning applications in predictive maintenance.

The centerpiece of this repository is the **Capstone Project**, which focuses on **Electric Vehicle (EV) Battery Health Prediction** using the NASA Prognostics Center of Excellence (PCoE) dataset. This project demonstrates end-to-end data science workflows including data engineering, exploratory data analysis, predictive modeling, and deployment.

---

## 📂 Repository Structure

```
Data-Science-and-Engineering-Analytics/
├── Capstone_Project/          # Flagship EV Battery Health Prediction System
├── Projects/                   # Major and minor data science projects
│   ├── Main_Project_Data_Analysis/    # Engineering materials data analysis
│   ├── Main_Project_ML/               # Vehicle Energy Dataset ML project
│   ├── Confidence_Interval/            # Statistical CI analysis application
│   ├── Central_Limit_Theorem/         # CLT demonstration projects
│   └── Mini_Projects/                  # Smaller focused projects
├── Assignments/                # Coursework and practice exercises
├── Training_Notes/            # Educational notebooks and learning materials
├── AI/                        # AI prompts and documentation
├── s3_upload_script.ipynb     # AWS S3 automation script
└── requirements.txt           # Python dependencies
```

---

## 🔋 Capstone Project: EV Battery Health Prediction

### **Objective**
To develop an **Intelligent Predictive Maintenance System** that estimates the **State of Health (SoH)** and forecasts the **Remaining Useful Life (RUL)** of Lithium-ion batteries using machine learning and deep learning techniques.

### **Source Data**

#### **Primary Dataset: NASA PCoE Battery Data Set**
- **Dataset Name**: NASA Prognostics Center of Excellence (PCoE) Battery Data Set
- **Battery Types**: Li-ion 18650 batteries
- **Battery IDs**: B0005, B0006, B0007, B0018, B0028
- **Format**: Raw MATLAB (`.mat`) files containing nested hierarchical structures
- **Location**: `Capstone_Project/source_data/`

#### **Data Structure**
Each `.mat` file contains:
- **Cycle-level data**: Charge, discharge, and impedance operations
- **Measured Parameters**:
  - `Voltage_measured`: Battery terminal voltage (Volts)
  - `Current_measured`: Battery output current (Amps)
  - `Temperature_measured`: Battery temperature (°C)
  - `Current_charge`: Current at charger/load (Amps)
  - `Voltage_charge`: Voltage at charger/load (Volts)
  - `Time`: Time vector for each cycle (seconds)
  - `Capacity`: Battery capacity (Ahr) for discharge cycles
- **Operational Profiles**:
  - **Charge**: Constant Current (CC) at 1.5A until 4.2V, then Constant Voltage (CV) until 20mA
  - **Discharge**: Constant Current (CC) at 2A until voltage thresholds (2.7V, 2.5V, 2.2V, 2.5V for batteries 5, 6, 7, 18 respectively)
  - **Impedance**: EIS frequency sweep from 0.1Hz to 5kHz

#### **Processed Data Files**
- `Flattened_b0005.csv`, `Flattened_b0006.csv`, `Flattened_b0007.csv`, `Flattened_b0018.csv`: Flattened hierarchical data
- `final_df_export.csv`: Complete aggregated dataset
- `final_df_export - b0005 10 cycles.csv`: Sample subset for testing

#### **Model Artifacts**
- `lstm_soh_model.h5`, `gru_soh_model.h5`: Deep learning models (Keras/TensorFlow)
- `gru_soh_model.joblib`: Serialized GRU model
- `soh_scaler.joblib`: Feature scaler for preprocessing

### **Key Operations & Tasks**

#### 1. **Data Engineering & Pipeline**
- **Extraction**: Parsed complex nested structures from raw `.mat` files using `scipy.io.loadmat`
- **Transformation**: 
  - Flattened hierarchical data into structured Pandas DataFrames
  - Aggregated cycle-level metrics (mean voltage, current, temperature)
  - Derived physics-based features (Voltage Drop `Delta_V`, Temperature Increase `Delta_T`, Discharge Time)
- **Storage**: 
  - Implemented automated pipeline to upload processed data to **AWS S3** using `boto3`
  - Data stored in both CSV and Parquet formats for efficient access
  - Cloud integration for scalable deployment

#### 2. **Exploratory Data Analysis (EDA) & Preprocessing**
- **Degradation Analysis**: 
  - Visualized capacity fade over cycle life
  - Identified "knee points" (onset of rapid degradation)
  - Tracked SoH decline patterns across different batteries
- **Correlation Analysis**: 
  - Generated heatmaps to identify relationships between voltage, current, temperature, and capacity
  - Identified key features for predictive modeling
- **Outlier Detection**: 
  - Applied **Z-Score** analysis to identify and handle anomalies in sensor data
  - Implemented robust data quality checks
- **Feature Engineering**: 
  - Derived time-series features (rolling averages, trends)
  - Created cycle-based aggregations
  - Engineered domain-specific metrics

#### 3. **Predictive Modeling**

##### **Supervised Learning (SoH Estimation)**
- **Random Forest Regressor**: 
  - Achieved high accuracy (R² ~0.94) in mapping cycle parameters to SoH
  - Provides feature importance for interpretability
- **XGBoost**: 
  - Utilized gradient boosting for robust baseline performance
  - R²: 0.9276, MAE: 1.76%
  - Strong alternative to Random Forest

##### **Deep Learning (Time-Series Forecasting)**
- **LSTM (Long Short-Term Memory)**: 
  - Captured long-term dependencies in degradation trends for RUL prediction
  - Requires sequence of 10 cycles for prediction
  - Performance: R² 0.29–0.81 (varies with architecture tuning)
- **GRU (Gated Recurrent Unit)**: 
  - Efficient alternative to LSTM for sequential data
  - Best performer: MAE 1.50% (exceeded ±5% error target)
  - Suitable for production deployment

##### **Unsupervised Learning**
- **K-Means Clustering**: 
  - Segmented battery life into distinct aging stages based on operational data
  - Identified degradation patterns and failure modes

##### **Reinforcement Learning**
- **Q-Learning**: 
  - Developed an agent to optimize charging profiles (current/voltage control)
  - Aims to extend battery life through intelligent charging strategies

#### 4. **Model Explainability**
- **SHAP (SHapley Additive exPlanations)**: 
  - Used to interpret model decisions
  - Quantified the impact of features like voltage and temperature on SoH predictions
  - Enhanced model transparency for engineering decisions

#### 5. **Deployment**
- **Web Application**: 
  - Built interactive dashboard using **Streamlit** (`app.py`, `app_v2.py`)
  - Features:
    - Real-time SoH and RUL predictions
    - Support for single-cycle predictions (RF/XGBoost)
    - Sequence-based predictions (LSTM/GRU) with 10-cycle history
    - CSV file upload capability
    - AWS S3 integration for cloud data access
    - Interactive visualizations and model comparison
  - Deployment-ready with model loading, preprocessing, and prediction pipeline

### **Notable Files**

#### **Core Analysis Notebook**
- `Capstone_Project/jupyter_notebooks/EV_Battery_Health_Prediction.ipynb`: 
  - Comprehensive notebook containing the entire analysis workflow
  - Includes data extraction, EDA, feature engineering, model training, evaluation, and visualization
  - Documented with markdown explanations and results

#### **Deployment Applications**
- `Capstone_Project/jupyter_notebooks/app.py`: 
  - Streamlit application for model deployment
  - Supports all four models (RF, XGBoost, LSTM, GRU)
  - Handles both manual input and CSV file uploads
- `Capstone_Project/jupyter_notebooks/app_v2.py`: 
  - Enhanced version with AWS S3 integration
  - Supports demo data, CSV upload, and S3 data loading
  - Advanced features for battery selection and cycle limiting

#### **Documentation**
- `Capstone_Project/jupyter_notebooks/EV_Battery_Health_Prediction.html`: 
  - Exported HTML documentation of the analysis
- `Capstone_Project/jupyter_notebooks/EV_Battery_Health_Prediction.docx`: 
  - Word document version of the project report

#### **Project Requirements**
- `Capstone_Project/requirement/EV_Battery_SoH_RUL_Capstone_Guide.pdf`: Project guidelines
- `Capstone_Project/requirement/Intelligent Predictive Maintenance and Battery Health Forecasting System for Electric Vehicles.docx`: Detailed project specification

---

## 🚀 Other Projects

### **Main Project: Data Analysis**
**Location**: `Projects/Main_Project_Data_Analysis/`

#### **Source Data**
- **Engineering Materials Dataset**: 
  - `Data.csv`, `Cleaned_Data.csv`, `Data_cleaned.csv`: Raw and cleaned material properties data
  - `material.csv`: Material specifications
  - `esg_indicators.csv`: Environmental, Social, and Governance indicators
  - `ev_performance_data.csv`: Electric vehicle performance metrics
  - `Merged_Data_Subset.csv`: Combined dataset for analysis
  - `Data_with_custom_metrics.csv`: Enhanced dataset with derived metrics
  - `Top_5_Strength_Ductility.csv`, `Top_5_Strength_Hardness.csv`, `Top_5_Strength_Weight.csv`: Top material rankings

#### **Key Operations & Tasks**
- **Data Quality Assessment**: 
  - Implemented `data_quality_check.py` for comprehensive data validation
  - Missing value analysis, outlier detection, data type validation
- **Exploratory Data Analysis**: 
  - Material property analysis (strength, ductility, hardness, weight)
  - Performance correlation analysis
  - ESG indicator evaluation
- **Data Cleaning & Preprocessing**: 
  - Handling missing values and inconsistencies
  - Data normalization and standardization
- **Insights Generation**: 
  - Top material rankings based on different criteria
  - Performance optimization recommendations
  - Engineering material selection guidance

#### **Notable Files**
- `Engineering_Materials__Project_AravindhG_JupyterNotebook.ipynb`: Main analysis notebook
- `Engineering_Materials_Project_AravindhG_PPT.pptx`: Presentation slides
- `Engineering_Materials_Project_AravindhG.docx`: Project documentation

---

### **Main Project: Machine Learning (VED)**
**Location**: `Projects/Main_Project_ML/`

#### **Source Data**
- **Vehicle Energy Dataset (VED)**: 
  - Static data: ICE, HEV, PHEV, EV vehicle specifications
  - Dynamic data: Time-series driving data from vehicles in Ann Arbor, Michigan
  - Data sources: Google Drive folders and AWS S3 bucket (`s3aravindh973515031797`)
  - GitHub Repository: https://github.com/gsoh/VED

#### **Key Operations & Tasks**
- **Data Loading & Preprocessing**: 
  - Load static data from Excel files (ICE, HEV, PHEV, EV)
  - Sample and concatenate dynamic CSV files (50% sampling strategy)
  - Handle missing values (`'NO DATA'` → `NaN`)
  - Merge static datasets into unified DataFrame
- **Machine Learning Workflows**: 
  - Regression models for energy consumption prediction
  - Clustering for driving pattern analysis
  - Classification for vehicle type identification
- **Cloud Integration**: 
  - Upload processed data to AWS S3 in CSV and Parquet formats
  - Secure authentication using environment variables
  - Google Drive integration via PyDrive

#### **Notable Files**
- `VED_ML.ipynb`: Main machine learning notebook
- `VED_ML_Streamlit.py`, `VED_ML_Streamlit_v2.py`: Interactive Streamlit dashboards
- `VED_EDA_Notepad.txt`: Exploratory data analysis notes
- `VED_ML_doc.docx`: Project documentation
- `Streamlit Dashboard PDF/`: Dashboard documentation (6 PDF files covering data loading, preprocessing, visualization, EDA, and modeling)

#### **Sub-projects**
- `EV Charging Patterns/`: 
  - `EV_Charging_Patterns.ipynb`: Charging pattern analysis
  - `EV_Charging_Patterns_doc.docx`: Documentation

---

### **Confidence Interval Project**
**Location**: `Projects/Confidence_Interval/`

#### **Source Data**
- Simulated Electric Vehicle data for educational purposes
- Sample EV range data, model comparison data
- Industry benchmarks and reference information

#### **Key Operations & Tasks**
- **Statistical Analysis**: 
  - Confidence interval calculations (Z-distribution and T-distribution)
  - Margin of error analysis
  - Prediction interval computation
  - Proportion confidence intervals
- **Interactive Visualization**: 
  - Real-time CI calculations based on user inputs
  - Multiple plots for educational understanding
  - Monte Carlo simulation for CI coverage
- **Engineering Applications**: 
  - EV range analysis
  - Charging cost estimation
  - Quality control evaluation
  - Model comparison frameworks

#### **Notable Files**
- `app.py`: Main Streamlit application (modular design)
- `config.py`: Configuration and settings management
- `statistics.py`: Statistical calculations module
- `visualizations.py`: Plotting and visualization functions
- `ui_components.py`: Streamlit UI components
- `data.py`: Sample data and constants
- `Project_CI_Streamlit.py`: Original monolithic implementation
- `README.md`: Comprehensive project documentation
- `MODULARIZATION_SUMMARY.md`: Code refactoring documentation

---

### **Central Limit Theorem Project**
**Location**: `Projects/Central_Limit_Theorem/`

#### **Key Operations & Tasks**
- **CLT Demonstration**: 
  - Simulation-based verification of sampling distributions
  - Visualization of convergence to normal distribution
  - Multiple implementation approaches (Python, Solara, Streamlit)

#### **Notable Files**
- `Project_CLT_Python.py`: Python implementation
- `Project_CLT_Solara.py`: Solara-based interactive application
- `Project_CLT_Streamlit.py`: Streamlit web application
- `Project_CLT.ipynb`: Jupyter notebook version

---

### **Mini Projects**
**Location**: `Projects/Mini_Projects/`

#### **Projects Included**
1. **Data Logger Project** (`Miniproject_Data_Logger_12April2025.ipynb`):
   - Test bench data logging implementation
   - Synthetic data generation and logging simulation

2. **Vision Project** (`Miniproject_Vision_12April2025.ipynb`):
   - Computer vision applications
   - Image processing and analysis

3. **Energy Efficiency Project** (`Miniproject_Energy_Efficiency_5April2025.ipynb`):
   - Energy efficiency analysis
   - Performance optimization

4. **General Mini Project** (`Miniproject.ipynb`):
   - Various focused data science techniques

---

## 📚 Assignments

**Location**: `Assignments/`

### **Source Data**
- `dod_cycle_life.csv`: Battery depth of discharge vs cycle life analysis
- `summary_statistics.csv`: Statistical summary data for various analyses

### **Key Operations & Tasks**

#### **Jupyter Notebook Assignments**
1. **Assignment30_March2025.ipynb**: 
   - Foundation-level NumPy operations
   - Array creation, properties, mathematical operations, indexing and slicing
   - Learning objectives: Build strong foundation in NumPy array operations

2. **Assignment23_March2025.ipynb**: 
   - Comprehensive NumPy and Pandas exercises
   - Topics: Lottery generation, temperature conversion, buzzword bingo, dice rolling, supermarket sales, IMDb analysis, Olympic statistics
   - Learning objectives: Practical application of NumPy and Pandas

3. **Assignment12_April2025.ipynb**: 
   - Advanced data manipulation and visualization
   - Topics: Data mapping, apply functions, eval function, 3D plotting
   - Learning objectives: Master advanced Pandas operations

4. **Assignment_Gaussian_Elimination.ipynb**: 
   - Implementation of Gaussian elimination for solving linear systems
   - Topics: Linear algebra, numerical methods, matrix operations
   - Learning objectives: Understand numerical methods

5. **Linear_Regression.ipynb**: 
   - Linear regression implementation and analysis
   - Topics: Regression analysis, model fitting, statistical evaluation
   - Learning objectives: Master regression fundamentals

6. **VED.ipynb** & **DEVRT.ipynb**: 
   - Advanced data analysis projects
   - Domain-specific analytical methods for vehicle emission data

### **Streamlit Applications**

1. **Bayes_Naive_Streamlit.py**: 
   - Interactive Naive Bayes classifier
   - Features: Real-time classification, model training interface, results visualization
   - Use cases: Text classification, spam detection, sentiment analysis

2. **Gaussian_Elimination_Streamlit.py**: 
   - Interactive Gaussian elimination solver
   - Features: Matrix input, step-by-step solution display, error handling
   - Use case: Educational tool for linear algebra

3. **KMeansClustering_Streamlit.py**: 
   - Interactive K-means clustering application
   - Features: Data upload, cluster visualization, parameter tuning
   - Use cases: Customer segmentation, pattern recognition

4. **Regression_Correlation_Streamlit.py**: 
   - Interactive regression and correlation analysis
   - Features: Data upload, regression fitting, correlation matrices, statistical metrics
   - Use cases: Statistical analysis, predictive modeling

5. **SupportVectorMachine_Streamlit.py**: 
   - Interactive SVM classifier and regressor
   - Features: Kernel selection, hyperparameter tuning, decision boundary visualization
   - Use cases: Classification problems, pattern recognition

6. **VED_Streamlit.py** / **VED_Streamlit_App.py** / **VED_Streamlit_App_v2.py**: 
   - Vehicle Emission Data analysis applications
   - Features: Multi-version implementation with enhanced features
   - Use cases: Environmental data analysis, emission monitoring

### **Python Scripts**

1. **Chain_Rule_Partial_Derivatives.py**: 
   - Implementation of chain rule for partial derivatives
   - Topics: Calculus, optimization, gradient computation
   - Use cases: Machine learning optimization, engineering calculations

2. **VED.py**: 
   - Core VED analysis functions
   - Features: Data processing, statistical analysis, emission calculations

3. **generate_sample_logs.py**: 
   - Test bench data logger for generating sample log files
   - Features: Synthetic data generation, logging simulation
   - Use cases: System testing, data pipeline validation

### **Notable Files**
- `Assignments/README.md`: Comprehensive assignment documentation with detailed descriptions

---

## 📖 Training Notes

**Location**: `Training_Notes/`

### **Content Overview**

1. **Python_Basics_16Mar2025_Aravindh.ipynb**: 
   - Core Python programming concepts for data science
   - Topics: Data structures, control flow, functions, OOP, file I/O
   - Foundation for data science workflows

2. **NumPy_22Mar2025_Aravindh.ipynb**: 
   - Numerical computing fundamentals
   - Topics: Array manipulation, broadcasting, linear algebra, random number generation
   - Essential for numerical data processing

3. **Pandas_23Mar2025_Aravindh.ipynb**: 
   - Data manipulation and analysis using pandas
   - Topics: DataFrames, Series, data cleaning, merging, grouping, time series analysis
   - Core library for data analysis

4. **Scikit_learn_13April2025.ipynb**: 
   - Machine learning implementation using scikit-learn
   - Topics: Model selection, preprocessing, classification, regression, clustering, evaluation metrics
   - Comprehensive ML framework coverage

5. **mymodule.py**: 
   - Custom Python module for reusable functions
   - Supporting utilities for various projects

### **Learning Path**
- **Beginner**: Python Basics → NumPy Fundamentals
- **Intermediate**: Pandas → Scikit-learn
- **Advanced**: Integration with projects and assignments

---

## 🤖 AI Resources

**Location**: `AI/`

### **Files**
- `GPT_Data_Science_Simplifier_Prompt.md`: 
  - Custom GPT prompt for simplifying data science concepts
  - Designed to explain complex topics in a professional yet accessible manner
  - Includes structure for formulas, diagrams, analogies, and Python syntax
  - Focuses on EV domain examples and cricket analogies for relatability

---

## 🛠️ Installation & Usage

### **Prerequisites**
- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### **Setup Instructions**

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Data-Science-and-Engineering-Analytics
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Jupyter Notebooks:**
   ```bash
   jupyter notebook
   ```
   Navigate to any `.ipynb` file and run cells sequentially.

4. **Run Streamlit Applications:**
   ```bash
   # Capstone Project App
   cd Capstone_Project/jupyter_notebooks
   streamlit run app.py
   
   # Or use app_v2.py for enhanced features
   streamlit run app_v2.py
   
   # Other Streamlit apps
   cd ../../Assignments
   streamlit run <app_name>.py
   ```

5. **AWS S3 Configuration (for cloud features):**
   Create a `.env` file in the project root:
   ```env
   AWS_ACCESS_KEY_ID=your_key
   AWS_SECRET_ACCESS_KEY=your_secret
   AWS_DEFAULT_REGION=your_region
   ```

---

## 📦 Dependencies

### **Core Data Science**
- `pandas >= 1.5` - Data manipulation and analysis
- `numpy >= 1.24.0` - Numerical computing
- `scipy >= 1.10` - Scientific computing

### **Visualization**
- `matplotlib >= 3.5` - Static plotting
- `plotly >= 5.14` - Interactive visualizations
- `seaborn >= 0.12` - Statistical data visualization

### **Machine Learning**
- `scikit-learn >= 1.2` - Machine learning algorithms
- `xgboost >= 1.7.6` - Gradient boosting framework
- `shap >= 0.45.0` - Model interpretability

### **Deep Learning**
- `tensorflow >= 2.12.0` - Deep learning framework
- `keras >= 2.15.0` - High-level neural network API

### **Web Applications**
- `streamlit >= 1.30` - Interactive web apps

### **Cloud & Data Processing**
- `boto3 >= 1.35.100` - AWS SDK for S3 integration
- `openpyxl >= 3.1` - Excel file handling
- `tabulate >= 0.9.0` - Table formatting

### **Environment Management**
- `python-dotenv >= 1.0.1` - Environment variables
- `PyDrive2 == 1.21.3` - Google Drive integration

### **Utilities**
- `joblib >= 1.1.0` - Model serialization
- `random >= 3.0.0` - Random number generation
- `os >= 3.0.0` - Operating system interface
- `datetime >= 4.0.0` - Date and time handling

---

## 📊 Project Statistics

- **Primary Language**: Jupyter Notebook (98.3%), Python (1.7%)
- **Total Projects**: 1 Capstone + 5 Major Projects + Multiple Mini Projects
- **Total Assignments**: 6+ Jupyter Notebooks + 6+ Streamlit Applications
- **Training Materials**: 4+ Comprehensive Notebooks
- **Deployment Applications**: 10+ Streamlit Apps

---

## 🎯 Key Features

- **End-to-End Workflows**: From data extraction to model deployment
- **Multiple ML Paradigms**: Supervised, Unsupervised, Deep Learning, Reinforcement Learning
- **Cloud Integration**: AWS S3 and Google Drive support
- **Interactive Dashboards**: Streamlit applications for real-time analysis
- **Comprehensive Documentation**: Detailed READMEs, code comments, and project reports
- **Educational Focus**: Progressive difficulty from basics to advanced applications
- **Real-World Applications**: EV battery health, material science, vehicle energy analysis

---

## 🤝 Contributing

This repository serves as a learning resource and project portfolio. When working with these files:

1. **Understand before modifying**: Read through the code and comments
2. **Maintain documentation**: Keep comments and docstrings updated
3. **Test thoroughly**: Ensure all functionalities work as expected
4. **Follow conventions**: Maintain consistent coding style

---

## 📞 Support & Contact

For questions, issues, or suggestions:
1. Check the documentation in each project/assignment folder
2. Review the code comments and inline documentation
3. Ensure all required dependencies are installed
4. Verify data file paths and formats

---

## 📄 License

Please refer to repository settings for licensing information.

---

## 🙏 Acknowledgments

- **NASA PCoE**: For providing the battery dataset
- **VED Dataset**: GitHub repository (https://github.com/gsoh/VED)
- **Streamlit Community**: For the excellent web app framework
- **Scientific Python Stack**: NumPy, SciPy, Matplotlib, Pandas, Scikit-learn, TensorFlow

---

*This repository serves as a comprehensive learning resource and project portfolio in the field of Data Science and Engineering Analytics. Each section is designed to provide practical insights and hands-on experience with modern data science tools and techniques, from fundamental concepts to advanced applications in predictive maintenance and engineering analytics.*
