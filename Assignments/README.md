# Data Science and Engineering Analytics - Assignments

## Overview
This repository contains a comprehensive collection of data science and engineering analytics assignments, projects, and applications. The assignments cover fundamental to advanced topics in data science, machine learning, numerical methods, and interactive web applications using Streamlit.

## Repository Structure

### 📓 Jupyter Notebooks (.ipynb)

#### **Assignment12_April2025.ipynb**
- **Purpose**: Advanced data manipulation and visualization techniques
- **Topics Covered**:
  - Data mapping operations with Pandas
  - Apply functions for data transformation
  - Eval function demonstrations
  - 3D plotting and visualization using matplotlib
- **Key Methods**: `map()`, `apply()`, `eval()`, 3D scatter plots
- **Learning Objectives**: Master advanced Pandas operations and data visualization

#### **Assignment23_March2025.ipynb**
- **Purpose**: Comprehensive NumPy and Pandas exercises with practical applications
- **Topics Covered**:
  - **NumPy Section**:
    - Lottery number generation and randomization
    - Performance comparison between lists and arrays
    - Temperature conversion applications
    - Interactive buzzword bingo game
    - Dice rolling simulations
  - **Pandas Section**:
    - Supermarket sales tracking and analysis
    - IMDb movie dataset analysis
    - Missing data detection and handling
    - Birthday analysis and age calculations
    - Olympic medal statistics and rankings
- **Key Methods**: Random sampling, performance benchmarking, data cleaning, statistical analysis
- **Learning Objectives**: Practical application of NumPy and Pandas in real-world scenarios

#### **Assignment30_March2025.ipynb**
- **Purpose**: Foundation-level NumPy operations and array manipulation
- **Topics Covered**:
  - **Module 1**: Array creation and types (1D, 2D, 3D arrays)
  - **Module 2**: Array properties (size, shape, dtype, type conversion)
  - **Module 3**: Mathematical operations (element-wise operations, broadcasting, dot products)
  - **Module 4**: Array indexing and slicing (middle row/column extraction, boolean masking)
- **Key Methods**: `array()`, `ones()`, `arange()`, `reshape()`, `dtype` conversion
- **Learning Objectives**: Build strong foundation in NumPy array operations

#### **Assignment_Gaussian_Elimination.ipynb**
- **Purpose**: Implementation of Gaussian elimination method for solving linear systems
- **Topics Covered**: Linear algebra, numerical methods, system of equations
- **Key Methods**: Matrix operations, row operations, back substitution
- **Learning Objectives**: Understand numerical methods for linear system solving

#### **Linear_Regression.ipynb**
- **Purpose**: Implementation and analysis of linear regression models
- **Topics Covered**: Regression analysis, model fitting, statistical evaluation
- **Key Methods**: Least squares, correlation analysis, prediction
- **Learning Objectives**: Master regression analysis fundamentals

#### **VED.ipynb** & **DEVRT.ipynb**
- **Purpose**: Advanced data analysis projects (specific domain applications)
- **Topics Covered**: Specialized data analysis techniques
- **Key Methods**: Domain-specific analytical methods
- **Learning Objectives**: Apply data science to specific engineering problems

### 🖥️ Streamlit Applications (.py)

#### **Bayes_Naive_Streamlit.py**
- **Purpose**: Interactive Naive Bayes classifier web application
- **Features**: Real-time classification, model training interface, results visualization
- **Technologies**: Streamlit, scikit-learn, pandas
- **Use Case**: Text classification, spam detection, sentiment analysis

#### **Gaussian_Elimination_Streamlit.py**
- **Purpose**: Interactive web interface for Gaussian elimination solver
- **Features**: Matrix input, step-by-step solution display, error handling
- **Technologies**: Streamlit, NumPy, linear algebra libraries
- **Use Case**: Educational tool for learning linear algebra

#### **KMeansClustering_Streamlit.py**
- **Purpose**: Interactive K-means clustering application
- **Features**: Data upload, cluster visualization, parameter tuning
- **Technologies**: Streamlit, scikit-learn, matplotlib, plotly
- **Use Case**: Customer segmentation, data exploration, pattern recognition

#### **Regression_Correlation_Streamlit.py**
- **Purpose**: Interactive regression and correlation analysis tool
- **Features**: Data upload, regression fitting, correlation matrices, statistical metrics
- **Technologies**: Streamlit, pandas, scikit-learn, seaborn
- **Use Case**: Statistical analysis, predictive modeling, data relationships

#### **SupportVectorMachine_Streamlit.py**
- **Purpose**: Interactive SVM classifier and regressor application
- **Features**: Kernel selection, hyperparameter tuning, decision boundary visualization
- **Technologies**: Streamlit, scikit-learn, matplotlib
- **Use Case**: Classification problems, pattern recognition, decision boundary analysis

#### **VED_Streamlit.py** / **VED_Streamlit_App.py** / **VED_Streamlit_App_v2.py**
- **Purpose**: Specialized VED (Vehicle Emission Data) analysis applications
- **Features**: Multi-version implementation with enhanced features
- **Technologies**: Streamlit, pandas, data visualization libraries
- **Use Case**: Environmental data analysis, emission monitoring, regulatory compliance

### 📊 Datasets (.csv)

#### **dod_cycle_life.csv**
- **Purpose**: Battery depth of discharge vs cycle life analysis
- **Structure**: DoD (Depth of Discharge) vs Cycle Life data
- **Use Case**: Battery performance analysis, energy storage optimization
- **Variables**: DoD percentages, corresponding cycle life values

#### **summary_statistics.csv**
- **Purpose**: Statistical summary data for various analyses
- **Structure**: Computed statistical metrics and summaries
- **Use Case**: Quick reference for statistical analysis, report generation

### 🐍 Python Scripts (.py)

#### **Chain_Rule_Partial_Derivatives.py**
- **Purpose**: Implementation of chain rule for partial derivatives
- **Topics Covered**: Calculus, optimization, gradient computation
- **Key Methods**: Automatic differentiation, symbolic computation
- **Use Case**: Machine learning optimization, engineering calculations

#### **VED.py**
- **Purpose**: Core VED (Vehicle Emission Data) analysis functions
- **Features**: Data processing, statistical analysis, emission calculations
- **Technologies**: pandas, NumPy, matplotlib
- **Use Case**: Environmental analysis, automotive industry applications

#### **generate_sample_logs.py**
- **Purpose**: Test bench data logger for generating sample log files
- **Features**: Synthetic data generation, logging simulation, testing utilities
- **Use Case**: System testing, data pipeline validation, simulation studies

## 🚀 Getting Started

### Prerequisites
```bash
# Required Python packages
pip install numpy pandas matplotlib seaborn scikit-learn streamlit plotly jupyter
```

### Running Jupyter Notebooks
```bash
# Start Jupyter Notebook
jupyter notebook

# Open any .ipynb file and run cells sequentially
```

### Running Streamlit Applications
```bash
# Run any Streamlit app
streamlit run <app_name>.py

# Example:
streamlit run Bayes_Naive_Streamlit.py
```

### Using Datasets
```python
# Load CSV datasets
import pandas as pd

# Example: Load battery data
battery_data = pd.read_csv('dod_cycle_life.csv')
print(battery_data.head())
```

## 📚 Learning Path

### Beginner Level
1. **Assignment30_March2025.ipynb** - NumPy fundamentals
2. **Linear_Regression.ipynb** - Basic statistical modeling
3. **Assignment12_April2025.ipynb** - Data manipulation basics

### Intermediate Level
1. **Assignment23_March2025.ipynb** - Comprehensive data analysis
2. **Assignment_Gaussian_Elimination.ipynb** - Numerical methods
3. **Regression_Correlation_Streamlit.py** - Interactive analysis

### Advanced Level
1. **SupportVectorMachine_Streamlit.py** - Machine learning applications
2. **KMeansClustering_Streamlit.py** - Unsupervised learning
3. **VED.ipynb** & **DEVRT.ipynb** - Specialized applications

## 🛠️ Technical Stack

- **Languages**: Python 3.7+
- **Libraries**: 
  - Data Analysis: NumPy, Pandas, SciPy
  - Visualization: Matplotlib, Seaborn, Plotly
  - Machine Learning: Scikit-learn
  - Web Apps: Streamlit
  - Notebooks: Jupyter
- **File Formats**: CSV, IPYNB, PY

## 📈 Key Features

- **Interactive Web Applications**: User-friendly Streamlit interfaces
- **Comprehensive Documentation**: Well-commented code with explanations
- **Real-world Datasets**: Practical data for hands-on learning
- **Progressive Difficulty**: From basics to advanced implementations
- **Multiple Approaches**: Different methods for solving similar problems
- **Educational Focus**: Clear learning objectives and outcomes

## 🎯 Use Cases

- **Educational**: Course assignments and learning materials
- **Research**: Data analysis and modeling techniques
- **Professional**: Template code for data science projects
- **Prototyping**: Quick application development using Streamlit
- **Analysis**: Statistical and machine learning model implementations

## 📝 File Naming Convention

- **Assignments**: `Assignment<DD>_<Month><YYYY>.ipynb`
- **Streamlit Apps**: `<Functionality>_Streamlit.py`
- **Datasets**: `<descriptive_name>.csv`
- **Utilities**: `<functionality>.py`

## 🤝 Contributing

This repository serves as a learning resource. When working with these files:

1. **Understand before modifying**: Read through the code and comments
2. **Maintain documentation**: Keep comments and docstrings updated
3. **Test thoroughly**: Ensure all functionalities work as expected
4. **Follow conventions**: Maintain consistent coding style

## 📞 Support

For questions or issues with any of the assignments or applications:

1. Check the inline comments and documentation
2. Review the learning objectives for context
3. Ensure all required dependencies are installed
4. Verify data file paths and formats

## 🎓 Learning Outcomes

After completing these assignments, learners will have:

- **Strong foundation** in NumPy and Pandas
- **Practical experience** with data visualization
- **Understanding** of machine learning algorithms
- **Skills** in building interactive web applications
- **Knowledge** of numerical methods and statistical analysis
- **Experience** with real-world data science workflows

---

*This repository represents a comprehensive journey through data science and engineering analytics, from fundamental concepts to advanced applications. Each file serves a specific purpose in building a well-rounded skill set in data science and analytics.*
