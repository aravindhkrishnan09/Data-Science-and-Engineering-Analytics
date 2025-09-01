# Confidence Interval Analysis for Electric Vehicles

A comprehensive Streamlit application that provides interactive confidence interval analysis for Electric Vehicle (EV) data. This application is designed to help engineers, researchers, and students understand statistical concepts through practical EV industry examples.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Modules Documentation](#modules-documentation)
- [Statistical Concepts Covered](#statistical-concepts-covered)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This application demonstrates confidence interval analysis using real-world Electric Vehicle data scenarios. It provides interactive visualizations, calculations, and educational content to help users understand:

- Confidence intervals and their interpretation
- Margin of error calculations
- T-distribution vs Z-distribution usage
- Prediction intervals
- Model comparison using confidence intervals
- Engineering decision-making frameworks

## ✨ Features

- **Interactive Controls**: Adjustable parameters for true mean, standard deviation, sample size, and confidence level
- **Real-time Calculations**: Dynamic confidence interval calculations based on user inputs
- **Visual Learning**: Multiple plots and visualizations for better understanding
- **Educational Content**: Comprehensive explanations and examples
- **Engineering Applications**: Practical EV industry scenarios
- **Modular Design**: Well-organized code structure for easy maintenance and extension

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download the project files**

2. **Navigate to the project directory**
   ```bash
   cd Projects/Confidence_Interval
   ```

3. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to the URL shown in the terminal (typically `http://localhost:8501`)

## 📖 Usage

### Getting Started

1. **Launch the Application**: Run `streamlit run app.py` in your terminal
2. **Adjust Parameters**: Use the sidebar sliders to modify:
   - True Mean ($): The assumed population mean
   - Sample Standard Deviation ($): Variability in the data
   - Sample Size: Number of observations
   - Confidence Level (%): Desired confidence level (90%, 95%, 99%)

3. **Explore Sections**: Navigate through the expandable sections to learn different concepts

### Key Sections

- **Confidence Interval**: Basic CI calculation and visualization
- **Confidence Level**: Understanding confidence levels through simulation
- **Margin of Error**: Impact of margin of error on interval width
- **Calculation Tasks**: T-distribution and proportion confidence intervals
- **Interpretation**: Decision-making scenarios and examples
- **Engineering Application**: EV model comparison
- **Reflection**: Real-world applications in EV industry
- **Summary**: Comprehensive statistical metrics reference

## 📁 Project Structure

```
Confidence_Interval/
├── app.py                 # Main application entry point
├── config.py              # Configuration and settings
├── statistics.py          # Statistical calculations
├── visualizations.py      # Plotting and visualization functions
├── ui_components.py       # Streamlit UI components
├── data.py               # Sample data and constants
├── requirements.txt      # Python dependencies
├── README.md            # This documentation file
└── Project_CI_Streamlit.py  # Original monolithic file (for reference)
```

## 📚 Modules Documentation

### `app.py`
**Main Application Entry Point**
- Orchestrates all modules
- Sets up the application flow
- Handles user input processing

**Key Functions:**
- `main()`: Primary application function

### `config.py`
**Configuration and Settings Management**
- Page configuration settings
- Default values and input ranges
- Simulation parameters
- Plotting configuration

**Key Functions:**
- `setup_page_config()`: Configure Streamlit page
- `get_user_inputs()`: Handle sidebar user inputs

**Constants:**
- `PAGE_CONFIG`: Streamlit page settings
- `DEFAULT_VALUES`: Default parameter values
- `INPUT_RANGES`: Valid input ranges for sliders
- `SIMULATION_CONFIG`: Simulation settings
- `PLOT_CONFIG`: Visualization settings

### `statistics.py`
**Statistical Calculations Module**
- All mathematical computations
- Confidence interval formulas
- Distribution calculations
- Model comparison logic

**Key Functions:**
- `calculate_confidence_interval()`: Compute CI for population mean
- `calculate_proportion_confidence_interval()`: CI for proportions
- `calculate_prediction_interval()`: Prediction intervals
- `simulate_confidence_intervals()`: Monte Carlo simulation
- `calculate_t_distribution_ci()`: T-distribution confidence intervals
- `compare_two_models()`: EV model comparison

### `visualizations.py`
**Plotting and Visualization Functions**
- All matplotlib plotting functions
- Interactive chart generation
- Statistical visualization

**Key Functions:**
- `plot_confidence_interval_normal_dist()`: CI on normal distribution
- `plot_confidence_intervals_simulation()`: Multiple CI visualization
- `plot_margin_of_error_effect()`: Margin of error impact
- `plot_t_distribution_ci()`: T-distribution CI plots
- `plot_proportion_ci()`: Proportion CI visualization
- `plot_ci_vs_prediction_interval()`: CI vs PI comparison
- `plot_model_comparison()`: EV model comparison plots

### `ui_components.py`
**Streamlit UI Components**
- All expandable sections
- User interface elements
- Content rendering functions

**Key Functions:**
- `render_confidence_interval_section()`: CI section UI
- `render_confidence_level_section()`: Confidence level section
- `render_margin_of_error_section()`: Margin of error section
- `render_calculation_tasks_section()`: Calculation tasks
- `render_interpretation_section()`: Interpretation examples
- `render_engineering_application_section()`: Engineering applications
- `render_reflection_section()`: Reflection content
- `render_summary_section()`: Statistical summary table

### `data.py`
**Data and Constants Management**
- Sample datasets
- Reference information
- Industry benchmarks
- Statistical formulas

**Key Data Structures:**
- `SAMPLE_EV_RANGES`: Sample EV range data
- `EV_MODEL_COMPARISON_DATA`: Model comparison data
- `CONFIDENCE_LEVELS`: Confidence level reference
- `STATISTICAL_FORMULAS`: Formula reference
- `EV_BENCHMARKS`: Industry benchmarks
- `DECISION_FRAMEWORK`: Engineering decision framework

**Key Functions:**
- `get_sample_data()`: Retrieve sample data
- `get_confidence_level_info()`: Get CI level information
- `get_statistical_formula()`: Access formula reference
- `get_ev_benchmarks()`: Get industry benchmarks

## 📊 Statistical Concepts Covered

### Core Concepts
- **Confidence Intervals**: Understanding CI interpretation and calculation
- **Margin of Error**: Impact of sample size and variability
- **Confidence Levels**: Trade-offs between precision and confidence
- **T-Distribution**: When and how to use t-distribution
- **Z-Distribution**: Large sample confidence intervals

### Advanced Topics
- **Proportion Confidence Intervals**: Categorical data analysis
- **Prediction Intervals**: Individual value prediction
- **Model Comparison**: Statistical comparison of EV models
- **Simulation**: Monte Carlo methods for understanding CI coverage

### Engineering Applications
- **EV Range Analysis**: Battery performance assessment
- **Charging Cost Estimation**: Cost analysis and planning
- **Quality Control**: Manufacturing consistency evaluation
- **Model Selection**: Data-driven EV model comparison

## 🔧 Customization

### Adding New Statistical Methods
1. Add calculation function to `statistics.py`
2. Create corresponding visualization in `visualizations.py`
3. Add UI component in `ui_components.py`
4. Update configuration in `config.py` if needed

### Modifying Data Sources
1. Update sample data in `data.py`
2. Modify data retrieval functions
3. Update UI components to use new data

### Extending Visualizations
1. Add new plotting functions to `visualizations.py`
2. Import and use in `ui_components.py`
3. Update configuration for new plot settings

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to all functions
- Include type hints where appropriate
- Write clear, descriptive variable names

### Testing
- Test all statistical calculations
- Verify visualizations render correctly
- Check UI responsiveness
- Validate user input handling

## 📄 License

This project is designed for educational purposes. Feel free to use, modify, and distribute for learning and research.

## 🙏 Acknowledgments

- **Statistical Concepts**: Based on fundamental statistical principles
- **EV Industry Data**: Simulated data for educational purposes
- **Streamlit Community**: For the excellent web app framework
- **Scientific Python Stack**: NumPy, SciPy, Matplotlib, Pandas

## 📞 Support

For questions, issues, or suggestions:
1. Check the documentation in each module
2. Review the code comments
3. Test with different parameter values
4. Consult statistical references for theoretical background

---

**Note**: This application is designed for educational purposes. The EV data used is simulated and should not be used for actual engineering decisions without proper validation.
