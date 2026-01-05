# Engineering Materials Data Analysis

[Engineering Materials Data Analysis](https://engineering-materials-da-wus3wrn.gamma.site/)

This repository contains a Python script for basic data quality checks and a Jupyter Notebook for detailed data analysis of engineering materials. The analysis focuses on initial exploration, handling missing values and outliers, groupwise comparisons, design ratio analysis, and hardness scale correlation.

## Project Overview

The main goal of this project is to perform a comprehensive data analysis on a dataset of engineering materials. This includes cleaning the data, identifying patterns, comparing different material and heat treatment types, and deriving custom metrics for material selection.

## Files

- `g:\DIYguru\Data-Science-and-Engineering-Analytics\Projects\Main_Project_Data_Analysis\data_quality_check.py`: A Python script containing a function to perform basic data quality checks on a pandas DataFrame.
- `g:\DIYguru\Data-Science-and-Engineering-Analytics\Projects\Main_Project_Data_Analysis\Engineering_Materials__Project_AravindhG_JupyterNotebook.ipynb`: A Jupyter Notebook detailing the step-by-step data analysis process.

## Data Quality Check

The `data_quality_check.py` script provides a reusable function `data_quality_check(df)` that performs the following operations on a given pandas DataFrame:

1.  **Basic Information**: Prints `df.info()` to show a summary of the DataFrame.
2.  **Missing Values**: Identifies and prints the count of missing values per column.
3.  **Duplicate Rows**: Counts and prints the total number of duplicate rows.
4.  **Data Types**: Displays the data types of each column.
5.  **Unique Values**: Shows the number of unique values for each column.
6.  **Descriptive Statistics**: Provides descriptive statistics (mean, std, min, max, quartiles) for numeric columns.
7.  **Negative Values Check**: Identifies columns containing negative values among numeric types.
8.  **Unique Values and Data Types (with List)**: Lists unique values and their data types for each column, including a full list of unique entries.

## Jupyter Notebook Analysis Steps

The `Engineering_Materials__Project_AravindhG_JupyterNotebook.ipynb` notebook guides through the following analysis steps:

### Task 1: Initial Exploration & Summary

-   **Import Datasets**: Loads the `Data.csv` file into a pandas DataFrame.
-   **Identify Total Materials and Heat Treatment Types**: Calculates and displays the total number of materials, unique materials, total heat treatments, and unique heat treatments in the dataset.
-   **Check for Missing or Inconsistent Data**: Utilizes the `data_quality_check` function from the `data_quality_check.py` script to get an initial overview of data quality issues.

### Handle Missing Values

-   **Cleaning 'Sy' Column**: Removes 'max' string from the 'Sy' (Yield Strength) column and converts it to an integer data type.
-   **Filling Missing Object Values**: Fills missing values in 'Heat treatment' and 'Desc' (Description) columns with 'Unknown'.
-   **Filling Missing Numeric Values**: Fills missing values in numeric columns (`int64`, `float64`) with their respective mode values.
-   **Strip Whitespace**: Removes leading and trailing whitespace from all string (object) columns.

### Identify and Handle Outliers

-   **Outlier Identification**: A function `identify_outliers` is defined to detect outliers in numeric columns using the Interquartile Range (IQR) method (1.5 * IQR rule). It prints the columns containing outliers and the count of outliers in each.
-   **Plot Outliers**: Box plots are generated for each identified outlier-containing column to visualize the distribution and outliers, along with statistical summaries (mean, Q1, Q3, bounds).
-   **Handle Outliers**: A `handle_outliers` function is applied to cap outliers at the calculated lower and upper bounds (1.5 * IQR rule) for each numeric column.
-   **Comparison Plots**: Histograms and box plots are generated to compare the distributions of selected columns before and after outlier handling, showing the effect of the cleaning process.
-   **Export Cleaned Data**: The cleaned DataFrame is saved to a new CSV file named `Cleaned_Data.csv`.

### Task 2: Groupwise Comparison

-   **Group Average Properties by Material Type**: Calculates the mean of 'Su' (Ultimate Strength), 'Sy' (Yield Strength), 'A5' (Elongation), 'E' (Young's Modulus), 'G' (Shear Modulus), and 'HV' (Vickers Hardness) for each unique 'Material' type.
-   **Group Average Properties by Heat Treatment**: Calculates the mean of the same properties for each unique 'Heat treatment' type.
-   **Comparing Efficiency**: Defines an 'Efficiency' metric (simple average of selected properties) and plots the top 5 material types and heat treatment types by this metric using horizontal bar charts.
-   **Group Average Properties by Material and Heat Treatment**: Calculates the mean of the properties for each unique combination of 'Material' and 'Heat treatment'.
-   **Plotting Grouped Averages**: Bar charts are used to visualize the top 5 combinations of material and heat treatment types based on strength, ductility, and hardness efficiency metrics.

### Task 3: Design Ratio Analysis

-   **Calculate Custom Metrics**: New columns are created for custom metrics:
    -   `Strength_Hardness`: Ratio of Ultimate Strength (`Su`) to Brinell Hardness (`Bhn`).
    -   `Strength_Ductility`: Product of Ultimate Strength (`Su`) and Elongation (`A5`).
    -   `Strength_Weight`: Ratio of Ultimate Strength (`Su`) to Density (`Ro`).
-   **Rank Materials**: Ranks materials based on these custom metrics in descending order.
-   **Export Data with Custom Metrics**: The DataFrame including the new custom metrics and ranks is saved to `Data_with_custom_metrics.csv`.
-   **Display Top Materials**: Prints the top 5 materials for each custom metric.
-   **Plot Custom Metrics**: Bar plots are generated to visualize the top 5 materials for each custom metric.

### Task 4: Hardness Scale Correlation

-   **Group Hardness Data**: Groups the data by 'Material' and 'Heat treatment' to calculate the mean of 'Bhn' and 'HV'.
-   **Analyze Relationship**: Plots a scatter plot with a regression line to visualize the relationship between Brinell (Bhn) and Vickers (HV) hardness.
-   **Detect and Plot Divergence**: Calculates the difference between 'HV' and 'Bhn' and flags materials where this difference exceeds a certain threshold (e.g., 10 units). A scatter plot highlights these diverging samples.

### Task 5: Elasticity and Deformability Insight

-   This section is not detailed in the provided notebook cells but would typically involve analyzing Young's Modulus (E) and Shear Modulus (G) to understand material elasticity and deformability characteristics.

## Dependencies

-   `pandas`
-   `matplotlib`
-   `seaborn`
-   `numpy`

## Usage

To run the analysis:

1.  Ensure you have Python and the required libraries installed.
2.  Place `data_quality_check.py` and `Engineering_Materials__Project_AravindhG_JupyterNotebook.ipynb` in the same directory as your `Data.csv` file.
3.  Open the Jupyter Notebook (`Engineering_Materials__Project_AravindhG_JupyterNotebook.ipynb`) in a Jupyter environment (e.g., Jupyter Lab, Jupyter Notebook).
4.  Run all cells in the notebook sequentially.

## Output Files

-   `Data_cleaned.csv`: The DataFrame after handling missing values and outliers.
-   `Data_with_custom_metrics.csv`: The cleaned DataFrame with additional custom metrics and ranks.
-   `Top_5_Strength_Hardness.csv`: A CSV file containing the top 5 materials based on the Strength-to-Hardness ratio.
-   `Top_5_Strength_Ductility.csv`: A CSV file containing the top 5 materials based on the Strength-to-Ductility index.
-   `Top_5_Strength_Weight.csv`: A CSV file containing the top 5 materials based on the Strength-to-Weight proxy.
