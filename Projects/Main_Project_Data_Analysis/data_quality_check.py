import pandas as pd

def data_quality_check(df):
    """
    Runs basic data quality checks on a given pandas DataFrame.
    Checks for missing values, duplicate records, data types, unique values,
    descriptive statistics, and presence of negative values in numeric columns.
    """   

    # 1. Basic information
    print("1. Dataset Information:")
    print("-" * 30)
    print(df.info())
    
    # 2. Missing values
    print("\n2. Missing Values per Column:")
    print("-" * 30)
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    
    # 3. Duplicate rows
    print("\n3. Duplicate Rows Count:")
    print("-" * 30)
    duplicate_count = df.duplicated().sum()
    print(f"Total duplicate rows: {duplicate_count}")
    
    # 4. Data types
    print("\n4. Data Types per Column:")
    print("-" * 30)
    print(df.dtypes)
    
    # 5. Unique values
    print("\n5. Unique Values per Column:")
    print("-" * 30)
    print(df.nunique())
    
    # 6. Descriptive statistics for numeric columns
    print("\n6. Descriptive Statistics for Numeric Columns:")
    print("-" * 30)
    print(df.describe())
    
    # 7. Check for negative values in numeric columns
    print("\n7. Columns Containing Negative Values:")
    print("-" * 30)
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if (df[col] < 0).any():
            print(f" - Column '{col}' contains negative values.")
    
    # 8. Unique values and their data types
    print("\n8. Unique Values and Their Data Types per Column (with List of Unique Values):")
    print("-" * 30)
    for col in df.columns:
        unique_values = df[col].unique()
        unique_count = len(unique_values)
        dtype = df[col].dtype
        print(f" - Column '{col}': {unique_count} unique values, Data type: {dtype}")
        print("   Unique values:")
        for value in unique_values:
            print(f"     - {value}")