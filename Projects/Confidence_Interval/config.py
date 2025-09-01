"""
Configuration file for Confidence Interval EV Analysis Streamlit App

This module contains all configuration settings, constants, and page setup
for the Electric Vehicle Confidence Interval analysis application.
"""

import streamlit as st

# Page Configuration
PAGE_CONFIG = {
    "page_title": "Confidence Interval for EV Charging Costs",
    "layout": "wide"
}

# Default Values for User Inputs
DEFAULT_VALUES = {
    "true_mean": 50,
    "sample_std": 10,
    "sample_size": 100,
    "confidence_level": 95
}

# Input Ranges
INPUT_RANGES = {
    "true_mean": {"min": 30, "max": 70, "step": 1},
    "sample_std": {"min": 5, "max": 20, "step": 1},
    "sample_size": {"min": 10, "max": 500, "step": 10},
    "confidence_level": {"min": 90, "max": 99, "step": 1}
}

# Simulation Settings
SIMULATION_CONFIG = {
    "num_simulations": 100,
    "random_seed": 42
}

# Plotting Configuration
PLOT_CONFIG = {
    "figsize": (10, 5),
    "dpi": 300,
    "colors": {
        "normal_dist": "blue",
        "ci_lines": "green",
        "mean_line": "red",
        "fill_area": "lightgreen",
        "margin_error": "orange"
    }
}

# Default EV Data
DEFAULT_EV_DATA = [285, 290, 295, 270, 275, 300, 310, 280, 295, 288]

# EV Model Comparison Data
EV_MODEL_DATA = {
    "model_a": {"mean": 295, "std": 10, "n": 20},
    "model_b": {"mean": 310, "std": 8, "n": 20}
}

def setup_page_config():
    """Configure the Streamlit page settings"""
    st.set_page_config(**PAGE_CONFIG)
    st.title("Confidence Statistics for Electric Vehicle (EV) Charging Costs")

def get_user_inputs():
    """
    Get user inputs from the sidebar
    
    Returns:
        dict: Dictionary containing all user input values
    """
    st.sidebar.header("User Input")
    
    true_mean = st.sidebar.slider(
        "True Mean ($)", 
        min_value=INPUT_RANGES["true_mean"]["min"],
        max_value=INPUT_RANGES["true_mean"]["max"],
        value=DEFAULT_VALUES["true_mean"],
        step=INPUT_RANGES["true_mean"]["step"]
    )
    
    sample_std = st.sidebar.slider(
        "Sample Standard Deviation ($)",
        min_value=INPUT_RANGES["sample_std"]["min"],
        max_value=INPUT_RANGES["sample_std"]["max"],
        value=DEFAULT_VALUES["sample_std"],
        step=INPUT_RANGES["sample_std"]["step"]
    )
    
    sample_size = st.sidebar.slider(
        "Sample Size",
        min_value=INPUT_RANGES["sample_size"]["min"],
        max_value=INPUT_RANGES["sample_size"]["max"],
        value=DEFAULT_VALUES["sample_size"],
        step=INPUT_RANGES["sample_size"]["step"]
    )
    
    confidence_level = st.sidebar.slider(
        "Confidence Level (%)",
        min_value=INPUT_RANGES["confidence_level"]["min"],
        max_value=INPUT_RANGES["confidence_level"]["max"],
        value=DEFAULT_VALUES["confidence_level"],
        step=INPUT_RANGES["confidence_level"]["step"]
    )
    
    return {
        "true_mean": true_mean,
        "sample_std": sample_std,
        "sample_size": sample_size,
        "confidence_level": confidence_level
    }
