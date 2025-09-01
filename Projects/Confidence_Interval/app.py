"""
Main application file for Confidence Interval EV Analysis

This is the main entry point for the Streamlit application that provides
interactive confidence interval analysis for Electric Vehicle data.
"""

import numpy as np
import streamlit as st

# Import local modules
from config import setup_page_config, get_user_inputs, SIMULATION_CONFIG
from ui_components import (
    render_confidence_interval_section,
    render_confidence_level_section,
    render_margin_of_error_section,
    render_calculation_tasks_section,
    render_interpretation_section,
    render_engineering_application_section,
    render_reflection_section,
    render_summary_section
)

def main():
    """
    Main function to run the Streamlit application
    """
    # Setup page configuration
    setup_page_config()
    
    # Set random seed for reproducibility
    np.random.seed(SIMULATION_CONFIG["random_seed"])
    
    # Get user inputs from sidebar
    user_inputs = get_user_inputs()
    
    # Extract input values
    true_mean = user_inputs["true_mean"]
    sample_std = user_inputs["sample_std"]
    sample_size = user_inputs["sample_size"]
    confidence_level = user_inputs["confidence_level"]
    
    # Render all sections
    render_confidence_interval_section(true_mean, sample_std, sample_size, confidence_level)
    render_confidence_level_section(true_mean, sample_std, sample_size, confidence_level)
    render_margin_of_error_section(true_mean, sample_std, sample_size, confidence_level)
    render_calculation_tasks_section(confidence_level)
    render_interpretation_section(confidence_level)
    render_engineering_application_section(confidence_level)
    render_reflection_section()
    render_summary_section()

if __name__ == "__main__":
    main()
