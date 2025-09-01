"""
Visualization module for Confidence Interval EV Analysis

This module contains all plotting and visualization functions used
for displaying confidence interval analysis results.
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def plot_confidence_interval_normal_dist(true_mean, sample_std, ci_lower, ci_upper, confidence_level):
    """
    Plot confidence interval on a normal distribution curve
    
    Args:
        true_mean (float): True population mean
        sample_std (float): Sample standard deviation
        ci_lower (float): Lower bound of confidence interval
        ci_upper (float): Upper bound of confidence interval
        confidence_level (float): Confidence level as percentage
    """
    x = np.linspace(true_mean - 4 * sample_std, true_mean + 4 * sample_std, 500)
    y = (1 / (sample_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - true_mean) / sample_std)**2)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.plot(x, y, label="Normal Distribution", color="blue")
    ax.axvline(ci_lower, color="green", linestyle="--", label=f"CI Lower = ${ci_lower:.2f}")
    ax.axvline(ci_upper, color="green", linestyle="--", label=f"CI Upper = ${ci_upper:.2f}")
    ax.axvline(true_mean, color="red", linestyle=":", label=f"Mean = ${true_mean}")
    ax.fill_between(x, y, where=(x >= ci_lower) & (x <= ci_upper), color="lightgreen", alpha=0.5)
    ax.legend()
    ax.set_title(f"{confidence_level}% Confidence Interval on a Normal Distribution")
    ax.set_xlabel("Value ($)")
    ax.set_ylabel("Probability Density")
    
    st.pyplot(fig)
    plt.close()

def plot_confidence_intervals_simulation(intervals, true_mean, captures, num_simulations):
    """
    Plot multiple confidence intervals from simulation
    
    Args:
        intervals (list): List of (lower, upper) tuples for each interval
        true_mean (float): True population mean
        captures (int): Number of intervals that captured the true mean
        num_simulations (int): Total number of simulations
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for i, (low, high) in enumerate(intervals):
        color = "green" if low <= true_mean <= high else "red"
        ax.plot([low, high], [i, i], color=color)
    
    ax.axvline(true_mean, color="black", linestyle="--", label="True Mean = 50")
    ax.set_title(f"{captures}/{num_simulations} intervals captured the true mean")
    ax.set_xlabel("Value")
    ax.set_ylabel("Sample Index")
    ax.legend()
    
    st.pyplot(fig)
    plt.close()

def plot_margin_of_error_effect(true_mean, sample_std, ci_lower, ci_upper, margin_of_error):
    """
    Plot the effect of margin of error on confidence interval width
    
    Args:
        true_mean (float): True population mean
        sample_std (float): Sample standard deviation
        ci_lower (float): Lower bound of confidence interval
        ci_upper (float): Upper bound of confidence interval
        margin_of_error (float): Margin of error
    """
    x = np.linspace(true_mean - 4 * sample_std, true_mean + 4 * sample_std, 500)
    y = (1 / (sample_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - true_mean) / sample_std)**2)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.axvline(true_mean - margin_of_error, color="orange", linestyle="--", label=f"Lower = ${ci_lower:.2f}")
    ax.axvline(true_mean + margin_of_error, color="orange", linestyle="--", label=f"Upper = ${ci_upper:.2f}")
    ax.axvline(true_mean, color="red", linestyle=":", label=f"Mean = ${true_mean}")
    ax.set_title("Effect of Margin of Error on Confidence Interval")
    ax.fill_between(x, y, where=(x >= ci_lower) & (x <= ci_upper), color="orange", alpha=0.4)
    ax.legend()
    ax.set_xlabel("Value ($)")
    ax.set_ylabel("Probability Density")
    
    st.pyplot(fig)
    plt.close()

def plot_t_distribution_ci(mean, margin_error, confidence_level):
    """
    Plot confidence interval for t-distribution analysis
    
    Args:
        mean (float): Sample mean
        margin_error (float): Margin of error
        confidence_level (float): Confidence level as percentage
    """
    fig, ax = plt.subplots(figsize=(5, 1))
    ax.errorbar(mean, 0, xerr=margin_error, fmt='o', color='navy', capsize=5)
    ax.axvline(mean, color='green', linestyle='--', label='Mean')
    ax.set_xlim(mean - 3 * margin_error, mean + 3 * margin_error)
    ax.set_yticks([])
    ax.set_xlabel("EV Range (km)")
    ax.set_title(f"{confidence_level}% Confidence Interval")
    ax.legend()
    
    st.pyplot(fig)
    plt.close()

def plot_proportion_ci(p_hat, margin_error, confidence_level):
    """
    Plot confidence interval for proportion analysis
    
    Args:
        p_hat (float): Sample proportion
        margin_error (float): Margin of error
        confidence_level (float): Confidence level as percentage
    """
    fig, ax = plt.subplots(figsize=(6, 1.5))
    ax.errorbar(p_hat, 0, xerr=margin_error, fmt='o', color='purple', capsize=6)
    ax.axvline(p_hat, color='green', linestyle='--', label='Sample Proportion')
    ax.set_xlim(max(0, p_hat - 3 * margin_error), min(1, p_hat + 3 * margin_error))
    ax.set_yticks([])
    ax.set_xlabel("Proportion")
    ax.set_title(f"{confidence_level}% Confidence Interval for Charging Issues")
    ax.legend()
    
    st.pyplot(fig)
    plt.close()

def plot_ci_vs_prediction_interval(mean, ci_margin, pi_margin, confidence_level):
    """
    Plot comparison between confidence interval and prediction interval
    
    Args:
        mean (float): Sample mean
        ci_margin (float): Confidence interval margin of error
        pi_margin (float): Prediction interval margin of error
        confidence_level (float): Confidence level as percentage
    """
    fig, ax = plt.subplots(figsize=(7, 2.5))

    # Confidence Interval
    ax.errorbar(mean, 0.25, xerr=ci_margin, fmt='o', color='green', capsize=5, label=f'{confidence_level}% CI (mean)')
    # Prediction Interval
    ax.errorbar(mean, -0.25, xerr=pi_margin, fmt='o', color='orange', capsize=5, label=f'{confidence_level}% PI (new EV)')

    # Labels and aesthetics
    ax.axvline(mean, color='blue', linestyle='--', label='Sample Mean')
    ax.set_xlim(mean - pi_margin*1.5, mean + pi_margin*1.5)
    ax.set_yticks([])
    ax.set_xlabel("EV Range (km)")
    ax.set_title("Confidence Interval vs Prediction Interval")
    ax.legend()
    
    st.pyplot(fig)
    plt.close()

def plot_model_comparison(model_a_mean, model_b_mean, margin_a, margin_b, confidence_level):
    """
    Plot comparison of two EV models using confidence intervals
    
    Args:
        model_a_mean (float): Mean of model A
        model_b_mean (float): Mean of model B
        margin_a (float): Margin of error for model A
        margin_b (float): Margin of error for model B
        confidence_level (float): Confidence level as percentage
    """
    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.errorbar(model_a_mean, 1, xerr=margin_a, fmt='o', capsize=5, color='blue', label='Model A')
    ax.errorbar(model_b_mean, 0, xerr=margin_b, fmt='o', capsize=5, color='green', label='Model B')
    ax.axvline(model_a_mean, linestyle='--', color='blue', alpha=0.5)
    ax.axvline(model_b_mean, linestyle='--', color='green', alpha=0.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Model B", "Model A"])
    ax.set_xlabel("EV Range (km)")
    ax.set_title(f"{confidence_level}% Confidence Intervals for Model A & B")
    ax.legend()
    
    st.pyplot(fig)
    plt.close()
