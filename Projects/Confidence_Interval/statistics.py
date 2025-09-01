"""
Statistical calculations module for Confidence Interval EV Analysis

This module contains all statistical functions and calculations used
for confidence interval analysis in the Electric Vehicle application.
"""

import numpy as np
import scipy.stats as stats
from scipy.stats import norm

def calculate_z_score(confidence_level):
    """
    Calculate z-score for given confidence level
    
    Args:
        confidence_level (float): Confidence level as percentage (e.g., 95)
    
    Returns:
        float: Z-score for the given confidence level
    """
    alpha = 1 - (confidence_level / 100)
    return norm.ppf(1 - alpha / 2)

def calculate_confidence_interval(mean, std_dev, sample_size, confidence_level, use_t_dist=False):
    """
    Calculate confidence interval for population mean
    
    Args:
        mean (float): Sample mean
        std_dev (float): Sample standard deviation
        sample_size (int): Sample size
        confidence_level (float): Confidence level as percentage
        use_t_dist (bool): Whether to use t-distribution (for small samples)
    
    Returns:
        tuple: (lower_bound, upper_bound, margin_of_error)
    """
    if use_t_dist:
        df = sample_size - 1
        confidence = confidence_level / 100
        critical_value = stats.t.ppf((1 + confidence) / 2, df)
    else:
        critical_value = calculate_z_score(confidence_level)
    
    margin_of_error = critical_value * (std_dev / np.sqrt(sample_size))
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    
    return lower_bound, upper_bound, margin_of_error

def calculate_proportion_confidence_interval(success_count, total_count, confidence_level=95):
    """
    Calculate confidence interval for population proportion
    
    Args:
        success_count (int): Number of successes
        total_count (int): Total number of trials
        confidence_level (float): Confidence level as percentage
    
    Returns:
        tuple: (lower_bound, upper_bound, margin_of_error, p_hat, standard_error)
    """
    if total_count == 0:
        return None, None, None, None, None
    
    p_hat = success_count / total_count
    z_critical = norm.ppf(0.975)  # 95% confidence
    standard_error = np.sqrt(p_hat * (1 - p_hat) / total_count)
    margin_of_error = z_critical * standard_error
    
    lower_bound = p_hat - margin_of_error
    upper_bound = p_hat + margin_of_error
    
    return lower_bound, upper_bound, margin_of_error, p_hat, standard_error

def calculate_prediction_interval(mean, std_dev, sample_size, confidence_level):
    """
    Calculate prediction interval for a new observation
    
    Args:
        mean (float): Sample mean
        std_dev (float): Sample standard deviation
        sample_size (int): Sample size
        confidence_level (float): Confidence level as percentage
    
    Returns:
        tuple: (lower_bound, upper_bound, margin_of_error)
    """
    df = sample_size - 1
    confidence = confidence_level / 100
    t_critical = stats.t.ppf((1 + confidence) / 2, df)
    
    margin_of_error = t_critical * std_dev * np.sqrt(1 + 1/sample_size)
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    
    return lower_bound, upper_bound, margin_of_error

def simulate_confidence_intervals(true_mean, sample_std, sample_size, confidence_level, num_simulations=100):
    """
    Simulate multiple confidence intervals to demonstrate coverage
    
    Args:
        true_mean (float): True population mean
        sample_std (float): Sample standard deviation
        sample_size (int): Sample size
        confidence_level (float): Confidence level as percentage
        num_simulations (int): Number of simulations to run
    
    Returns:
        tuple: (intervals, captures, capture_rate)
    """
    z_score = calculate_z_score(confidence_level)
    intervals = []
    captures = 0
    
    for _ in range(num_simulations):
        sample = np.random.normal(true_mean, sample_std, sample_size)
        sample_mean = np.mean(sample)
        margin_of_error = z_score * (sample_std / np.sqrt(sample_size))
        lower = sample_mean - margin_of_error
        upper = sample_mean + margin_of_error
        intervals.append((lower, upper))
        
        if lower <= true_mean <= upper:
            captures += 1
    
    capture_rate = captures / num_simulations
    return intervals, captures, capture_rate

def calculate_t_distribution_ci(data, confidence_level):
    """
    Calculate confidence interval using t-distribution for small samples
    
    Args:
        data (list or array): Sample data
        confidence_level (float): Confidence level as percentage
    
    Returns:
        dict: Dictionary containing all calculated statistics
    """
    data = np.array(data, dtype=float)
    n = len(data)
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)
    df = n - 1
    confidence = confidence_level / 100
    t_critical = stats.t.ppf((1 + confidence) / 2, df)
    margin_error = t_critical * (std_dev / np.sqrt(n))
    ci_lower = mean - margin_error
    ci_upper = mean + margin_error
    
    return {
        "sample_size": n,
        "mean": mean,
        "std_dev": std_dev,
        "degrees_of_freedom": df,
        "t_critical": t_critical,
        "margin_error": margin_error,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper
    }

def compare_two_models(model_a_data, model_b_data, confidence_level):
    """
    Compare two EV models using confidence intervals
    
    Args:
        model_a_data (dict): Data for model A (mean, std, n)
        model_b_data (dict): Data for model B (mean, std, n)
        confidence_level (float): Confidence level as percentage
    
    Returns:
        dict: Comparison results including CIs and overlap analysis
    """
    # Calculate CIs for both models
    ci_a = calculate_confidence_interval(
        model_a_data["mean"], 
        model_a_data["std"], 
        model_a_data["n"], 
        confidence_level, 
        use_t_dist=True
    )
    
    ci_b = calculate_confidence_interval(
        model_b_data["mean"], 
        model_b_data["std"], 
        model_b_data["n"], 
        confidence_level, 
        use_t_dist=True
    )
    
    # Check for overlap
    no_overlap = (ci_a[1] < ci_b[0]) or (ci_b[1] < ci_a[0])
    
    return {
        "model_a_ci": ci_a,
        "model_b_ci": ci_b,
        "no_overlap": no_overlap,
        "model_a_mean": model_a_data["mean"],
        "model_b_mean": model_b_data["mean"]
    }
