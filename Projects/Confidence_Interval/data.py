"""
Data module for Confidence Interval EV Analysis

This module contains all sample data, constants, and reference information
used throughout the confidence interval analysis application.
"""

# Sample EV Range Data (in kilometers)
SAMPLE_EV_RANGES = [285, 290, 295, 270, 275, 300, 310, 280, 295, 288]

# EV Model Comparison Data
EV_MODEL_COMPARISON_DATA = {
    "model_a": {
        "name": "Model A",
        "mean_range": 295,
        "std_dev": 10,
        "sample_size": 20,
        "description": "Standard EV Model"
    },
    "model_b": {
        "name": "Model B", 
        "mean_range": 310,
        "std_dev": 8,
        "sample_size": 20,
        "description": "Premium EV Model"
    }
}

# Charging Cost Data (in USD)
CHARGING_COST_DATA = {
    "mean": 50,
    "std_dev": 10,
    "sample_size": 100,
    "unit": "USD"
}

# Confidence Level Reference
CONFIDENCE_LEVELS = {
    90: {
        "z_score": 1.645,
        "description": "Lower confidence, narrower interval",
        "use_case": "Preliminary analysis"
    },
    95: {
        "z_score": 1.96,
        "description": "Standard confidence level",
        "use_case": "Most common in research"
    },
    99: {
        "z_score": 2.576,
        "description": "Higher confidence, wider interval",
        "use_case": "Critical applications"
    }
}

# Statistical Formulas Reference
STATISTICAL_FORMULAS = {
    "confidence_interval": {
        "formula": "x̄ ± z*(σ/√n)",
        "description": "Confidence interval for population mean",
        "when_to_use": "Large samples (n > 30) with known population standard deviation"
    },
    "t_confidence_interval": {
        "formula": "x̄ ± t*(s/√n)",
        "description": "Confidence interval using t-distribution",
        "when_to_use": "Small samples (n < 30) or unknown population standard deviation"
    },
    "proportion_confidence_interval": {
        "formula": "p̂ ± z*√(p̂(1-p̂)/n)",
        "description": "Confidence interval for population proportion",
        "when_to_use": "Categorical data, proportions, percentages"
    },
    "prediction_interval": {
        "formula": "x̄ ± t*(s√(1 + 1/n))",
        "description": "Prediction interval for new observation",
        "when_to_use": "Predicting individual values, not population parameters"
    }
}

# EV Industry Benchmarks
EV_BENCHMARKS = {
    "battery_range": {
        "minimum": 200,  # km
        "average": 350,  # km
        "excellent": 500,  # km
        "unit": "kilometers"
    },
    "charging_time": {
        "fast_charging": 30,  # minutes for 80%
        "standard_charging": 240,  # minutes for 80%
        "slow_charging": 480,  # minutes for 80%
        "unit": "minutes"
    },
    "charging_cost": {
        "home_charging": 0.12,  # USD per kWh
        "public_charging": 0.30,  # USD per kWh
        "fast_charging": 0.45,  # USD per kWh
        "unit": "USD per kWh"
    }
}

# Sample Size Guidelines
SAMPLE_SIZE_GUIDELINES = {
    "small": {
        "range": "n < 30",
        "distribution": "t-distribution",
        "use_case": "Pilot studies, preliminary research"
    },
    "medium": {
        "range": "30 ≤ n ≤ 100",
        "distribution": "z-distribution (approximate)",
        "use_case": "Standard research studies"
    },
    "large": {
        "range": "n > 100",
        "distribution": "z-distribution",
        "use_case": "Large-scale studies, population surveys"
    }
}

# Error Types and Interpretations
ERROR_INTERPRETATIONS = {
    "type_i_error": {
        "definition": "Rejecting null hypothesis when it's true",
        "probability": "α (alpha)",
        "example": "Concluding EV range is different when it's not"
    },
    "type_ii_error": {
        "definition": "Failing to reject null hypothesis when it's false",
        "probability": "β (beta)",
        "example": "Missing that EV range is actually different"
    },
    "margin_of_error": {
        "definition": "Half-width of confidence interval",
        "formula": "z*(σ/√n) or t*(s/√n)",
        "interpretation": "Maximum likely difference between sample and population"
    }
}

# Engineering Decision Making Framework
DECISION_FRAMEWORK = {
    "confidence_interval_overlap": {
        "no_overlap": "Strong evidence of difference between groups",
        "partial_overlap": "Inconclusive evidence, need larger samples",
        "complete_overlap": "No evidence of difference between groups"
    },
    "sample_size_considerations": {
        "power_analysis": "Determine sample size needed for desired power",
        "cost_benefit": "Balance precision with resource constraints",
        "practical_limits": "Consider feasibility of data collection"
    },
    "confidence_level_selection": {
        "90_percent": "When higher precision is needed",
        "95_percent": "Standard choice for most applications",
        "99_percent": "When high confidence is critical"
    }
}

def get_sample_data(data_type="ev_ranges"):
    """
    Get sample data based on type
    
    Args:
        data_type (str): Type of data to retrieve
        
    Returns:
        dict: Sample data and metadata
    """
    data_sources = {
        "ev_ranges": {
            "data": SAMPLE_EV_RANGES,
            "description": "Sample EV range data in kilometers",
            "source": "Simulated data for educational purposes"
        },
        "charging_costs": {
            "data": CHARGING_COST_DATA,
            "description": "Charging cost statistics",
            "source": "Industry average estimates"
        },
        "model_comparison": {
            "data": EV_MODEL_COMPARISON_DATA,
            "description": "EV model comparison data",
            "source": "Simulated performance data"
        }
    }
    
    return data_sources.get(data_type, data_sources["ev_ranges"])

def get_confidence_level_info(confidence_level):
    """
    Get information about a specific confidence level
    
    Args:
        confidence_level (int): Confidence level percentage
        
    Returns:
        dict: Information about the confidence level
    """
    return CONFIDENCE_LEVELS.get(confidence_level, CONFIDENCE_LEVELS[95])

def get_statistical_formula(formula_type):
    """
    Get statistical formula information
    
    Args:
        formula_type (str): Type of formula
        
    Returns:
        dict: Formula information
    """
    return STATISTICAL_FORMULAS.get(formula_type, {})

def get_ev_benchmarks(metric_type):
    """
    Get EV industry benchmarks
    
    Args:
        metric_type (str): Type of benchmark
        
    Returns:
        dict: Benchmark data
    """
    return EV_BENCHMARKS.get(metric_type, {})
