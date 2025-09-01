"""
Test script for Confidence Interval EV Analysis

This script tests the basic functionality of all modules to ensure
they work correctly together.
"""

import sys
import numpy as np

def test_imports():
    """Test that all modules can be imported successfully"""
    try:
        from config import setup_page_config, get_user_inputs, SIMULATION_CONFIG
        print("✓ config.py imported successfully")
        
        from statistics import (
            calculate_confidence_interval,
            calculate_proportion_confidence_interval,
            simulate_confidence_intervals
        )
        print("✓ statistics.py imported successfully")
        
        from visualizations import plot_confidence_interval_normal_dist
        print("✓ visualizations.py imported successfully")
        
        from ui_components import render_confidence_interval_section
        print("✓ ui_components.py imported successfully")
        
        from data import get_sample_data, SAMPLE_EV_RANGES
        print("✓ data.py imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_statistical_calculations():
    """Test basic statistical calculations"""
    try:
        from statistics import calculate_confidence_interval
        
        # Test confidence interval calculation
        ci_lower, ci_upper, margin_error = calculate_confidence_interval(
            mean=50, std_dev=10, sample_size=100, confidence_level=95
        )
        
        print(f"✓ Confidence interval calculation: ({ci_lower:.2f}, {ci_upper:.2f})")
        print(f"✓ Margin of error: {margin_error:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ Statistical calculation error: {e}")
        return False

def test_data_retrieval():
    """Test data retrieval functions"""
    try:
        from data import get_sample_data, SAMPLE_EV_RANGES
        
        # Test sample data retrieval
        ev_data = get_sample_data("ev_ranges")
        print(f"✓ Sample data retrieved: {len(ev_data['data'])} data points")
        
        # Test EV ranges data
        print(f"✓ EV ranges data: {SAMPLE_EV_RANGES}")
        
        return True
    except Exception as e:
        print(f"✗ Data retrieval error: {e}")
        return False

def test_configuration():
    """Test configuration settings"""
    try:
        from config import DEFAULT_VALUES, INPUT_RANGES, SIMULATION_CONFIG
        
        print(f"✓ Default values: {DEFAULT_VALUES}")
        print(f"✓ Input ranges configured")
        print(f"✓ Simulation config: {SIMULATION_CONFIG}")
        
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing Confidence Interval EV Analysis Application")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("Statistical Calculations", test_statistical_calculations),
        ("Data Retrieval", test_data_retrieval),
        ("Configuration", test_configuration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"✗ {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application is ready to run.")
        print("\nTo run the application:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run the app: streamlit run app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
