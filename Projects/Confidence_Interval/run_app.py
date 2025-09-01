#!/usr/bin/env python3
"""
Launcher script for Confidence Interval EV Analysis Application

This script provides an easy way to run the Streamlit application
with proper error handling and setup instructions.
"""

import sys
import subprocess
import os

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'streamlit',
        'numpy', 
        'scipy',
        'matplotlib',
        'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """Install required dependencies"""
    print("Installing required dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False

def run_application():
    """Run the Streamlit application"""
    print("Starting Confidence Interval EV Analysis Application...")
    print("=" * 60)
    print("📊 Interactive Confidence Interval Analysis for Electric Vehicles")
    print("=" * 60)
    
    try:
        # Run the Streamlit app
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n\nApplication stopped by user.")
    except Exception as e:
        print(f"✗ Error running application: {e}")
        return False
    
    return True

def main():
    """Main launcher function"""
    print("🚀 Confidence Interval EV Analysis Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("✗ Error: app.py not found in current directory")
        print("Please run this script from the Confidence_Interval directory")
        return 1
    
    # Check dependencies
    print("Checking dependencies...")
    missing_packages = check_dependencies()
    
    if missing_packages:
        print(f"✗ Missing packages: {', '.join(missing_packages)}")
        print("Installing dependencies...")
        
        if not install_dependencies():
            print("\n❌ Failed to install dependencies.")
            print("Please install manually using: pip install -r requirements.txt")
            return 1
    else:
        print("✓ All dependencies are installed!")
    
    # Run the application
    print("\nStarting application...")
    if run_application():
        print("\n✅ Application completed successfully!")
        return 0
    else:
        print("\n❌ Application failed to start properly.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
