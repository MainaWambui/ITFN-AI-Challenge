#!/usr/bin/env python3
"""
OMIcare Fraud Detection Application Launcher
This script ensures the application runs with proper error handling and setup.
"""

import sys
import subprocess
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'plotly', 
        'openai', 'reportlab', 'fpdf'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\n📦 Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True

def check_data_files():
    """Check if all required data files exist."""
    required_files = [
        'omicare_fraud_app.py',
        'claims_database.json',
        'requirements.txt',
        'policy-data/policy_details.csv',
        'policy-data/claim_notifications.csv',
        'policy-data/insurance_products.json',
        'policy-data/telematics.jsonl',
        'evidence-analysis/',
        'witness-statements/'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
            print(f"❌ Missing: {file_path}")
        else:
            print(f"✅ Found: {file_path}")
    
    if missing_files:
        print(f"\n📁 Missing files: {', '.join(missing_files)}")
        return False
    
    return True

def run_application():
    """Run the Streamlit application."""
    try:
        print("\n🚀 Starting OMIcare Fraud Detection Application...")
        print("📱 The application will open in your default web browser.")
        print("🌐 If it doesn't open automatically, go to: http://localhost:8501")
        print("\n" + "="*60)
        
        # Run streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "omicare_fraud_app.py"], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running application: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user.")
        return True
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main function to run all checks and start the application."""
    print("🛡️ OMIcare Fraud Detection Application")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        return 1
    
    # Check data files
    print("\n📁 Checking data files...")
    if not check_data_files():
        return 1
    
    # Run application
    print("\n✅ All checks passed!")
    if not run_application():
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
