@echo off
echo 🛡️ OMIcare Fraud Detection Application
echo =====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python is installed
echo.

REM Check if requirements are installed
echo 📦 Checking dependencies...
python -c "import streamlit, pandas, numpy, plotly" >nul 2>&1
if errorlevel 1 (
    echo ❌ Missing dependencies. Installing...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
)

echo ✅ Dependencies are ready
echo.

REM Run the application
echo 🚀 Starting OMIcare Fraud Detection Application...
echo 📱 The application will open in your default web browser.
echo 🌐 If it doesn't open automatically, go to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.

python -m streamlit run omicare_fraud_app.py

pause
