#!/bin/bash

echo "🛡️ OMIcare Fraud Detection Application"
echo "====================================="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    echo "Please install Python 3.8+ from https://python.org"
    exit 1
fi

echo "✅ Python is installed: $(python3 --version)"
echo

# Check if requirements are installed
echo "📦 Checking dependencies..."
if ! python3 -c "import streamlit, pandas, numpy, plotly" &> /dev/null; then
    echo "❌ Missing dependencies. Installing..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
fi

echo "✅ Dependencies are ready"
echo

# Run the application
echo "🚀 Starting OMIcare Fraud Detection Application..."
echo "📱 The application will open in your default web browser."
echo "🌐 If it doesn't open automatically, go to: http://localhost:8501"
echo
echo "Press Ctrl+C to stop the application"
echo

python3 -m streamlit run omicare_fraud_app.py
