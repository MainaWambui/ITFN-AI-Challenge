#!/bin/bash
# Installation script for OMIcare enhanced features

echo "🛡️ OMIcare Enhanced Features - Package Installer"
echo "=================================================="

# Activate virtual environment
source venv/bin/activate

echo "📦 Installing required packages..."

# Install packages one by one
pip install python-docx
pip install PyPDF2
pip install openai
pip install reportlab
pip install fpdf

echo "✅ Installation complete!"
echo "🚀 You can now run the app with: python run_app.py"
echo "🔗 Access your app at: http://localhost:8501"
