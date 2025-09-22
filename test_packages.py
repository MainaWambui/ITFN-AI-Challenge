#!/usr/bin/env python3
"""
Test script to verify all packages are working correctly
"""

def test_packages():
    print("🧪 Testing package imports...")
    
    packages = [
        ("streamlit", "st"),
        ("pandas", "pd"),
        ("numpy", "np"),
        ("plotly", "px"),
        ("openai", "OpenAI"),
        ("reportlab", "reportlab"),
        ("fpdf", "fpdf"),
        ("PIL", "Image"),
        ("docx", "Document"),
        ("PyPDF2", "PyPDF2")
    ]
    
    success_count = 0
    for package_name, import_name in packages:
        try:
            if package_name == "streamlit":
                import streamlit as st
            elif package_name == "pandas":
                import pandas as pd
            elif package_name == "numpy":
                import numpy as np
            elif package_name == "plotly":
                import plotly.express as px
            elif package_name == "openai":
                from openai import OpenAI
            elif package_name == "reportlab":
                import reportlab
            elif package_name == "fpdf":
                import fpdf
            elif package_name == "PIL":
                from PIL import Image
            elif package_name == "docx":
                from docx import Document
            elif package_name == "PyPDF2":
                import PyPDF2
            
            print(f"✅ {package_name} - OK")
            success_count += 1
        except ImportError as e:
            print(f"❌ {package_name} - FAILED: {e}")
    
    print(f"\n📊 Results: {success_count}/{len(packages)} packages working")
    
    if success_count == len(packages):
        print("🎉 All packages are working correctly!")
        return True
    else:
        print("⚠️ Some packages are missing. Install them with:")
        print("pip install python-docx PyPDF2 openai reportlab fpdf Pillow")
        return False

if __name__ == "__main__":
    test_packages()
