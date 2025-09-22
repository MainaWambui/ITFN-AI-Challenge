#!/usr/bin/env python3
"""
Install missing dependencies for enhanced OMIcare features
"""

import subprocess
import sys

def install_package(package_name, import_name=None):
    """Install a package and verify it can be imported."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name} is already installed")
        return True
    except ImportError:
        print(f"📦 Installing {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"✅ {package_name} installed successfully")
            return True
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package_name}")
            return False

def main():
    print("🛡️ OMIcare Enhanced Features - Dependency Installer")
    print("=" * 60)
    
    packages = [
        ("Pillow", "PIL"),
        ("python-docx", "docx"),
        ("PyPDF2", "PyPDF2"),
        ("openai", "openai"),
        ("reportlab", "reportlab"),
        ("fpdf", "fpdf")
    ]
    
    success_count = 0
    for package_name, import_name in packages:
        if install_package(package_name, import_name):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Installation Summary: {success_count}/{len(packages)} packages installed")
    
    if success_count == len(packages):
        print("🎉 All enhanced features are now available!")
        print("🚀 You can now use photo analysis and document processing.")
    else:
        print("⚠️ Some features may not be available.")
        print("💡 Try running the installer again or install packages manually.")
    
    print("\n🔗 Run your app with: python run_app.py")

if __name__ == "__main__":
    main()
