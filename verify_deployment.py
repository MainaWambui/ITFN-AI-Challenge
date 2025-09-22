#!/usr/bin/env python3
"""
OMIcare Fraud Detection - Deployment Verification Script
This script verifies that all files and dependencies are ready for judge deployment.
"""

import os
import sys
from pathlib import Path

def verify_file_structure():
    """Verify all required files and directories exist."""
    print("📁 Verifying file structure...")
    
    required_files = [
        'omicare_fraud_app.py',
        'requirements.txt',
        'README.md',
        'DEPLOYMENT_GUIDE.md',
        'run_app.py',
        'run_app.bat',
        'run_app.sh',
        'claims_database.json'
    ]
    
    required_dirs = [
        'policy-data',
        'evidence-analysis',
        'witness-statements'
    ]
    
    missing_files = []
    missing_dirs = []
    
    # Check files
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
            print(f"❌ Missing file: {file_path}")
        else:
            print(f"✅ Found: {file_path}")
    
    # Check directories
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
            print(f"❌ Missing directory: {dir_path}")
        else:
            print(f"✅ Found: {dir_path}")
    
    return len(missing_files) == 0 and len(missing_dirs) == 0

def verify_data_files():
    """Verify critical data files exist and are readable."""
    print("\n📊 Verifying data files...")
    
    data_files = [
        'policy-data/policy_details.csv',
        'policy-data/claim_notifications.csv',
        'policy-data/insurance_products.json',
        'policy-data/telematics.jsonl'
    ]
    
    missing_data = []
    
    for file_path in data_files:
        if not Path(file_path).exists():
            missing_data.append(file_path)
            print(f"❌ Missing data file: {file_path}")
        else:
            # Check if file is readable
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    f.read(100)  # Read first 100 characters
                print(f"✅ Data file OK: {file_path}")
            except Exception as e:
                missing_data.append(f"{file_path} (unreadable: {e})")
                print(f"❌ Unreadable data file: {file_path}")
    
    return len(missing_data) == 0

def verify_witness_statements():
    """Verify witness statement files exist."""
    print("\n👥 Verifying witness statements...")
    
    witness_dir = Path('witness-statements')
    if not witness_dir.exists():
        print("❌ Missing witness-statements directory")
        return False
    
    # Check for CLAIM-001-NEW.md to CLAIM-016-NEW.md
    missing_witness = []
    for i in range(1, 17):
        claim_id = f"CLAIM-{i:03d}"
        witness_file = witness_dir / f"{claim_id}-NEW.md"
        if not witness_file.exists():
            missing_witness.append(str(witness_file))
            print(f"❌ Missing witness statement: {witness_file}")
        else:
            print(f"✅ Found: {witness_file}")
    
    return len(missing_witness) == 0

def verify_evidence_analysis():
    """Verify evidence analysis files exist."""
    print("\n🔍 Verifying evidence analysis...")
    
    evidence_dir = Path('evidence-analysis')
    if not evidence_dir.exists():
        print("❌ Missing evidence-analysis directory")
        return False
    
    # Check for CLAIM-001.md to CLAIM-016.md
    missing_evidence = []
    for i in range(1, 17):
        claim_id = f"CLAIM-{i:03d}"
        evidence_file = evidence_dir / f"{claim_id}.md"
        if not evidence_file.exists():
            missing_evidence.append(str(evidence_file))
            print(f"❌ Missing evidence analysis: {evidence_file}")
        else:
            print(f"✅ Found: {evidence_file}")
    
    return len(missing_evidence) == 0

def verify_requirements():
    """Verify requirements.txt has all necessary dependencies."""
    print("\n📦 Verifying requirements.txt...")
    
    if not Path('requirements.txt').exists():
        print("❌ Missing requirements.txt")
        return False
    
    required_deps = [
        'streamlit', 'pandas', 'numpy', 'plotly', 
        'python-dateutil', 'openai', 'reportlab', 'fpdf'
    ]
    
    try:
        with open('requirements.txt', 'r') as f:
            content = f.read()
        
        missing_deps = []
        for dep in required_deps:
            if dep not in content:
                missing_deps.append(dep)
                print(f"❌ Missing dependency: {dep}")
            else:
                print(f"✅ Found dependency: {dep}")
        
        return len(missing_deps) == 0
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}")
        return False

def verify_application():
    """Verify the main application file is syntactically correct."""
    print("\n🐍 Verifying application syntax...")
    
    try:
        with open('omicare_fraud_app.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Basic syntax check
        compile(content, 'omicare_fraud_app.py', 'exec')
        print("✅ Application syntax is valid")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in application: {e}")
        return False
    except Exception as e:
        print(f"❌ Error verifying application: {e}")
        return False

def main():
    """Main verification function."""
    print("🛡️ OMIcare Fraud Detection - Deployment Verification")
    print("=" * 60)
    
    checks = [
        ("File Structure", verify_file_structure),
        ("Data Files", verify_data_files),
        ("Witness Statements", verify_witness_statements),
        ("Evidence Analysis", verify_evidence_analysis),
        ("Requirements", verify_requirements),
        ("Application Syntax", verify_application)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            if not check_func():
                all_passed = False
        except Exception as e:
            print(f"❌ Error in {check_name}: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("✅ The application is ready for judge deployment!")
        print("\n📋 Next steps for judges:")
        print("1. Run: python run_app.py")
        print("2. Or: ./run_app.sh (Linux/Mac)")
        print("3. Or: run_app.bat (Windows)")
        print("4. Open browser to: http://localhost:8501")
        return 0
    else:
        print("❌ SOME CHECKS FAILED!")
        print("Please fix the issues above before deployment.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
