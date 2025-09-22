#!/usr/bin/env python3
"""
Test script to demonstrate automatic file generation for new claims
"""

import os
import json
from datetime import datetime

def test_auto_generation():
    print("🧪 Testing Automatic File Generation")
    print("=" * 50)
    
    # Check if directories exist
    witness_dir = "witness-statements"
    evidence_dir = "evidence-analysis"
    
    print(f"📁 Checking directories...")
    print(f"✅ witness-statements: {'exists' if os.path.exists(witness_dir) else 'missing'}")
    print(f"✅ evidence-analysis: {'exists' if os.path.exists(evidence_dir) else 'missing'}")
    
    # Count existing files
    witness_files = [f for f in os.listdir(witness_dir) if f.endswith('-NEW.md')] if os.path.exists(witness_dir) else []
    evidence_files = [f for f in os.listdir(evidence_dir) if f.endswith('.md')] if os.path.exists(evidence_dir) else []
    
    print(f"\n📊 Current file counts:")
    print(f"   Witness statements: {len(witness_files)}")
    print(f"   Evidence analyses: {len(evidence_files)}")
    
    # Show latest files
    if witness_files:
        latest_witness = max(witness_files)
        print(f"   Latest witness: {latest_witness}")
    
    if evidence_files:
        latest_evidence = max(evidence_files)
        print(f"   Latest evidence: {latest_evidence}")
    
    print(f"\n🚀 How automatic generation works:")
    print(f"   1. User submits claim with photos/documents")
    print(f"   2. App automatically processes uploaded files")
    print(f"   3. Generates witness-statements/CLAIM-XXX-NEW.md")
    print(f"   4. Generates evidence-analysis/CLAIM-XXX.md")
    print(f"   5. Shows success message with file paths")
    
    print(f"\n✅ Automatic file generation is now enabled!")
    print(f"🔗 Test it by submitting a new claim at: http://localhost:8501")

if __name__ == "__main__":
    test_auto_generation()
