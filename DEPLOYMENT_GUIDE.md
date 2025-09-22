# 🚀 OMIcare Fraud Detection - Judge Deployment Guide

## Quick Start (Recommended)

### For Windows Judges:
1. **Double-click `run_app.bat`** - This will automatically:
   - Check Python installation
   - Install dependencies if needed
   - Start the application
   - Open browser to http://localhost:8501

### For Linux/Mac Judges:
1. **Run `./run_app.sh`** - This will automatically:
   - Check Python installation
   - Install dependencies if needed
   - Start the application
   - Open browser to http://localhost:8501

### For All Platforms (Manual):
1. **Run `python run_app.py`** - Cross-platform launcher with full error checking

---

## 📋 Prerequisites

- **Python 3.8 or higher** (Download from https://python.org)
- **Internet connection** (for initial dependency installation)
- **Web browser** (Chrome, Firefox, Safari, Edge)

---

## 🔧 Manual Installation (If Needed)

### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Application
```bash
# Option 1: Using Streamlit directly
streamlit run omicare_fraud_app.py

# Option 2: Using Python launcher
python run_app.py

# Option 3: Using platform-specific scripts
# Windows: run_app.bat
# Linux/Mac: ./run_app.sh
```

---

## 🎯 Testing the Application

### 1. Access the Application
- Open browser to **http://localhost:8501**
- You should see the OMIcare Claims Portal login page

### 2. Test User Roles
- **Fraud Analyst**: Full dashboard access with fraud analytics
- **Client**: Standard claim submission and tracking
- **Client Representative**: Limited access (no Ask Assistant)

### 3. Key Features to Test
- ✅ **Fraud Analyst Dashboard**: Overview, Evidence, Witness, Fraud Report tabs
- ✅ **Claim Submission**: Submit new claims with evidence
- ✅ **Claim Status**: Track claim progress
- ✅ **Risk Assessment**: CLAIM-001 (MEDIUM), CLAIM-002-010 (HIGH), CLAIM-011-016 (LOW)
- ✅ **Duplicate Detection**: Witness statement analysis
- ✅ **Role-based Access**: Different pages for different roles

---

## 📊 Expected Results

### Fraud Analyst Dashboard Overview:
- **High Risk Claims**: 9 (CLAIM-002 through CLAIM-010)
- **Medium Risk Claims**: 1 (CLAIM-001)
- **Low Risk Claims**: 6 (CLAIM-011 through CLAIM-016)
- **Total Claims**: 16

### Key Fraud Detection Features:
- **Duplicate Witness Statements**: Detected across multiple claims
- **Content Similarity**: High similarity scores for coordinated fraud
- **Suspicious Patterns**: Same driver names, vehicles, locations
- **Risk Level Consistency**: Same risk levels across Overview and Fraud Report

---

## 🐛 Troubleshooting

### Common Issues:

#### 1. "No module named 'streamlit'"
**Solution**: Run `pip install -r requirements.txt`

#### 2. "Python not found"
**Solution**: Install Python 3.8+ from https://python.org

#### 3. "Port 8501 already in use"
**Solution**: 
- Stop other Streamlit apps
- Or use: `streamlit run omicare_fraud_app.py --server.port 8502`

#### 4. "Permission denied" (Linux/Mac)
**Solution**: Run `chmod +x run_app.sh` then `./run_app.sh`

#### 5. Application won't start
**Solution**: 
- Check all data files are present
- Verify Python version (3.8+)
- Check internet connection for dependencies

---

## 📁 File Structure

```
itfn_ai_challenge/
├── omicare_fraud_app.py          # Main application
├── requirements.txt              # Python dependencies
├── README.md                     # Documentation
├── run_app.py                    # Cross-platform launcher
├── run_app.bat                   # Windows launcher
├── run_app.sh                    # Linux/Mac launcher
├── claims_database.json          # Live claims database
├── policy-data/                  # Policy and claim data
│   ├── policy_details.csv
│   ├── claim_notifications.csv
│   ├── insurance_products.json
│   └── telematics.jsonl
├── evidence-analysis/            # Evidence analysis files
│   └── CLAIM-001.md to CLAIM-016.md
└── witness-statements/           # Witness statements
    └── CLAIM-001-NEW.md to CLAIM-016-NEW.md
```

---

## 🎯 Demo Scenarios for Judges

### Scenario 1: Fraud Analyst Review
1. Login as "Fraud Analyst"
2. Go to Fraud Analyst Dashboard
3. Check Overview tab - verify risk level counts
4. Check Evidence tab - review photographic evidence
5. Check Witness tab - see duplicate detection results
6. Check Fraud Report tab - export PDF report

### Scenario 2: Client Claim Submission
1. Login as "Client"
2. Go to Submit Claim
3. Fill out claim form with sample data
4. Submit claim
5. Go to My Claims Status to track progress

### Scenario 3: Role-based Access Control
1. Login as "Client Representative"
2. Verify limited access (no Ask Assistant page)
3. Switch to "Client" role
4. Verify full access including Ask Assistant

---

## 📞 Support

If judges encounter issues:
- **Check this guide first**
- **Verify all files are present**
- **Ensure Python 3.8+ is installed**
- **Try the automated launchers first**

---

**Ready for deployment! The application includes comprehensive fraud detection, role-based access control, and interactive dashboards.**
