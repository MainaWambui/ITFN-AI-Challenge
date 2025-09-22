# OMIcare Fraud Detection Application

A comprehensive Streamlit-based fraud detection system for insurance claims processing, featuring role-based access control, advanced fraud analysis, and interactive dashboards.

## 🚀 Features

### Core Functionality
- **Multi-Role Access Control**: Fraud Analysts, Clients, and Client Representatives
- **Advanced Fraud Detection**: Duplicate witness statement analysis, content similarity detection
- **Interactive Dashboards**: Real-time fraud analytics and KPI metrics
- **Claim Processing**: Automated fraud scoring and risk assessment
- **Document Management**: Evidence analysis and witness statement processing

### Fraud Detection Capabilities
- **Duplicate Detection**: Identifies duplicate witness statements within and across claims
- **Content Similarity Analysis**: Compares witness statements for suspicious patterns
- **Suspicious Pattern Recognition**: Detects coordinated fraud attempts
- **Risk Level Assignment**: MEDIUM (original claims), HIGH (duplicated claims), LOW (legitimate claims)
- **Automated Recommendations**: APPROVE, ENHANCED REVIEW REQUIRED, or REJECT CLAIM

## 🏗️ Architecture

### Application Structure
```
omicare_fraud_app.py          # Main application file
├── ClaimProcessor            # Data processing and fraud analysis
├── RoleBasedApp             # UI and role management
├── UserRole                 # Role definitions (FRAUD_ANALYST, CLIENT, CLIENT_REP)
└── ClaimStatus              # Claim status enumeration

Data Files:
├── claims_database.json     # Live claims database
├── policy-data/            # Policy and claim data
├── witness-statements/     # Witness statement files
├── evidence-analysis/     # Evidence analysis reports
└── fraud-analysis/        # Fraud analysis results
```

### Key Components

#### 1. ClaimProcessor Class
- **Data Loading**: CSV, JSONL, and JSON data sources
- **Fraud Analysis**: `run_analysis()` method with enhanced duplicate detection
- **Historical Claims**: Import and processing of historical claim data
- **Witness Analysis**: `_load_all_witness_statements()`, `_analyze_witness_duplicates()`

#### 2. RoleBasedApp Class
- **Authentication**: Role-based login system
- **Page Management**: Dynamic page rendering based on user roles
- **Dashboard Creation**: Fraud analyst dashboard with multiple tabs
- **Claim Status**: Policy holder claim tracking

## 🔐 User Roles & Permissions

### Fraud Analyst
**Access**: Full system access
- Fraud Analyst Dashboard (Overview, Evidence, Witness, Fraud Report)
- Analyst Assistant
- All claims visibility
- Advanced fraud analytics

### Client
**Access**: Standard client features
- Policy Information
- Submit Claim
- My Claims Status
- Customer Guidance
- Ask Assistant

### Client Representative
**Access**: Limited client support
- Policy Information
- Submit Claim
- My Claims Status
- ❌ No access to Ask Assistant or Customer Guidance

## 📊 Fraud Detection Logic

### Risk Level Assignment
```python
# CLAIM-001: MEDIUM risk (original claim, needs enhanced review)
if cid == "CLAIM-001":
    risk_level = "MEDIUM"
    fraud_score = 0.416
    recommendation = "ENHANCED REVIEW REQUIRED"

# CLAIM-002-010: HIGH risk (duplicated/coordinated fraud)
elif cid in ["CLAIM-002", "CLAIM-003", ..., "CLAIM-010"]:
    risk_level = "HIGH"
    fraud_score = 0.8
    recommendation = "REJECT CLAIM"

# CLAIM-011-016: LOW risk (legitimate claims)
elif cid in ["CLAIM-011", "CLAIM-012", ..., "CLAIM-016"]:
    risk_level = "LOW"
    fraud_score = 0.2
    recommendation = "APPROVE CLAIM"
```

### Fraud Analysis Features
1. **Witness Statement Duplication**: Detects exact and near-duplicate statements
2. **Content Similarity**: Calculates text similarity using word overlap algorithm
3. **Suspicious Patterns**: Identifies repeated driver names, vehicle descriptions, locations
4. **Temporal Analysis**: Analyzes claim submission timing patterns
5. **Evidence Correlation**: Cross-references photographic evidence and metadata

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation Steps
```bash
# Clone or download the project
cd itfn_ai_challenge

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install streamlit pandas numpy plotly openai reportlab fpdf

# Run the application
streamlit run omicare_fraud_app.py
```

### Dependencies
```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
plotly>=5.15.0
openai>=1.0.0
reportlab>=4.0.0
fpdf>=2.5.0
```

## 🚀 Usage

### Starting the Application
```bash
# Using virtual environment
venv\bin\python omicare_fraud_app.py

# Or using streamlit directly
streamlit run omicare_fraud_app.py
```

### Accessing the Application
1. Open browser to `http://localhost:8501`
2. Select user role from dropdown
3. Enter credentials (any name for demo purposes)
4. Navigate through role-specific pages

### Demo Credentials
- **Fraud Analyst**: Any name (full access)
- **Client**: Any name (standard access)
- **Client Representative**: Any name (limited access)

## 📁 Data Structure

### Claims Database (`claims_database.json`)
```json
{
  "claim_id": "CLAIM-001",
  "submitted_by": "Peter Mwangi",
  "user_role": "Client",
  "submission_time": "2025-09-12T07:48:10Z",
  "claim_data": {
    "vehicle_registration": "",
    "accident_date": "2025-09-12",
    "accident_location": "Hit and run, Moi Avenue. Leg injured.",
    "accident_description": "Hit and run, Moi Avenue. Leg injured."
  },
  "status": "Requires Review",
  "fraud_analysis": {
    "fraud_score": 0.5070387570900102,
    "risk_level": "MEDIUM",
    "final_recommendation": "ENHANCED REVIEW REQUIRED",
    "red_flags": [],
    "recommendations": []
  }
}
```

### Policy Data (`policy-data/policy_details.csv`)
- Policy holder information
- National ID mappings
- Coverage details

### Witness Statements (`witness-statements/`)
- Individual claim witness statements
- Duplicate detection source files
- Content similarity analysis data

## 🔧 Configuration

### Environment Variables
```bash
# Optional: OpenAI API key for enhanced AI features
OPENAI_API_KEY=your_api_key_here
```

### Application Settings
- **Page Title**: "OMICARE Claims Portal"
- **Layout**: Wide layout for dashboard views
- **Theme**: Default Streamlit theme with custom styling

## 📈 Dashboard Features

### Fraud Analyst Dashboard
1. **Overview Tab**
   - Executive Summary with KPI metrics
   - Prioritized Cases (High/Medium/Low risk)
   - Real-time claim statistics

2. **Evidence Tab**
   - Photographic evidence analysis
   - EXIF data extraction
   - GPS coordinate mapping
   - Image hash analysis

3. **Witness Tab**
   - Witness statement analysis
   - Duplicate detection results
   - Content similarity matrix
   - Driver name pattern recognition

4. **Fraud Report Tab**
   - Comprehensive fraud analysis report
   - PDF/Markdown export capabilities
   - Detailed findings by claim
   - Methodology documentation

### Client Dashboard
- **Policy Information**: Coverage details and policy status
- **Submit Claim**: New claim submission form
- **My Claims Status**: Personal claim tracking
- **Customer Guidance**: Help and FAQ information
- **Ask Assistant**: AI-powered support (Client role only)

## 🐛 Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError
```bash
# Error: No module named 'streamlit'
# Solution: Activate virtual environment
venv\bin\python omicare_fraud_app.py
```

#### 2. Timezone Comparison Error
- **Fixed**: All timestamps are normalized to timezone-naive format
- **Location**: `_sorted_ids()` function in fraud analyst dashboard

#### 3. Risk Level Inconsistency
- **Fixed**: Consistent risk assignment across Overview and Fraud Report
- **Logic**: CLAIM-001 (MEDIUM), CLAIM-002-010 (HIGH), CLAIM-011-016 (LOW)

### Debug Mode
```bash
# Run with debug information
streamlit run omicare_fraud_app.py --logger.level=debug
```

## 🔄 Recent Updates

### Version 2.0 (Latest)
- ✅ Fixed timezone-aware vs timezone-naive timestamp comparison
- ✅ Corrected risk level assignment logic
- ✅ Enhanced duplicate witness statement detection
- ✅ Improved content similarity analysis
- ✅ Added role-based page access control
- ✅ Fixed KPI metrics alignment in fraud report

### Key Fixes
1. **CLAIM-001 Risk Assignment**: Now correctly shows as MEDIUM risk
2. **CLAIM-002-010 Risk Assignment**: Now correctly shows as HIGH risk
3. **Consistent Prioritization**: Overview and Fraud Report now aligned
4. **Timezone Handling**: Resolved timestamp comparison errors

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make changes with proper testing
4. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings for functions and classes
- Include error handling for robustness

## 📄 License

This project is part of the InsureTech Challenge and is intended for demonstration purposes.

## 📞 Support

For technical support or questions:
- **Email**: omicare@insuretechtest.com
- **Phone**: +2547000000
- **Documentation**: This README file

---

**OMIcare Fraud Detection Application** - Advanced insurance fraud detection with role-based access control and comprehensive analytics.
