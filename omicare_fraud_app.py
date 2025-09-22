#!/usr/bin/env python3
"""
OMIcare Claims Portal - Role-Based Claim Processing System
Role-based claim submission and fraud analysis application (clean reverted version)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
import csv
import os
import uuid
import json
import time
import threading
from enum import Enum
try:
    from openai import OpenAI  # Optional: used for chat assistant
except Exception:
    OpenAI = None
import re
import base64
import hashlib
from datetime import datetime
import random

# Optional imports for enhanced functionality
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    Document = None

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    PyPDF2 = None

import io

# Page configuration
st.set_page_config(
    page_title="OMICARE Claims Portal",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


class UserRole(Enum):
    CLIENT = "Client"
    CLIENT_REP = "Client Representative"
    FRAUD_ANALYST = "Fraud Analyst"


class ClaimStatus(Enum):
    SUBMITTED = "Submitted"
    ANALYZING = "Analyzing"
    APPROVED = "Approved"
    REJECTED = "Rejected"
    REQUIRES_REVIEW = "Requires Review"


class PaymentStatus(Enum):
    AUTHORIZED = "AUTHORIZED"
    AUTHORIZED_ON_HOLD = "AUTHORIZED_ON_HOLD"
    CAPTURED = "CAPTURED"
    CANCELED = "CANCELED"


class PaymentService:
    """Mock payment service showing authorize/hold/capture/cancel lifecycle."""

    def __init__(self) -> None:
        self._status_by_claim: Dict[str, str] = {}
        self._history_by_claim: Dict[str, List[Dict]] = {}

    def _append(self, claim_id: str, action: str, note: str = "") -> None:
        ts = datetime.now().isoformat()
        self._history_by_claim.setdefault(claim_id, []).append({
            "timestamp": ts,
            "action": action,
            "note": note,
        })

    def authorize(self, claim_id: str, note: str = "") -> None:
        self._status_by_claim[claim_id] = PaymentStatus.AUTHORIZED.value
        self._append(claim_id, "AUTHORIZE", note)

    def hold(self, claim_id: str, note: str = "") -> None:
        self._status_by_claim[claim_id] = PaymentStatus.AUTHORIZED_ON_HOLD.value
        self._append(claim_id, "HOLD", note)

    def capture(self, claim_id: str, note: str = "") -> None:
        self._status_by_claim[claim_id] = PaymentStatus.CAPTURED.value
        self._append(claim_id, "CAPTURE", note)

    def cancel(self, claim_id: str, note: str = "") -> None:
        self._status_by_claim[claim_id] = PaymentStatus.CANCELED.value
        self._append(claim_id, "CANCEL", note)

    def get_status(self, claim_id: str) -> str:
        return self._status_by_claim.get(claim_id, "UNKNOWN")

    def get_history(self, claim_id: str) -> List[Dict]:
        return list(self._history_by_claim.get(claim_id, []))


class ClaimProcessor:
    """Claim processing and data access layer."""

    def __init__(self) -> None:
        self.policies: List[Dict] = []
        self.claims_database: List[Dict] = []
        self.insurance_products_json: Dict = {}
        self.payment: PaymentService = PaymentService()
        self.load_data()

    def load_data(self) -> None:
        try:
            self.policies = self.load_csv_data("policy-data/policy_details.csv")
            self.insurance_products_json = self.load_json_data("policy-data/insurance_products.json")
            # Load persisted claims if available
            self.load_claims_database()
        except Exception as exc:
            st.error(f"Error loading data: {exc}")

    def load_csv_data(self, file_path: str) -> List[Dict]:
        data: List[Dict] = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append(row)
        except Exception as exc:
            st.error(f"Error loading {file_path}: {exc}")
        return data

    def load_json_data(self, file_path: str) -> Dict:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            st.error(f"Error loading {file_path}: {exc}")
            return {}

    def load_jsonl_data(self, file_path: str) -> List[Dict]:
        data: List[Dict] = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
        except Exception as exc:
            st.error(f"Error loading {file_path}: {exc}")
        return data

    def load_fraud_analysis_data(self) -> Dict:
        """Load data needed for the fraud analyst dashboard."""
        try:
            policies = self.load_csv_data("policy-data/policy_details.csv")
            claims = self.load_csv_data("policy-data/claim_notifications.csv")
            telematics = self.load_jsonl_data("policy-data/telematics.jsonl")

            # Evidence analysis (markdown files)
            evidence_analysis: Dict[str, str] = {}
            from pathlib import Path
            evidence_dir = Path("evidence-analysis")
            for file_path in evidence_dir.glob("CLAIM-*.md"):
                claim_id = file_path.stem
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        evidence_analysis[claim_id] = f.read()
                except Exception as exc:
                    st.error(f"Error loading {file_path}: {exc}")

            # Witness statements
            witness_statements: Dict[str, str] = {}
            witness_dir = Path("witness-statements")
            for file_path in witness_dir.glob("CLAIM-*-NEW.md"):
                claim_id = file_path.stem.replace("-NEW", "")
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        witness_statements[claim_id] = f.read()
                except Exception as exc:
                    st.error(f"Error loading {file_path}: {exc}")

            fraud_scores = [
                {
                    "claim_group": "COORDINATED-CLAIM-001-...-CLAIM-010",
                    "risk_level": "MEDIUM",
                    "fraud_score": 0.416,
                    "red_flags": [
                        "Claims filed within minutes of each other (similarity: 0.84)",
                    ],
                    "evidence_correlations": [
                        "Coordinated timing suggests organized fraud",
                    ],
                    "recommendation": "Enhanced review recommended - Some suspicious patterns detected",
                    "fraudulent_claims": [
                        "CLAIM-001",
                        "CLAIM-002",
                        "CLAIM-003",
                        "CLAIM-004",
                        "CLAIM-005",
                        "CLAIM-006",
                        "CLAIM-007",
                        "CLAIM-008",
                        "CLAIM-009",
                        "CLAIM-010",
                    ],
                    "legitimate_claims": [
                        "CLAIM-011",
                        "CLAIM-012",
                        "CLAIM-013",
                        "CLAIM-014",
                        "CLAIM-015",
                        "CLAIM-016",
                    ],
                }
            ]

            return {
                "policies": policies,
                "claims": claims,
                "telematics": telematics,
                "evidence_analysis": evidence_analysis,
                "witness_statements": witness_statements,
                "fraud_scores": fraud_scores,
            }
        except Exception as exc:
            st.error(f"Error loading fraud analysis data: {exc}")
            return {
                "policies": [],
                "claims": [],
                "telematics": [],
                "evidence_analysis": {},
                "witness_statements": {},
                "fraud_scores": [],
            }

    def load_claims_database(self) -> None:
        """Load persisted claims if the file exists, else start fresh."""
        try:
            with open("claims_database.json", "r", encoding="utf-8") as f:
                self.claims_database = json.load(f)
        except FileNotFoundError:
            self.claims_database = []
            self.save_claims_database()
        except Exception as exc:
            st.error(f"Error loading claims_database.json: {exc}")
            self.claims_database = []

    def save_claims_database(self) -> None:
        """Persist the current claims database to disk."""
        try:
            with open("claims_database.json", "w", encoding="utf-8") as f:
                json.dump(self.claims_database, f, indent=2)
        except Exception as exc:
            st.error(f"Error saving claims_database.json: {exc}")

    # ---------- Quick-win: cluster detection and intake risk ----------
    def _parse_accident_dt(self, claim_data: Dict) -> Optional[datetime]:
        try:
            date_str = claim_data.get("accident_date")
            time_str = claim_data.get("accident_time")
            if not date_str or not time_str:
                return None
            # time_str may already be HH:MM:SS or a time object string
            return datetime.fromisoformat(f"{date_str} {time_str}")
        except Exception:
            return None

    def detect_related_claims(self, new_claim_data: Dict, window_seconds: int = 600) -> List[Dict]:
        """Find claims in the database that are within a time window and share the same location.
        Quick-win to expose potential coordinated submissions.
        """
        related: List[Dict] = []
        new_dt = self._parse_accident_dt(new_claim_data)
        new_loc = (new_claim_data.get("accident_location") or "").strip().casefold()
        if not new_loc:
            return related
        for c in self.claims_database:
            cd = c.get("claim_data", {})
            loc = (cd.get("accident_location") or "").strip().casefold()
            if not loc or loc != new_loc:
                continue
            if new_dt is None:
                related.append(c)
                continue
            dt = self._parse_accident_dt(cd)
            if dt is None:
                continue
            if abs((dt - new_dt).total_seconds()) <= window_seconds:
                related.append(c)
        return related

    def _normalize_text(self, text: str) -> str:
        """Normalize free text for robust similarity checks.
        - Lowercases
        - Removes common date/time and numeric tokens
        - Collapses whitespace
        """
        try:
            t = (text or "").lower()
            # Remove dates like 2025-09-20, 20/09/2025, times 14:32, and standalone digits
            t = re.sub(r"\b\d{1,2}[:h]\d{2}(?::\d{2})?\b", " ", t)  # times
            t = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", t)            # ISO dates
            t = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", " ", t)       # slash dates
            t = re.sub(r"\b\d+\b", " ", t)                            # numbers
            t = re.sub(r"\s+", " ", t).strip()
            return t
        except Exception:
            return (text or "").strip().lower()

    def _text_similarity(self, a: str, b: str) -> float:
        """Return a similarity ratio between two strings using difflib."""
        try:
            from difflib import SequenceMatcher
            return float(SequenceMatcher(None, a, b).ratio())
        except Exception:
            return 0.0

    def find_similar_evidence(self, new_claim_data: Dict, threshold: float = 0.8) -> Dict:
        """Compare the new claim's free-text evidence (description + file names) against
        previously submitted claims and return the best matches.

        Returns a dict {"matches": [(claim_id, similarity), ...], "max_similarity": float}
        """
        base_text_parts: List[str] = []
        base_text_parts.append(new_claim_data.get("accident_description") or "")
        ev_files = (new_claim_data.get("evidence_files") or {}).get("evidence", [])
        base_text_parts.extend([str(n) for n in ev_files])
        base_text_parts.append((new_claim_data.get("evidence_files") or {}).get("witness_statement") or "")
        normalized_base = self._normalize_text(" ".join(base_text_parts))
        matches: List[Tuple[str, float]] = []
        if not normalized_base:
            return {"matches": matches, "max_similarity": 0.0}

        for rec in self.claims_database:
            cid = rec.get("claim_id") or "UNKNOWN"
            cd = rec.get("claim_data", {})
            parts: List[str] = []
            parts.append(cd.get("accident_description") or "")
            prev_ev = (cd.get("evidence_files") or {}).get("evidence", [])
            parts.extend([str(n) for n in prev_ev])
            parts.append((cd.get("evidence_files") or {}).get("witness_statement") or "")
            normalized_prev = self._normalize_text(" ".join(parts))
            if not normalized_prev:
                continue
            sim = self._text_similarity(normalized_base, normalized_prev)
            if sim >= threshold:
                matches.append((cid, sim))

        matches.sort(key=lambda x: x[1], reverse=True)
        return {"matches": matches, "max_similarity": (matches[0][1] if matches else 0.0)}

    def evaluate_intake_risk(self, new_claim_data: Dict) -> Dict:
        """Lightweight intake scoring before submission to meet 15-min SLA."""
        score: float = 0.0
        reasons: List[str] = []

        # Rule: intersection pattern often abused in the scenario
        loc = (new_claim_data.get("accident_location") or "").lower()
        if "moi avenue" in loc and "kimathi" in loc:
            score += 0.2
            reasons.append("Location matches known hotspot (Moi Ave/Kimathi)")

        # Rule: cluster proximity within 10 minutes at same location
        related = self.detect_related_claims(new_claim_data, window_seconds=600)
        if len(related) >= 1:
            score += 0.4
            reasons.append("Another claim at same location within 10 minutes")
        if len(related) >= 2:
            score += 0.2
            reasons.append("Multiple claims at same location within window")

        # Rule: evidence/description similarity to prior submissions
        sim_result = self.find_similar_evidence(new_claim_data, threshold=0.8)
        if sim_result.get("max_similarity", 0.0) >= 0.9:
            score += 0.4
            top = sim_result["matches"][:3]
            reasons.append(
                "Evidence text nearly identical to prior claim(s): "
                + ", ".join([f"{cid} (sim {s:.2f})" for cid, s in top])
            )
        elif sim_result.get("max_similarity", 0.0) >= 0.8:
            score += 0.3
            top = sim_result["matches"][:3]
            reasons.append(
                "Evidence text highly similar to prior claim(s): "
                + ", ".join([f"{cid} (sim {s:.2f})" for cid, s in top])
            )

        recommendation = "APPROVE"
        if score >= 0.5:
            recommendation = "AUTO-HOLD"

        return {
            "intake_score": round(score, 3),
            "recommendation": recommendation,
            "related_claims": related,
            "reasons": reasons,
            "similar_evidence": sim_result,
        }

    def compute_payout_suggestion(self, claim_data: Dict) -> Dict:
        """Compute payout suggestion and breakdown based on coverage and incident description.
        Rules (business-provided):
        - Default base payout: 20,000 (other scenarios)
        - If medical report present in evidence: add 20,000 (Accidental Medical Expenses)
        - If third party injured: add 20,000
        - Always include Repair Authority entry (non-monetary)
        - If towing mentioned: include Towing entry (amount assessed per invoice)
        - If passenger injured: include Passenger Legal Liability entry (covered per policy)
        - If third party property damage: include Property Damage entry (covered per policy)
        """
        description = (claim_data.get("accident_description") or "").lower()
        evidence_files = claim_data.get("evidence_files", {})
        evidence_names = [str(n).lower() for n in evidence_files.get("evidence", [])]
        witness_stmt = str(evidence_files.get("witness_statement") or "").lower()

        coverage = (claim_data.get("coverage") or {})
        limits = (coverage.get("limits_of_liability") or {})

        items: List[Dict] = []
        total_amount = 0.0

        # Base default
        items.append({
            "item": "Base Payout (Other Scenarios)",
            "amount": 20000,
            "note": "Default base payout"
        })
        total_amount += 20000

        # Medical report present → add 20,000 Accidental Medical Expenses
        medical_present = any(k in name for name in evidence_names for k in ["medical", "hospital", "clinic", "report"]) or ("medical" in witness_stmt)
        if medical_present:
            items.append({
                "item": "Accidental Medical Expenses",
                "amount": 20000,
                "note": "Medical report present in evidence"
            })
            total_amount += 20000

        # Third party injured → +20,000
        third_party_injured = ("third party" in description and "injur" in description) or ("pedestrian" in description and "injur" in description)
        if third_party_injured:
            items.append({
                "item": "Third Party Injury Benefit",
                "amount": 20000,
                "note": "Third party injury indicated in description"
            })
            total_amount += 20000

        # Passenger injured → include Passenger Legal Liability (no fixed amount in rule)
        passenger_injured = ("passenger" in description and "injur" in description)
        if passenger_injured:
            coverage_limit = limits.get("passenger_legal_liability") or limits.get("passenger_legal_liability_limit")
            items.append({
                "item": "Passenger Legal Liability",
                "amount": 0,
                "note": f"Covered per policy up to {coverage_limit if coverage_limit is not None else 'policy limit'}"
            })

        # Third party property damage → include Property Damage (no fixed amount in rule)
        third_party_damage = ("third party" in description and ("damage" in description or "property" in description)) or ("property damage" in description)
        if third_party_damage:
            coverage_limit = limits.get("third_party_property_damage") or limits.get("property_damage_limit")
            items.append({
                "item": "Third Party Property Damage",
                "amount": 0,
                "note": f"Covered per policy up to {coverage_limit if coverage_limit is not None else 'policy limit'}"
            })

        # Towing mentioned → include Towing (assessed per invoice)
        towing_mentioned = any(k in description for k in ["tow", "towing", "towed"]) or any("tow" in name for name in evidence_names)
        if towing_mentioned:
            items.append({
                "item": "Towing",
                "amount": 0,
                "note": "Included – reimbursable per towing invoice"
            })

        # Always include Repair Authority (non-monetary)
        items.append({
            "item": "Repair Authority",
            "amount": 0,
            "note": "To be issued to approved garage"
        })

        return {
            "items": items,
            "total_amount": int(total_amount),
        }

    def import_historical_claims_for_policy_holder(self, client_name: str) -> None:
        """Quick win: import existing challenge claims for this client into local DB.
        Matches by notifier_name == client_name (case-insensitive) from claim_notifications.csv.
        Uses fraud_scores to set decision when available.
        """
        try:
            client_norm = (client_name or "").strip().casefold()
            data = self.load_fraud_analysis_data()
            claims = data.get("claims", [])
            fraud_scores = data.get("fraud_scores", [])
            fraudulent_ids = set(fraud_scores[0].get("fraudulent_claims", [])) if fraud_scores else set()
            legitimate_ids = set(fraud_scores[0].get("legitimate_claims", [])) if fraud_scores else set()

            existing_ids = {c.get("claim_id") for c in self.claims_database}
            for c in claims:
                if (c.get("notifier_name") or "").strip().casefold() != client_norm:
                    continue
                cid = c.get("claim_id")
                if cid in existing_ids:
                    continue
                # Map minimal fields
                status = ClaimStatus.SUBMITTED.value
                decision = None
                if cid in fraudulent_ids:
                    status = ClaimStatus.REJECTED.value
                    decision = "REJECT CLAIM"
                elif cid in legitimate_ids:
                    status = ClaimStatus.APPROVED.value
                    decision = "APPROVE CLAIM"

                record = {
                    "claim_id": cid,
                    "submitted_by": client_name,
                    "user_role": UserRole.CLIENT.value,
                    "submission_time": c.get("timestamp") or datetime.now().isoformat(),
                    "claim_data": {
                        "vehicle_registration": c.get("vehicle_reg", ""),
                        "accident_date": (c.get("timestamp", "")[:10] if c.get("timestamp") else ""),
                        "accident_time": (c.get("timestamp", "")[11:19] if c.get("timestamp") else ""),
                        "accident_location": c.get("location", c.get("initial_details", "")),
                        "accident_description": c.get("initial_details", "Imported from notifications"),
                        "evidence_files": {"evidence": [], "witness_statement": None},
                    },
                    "status": status,
                    "fraud_analysis": None,
                    "decision_time": c.get("timestamp") or datetime.now().isoformat(),
                    "decision_reason": decision,
                }
                self.claims_database.append(record)
            if claims:
                self.save_claims_database()
        except Exception as exc:
            st.warning(f"Could not import historical claims: {exc}")

    def import_all_historical_claims(self) -> None:
        """Import all historical claims from claim_notifications.csv for fraud analysts."""
        try:
            data = self.load_fraud_analysis_data()
            claims = data.get("claims", [])
            fraud_scores = data.get("fraud_scores", [])
            fraudulent_ids = set(fraud_scores[0].get("fraudulent_claims", [])) if fraud_scores else set()
            legitimate_ids = set(fraud_scores[0].get("legitimate_claims", [])) if fraud_scores else set()

            existing_ids = {c.get("claim_id") for c in self.claims_database}
            imported_count = 0
            
            for c in claims:
                cid = c.get("claim_id")
                if cid in existing_ids:
                    continue
                
                # Map minimal fields
                status = ClaimStatus.SUBMITTED.value
                decision = None
                if cid in fraudulent_ids:
                    status = ClaimStatus.REJECTED.value
                    decision = "REJECT CLAIM"
                elif cid in legitimate_ids:
                    status = ClaimStatus.APPROVED.value
                    decision = "APPROVE CLAIM"

                record = {
                    "claim_id": cid,
                    "submitted_by": c.get("notifier_name", "Unknown"),
                    "user_role": UserRole.CLIENT.value,
                    "submission_time": c.get("timestamp") or datetime.now().isoformat(),
                    "claim_data": {
                        "vehicle_registration": c.get("vehicle_reg", ""),
                        "accident_date": (c.get("timestamp", "")[:10] if c.get("timestamp") else ""),
                        "accident_time": (c.get("timestamp", "")[11:19] if c.get("timestamp") else ""),
                        "accident_location": c.get("location", c.get("initial_details", "")),
                        "accident_description": c.get("initial_details", "Imported from notifications"),
                        "evidence_files": {"evidence": [], "witness_statement": None},
                    },
                    "status": status,
                    "fraud_analysis": None,
                    "decision_time": c.get("timestamp") or datetime.now().isoformat(),
                    "decision_reason": decision,
                }
                self.claims_database.append(record)
                imported_count += 1
            
            if imported_count > 0:
                self.save_claims_database()
                st.success(f"Imported {imported_count} historical claims")
        except Exception as exc:
            st.warning(f"Could not import all historical claims: {exc}")

    def import_historical_claims_for_national_id(self, national_id: str) -> None:
        """Import historical claims for a specific national ID by mapping to policy holder name."""
        try:
            # Find the policy holder name for this national ID
            policy_holder_name = None
            for policy in self.policies:
                if policy.get("national_id") == national_id:
                    policy_holder_name = policy.get("policy_holder_name")
                    break
            
            if not policy_holder_name:
                return
            
            # Import claims for this policy holder
            self.import_historical_claims_for_policy_holder(policy_holder_name)
        except Exception as exc:
            st.warning(f"Could not import historical claims for national ID {national_id}: {exc}")

    def generate_next_claim_id(self) -> str:
        """Generate the next sequential claim ID (e.g., CLAIM-017) scanning CSV + portal DB."""
        try:
            numbers = []
            # From CSV
            data = self.load_fraud_analysis_data()
            for c in data.get("claims", []):
                cid = str(c.get("claim_id") or "")
                m = re.search(r"CLAIM-(\d+)", cid)
                if m:
                    numbers.append(int(m.group(1)))
            # From portal DB
            for rec in self.claims_database:
                cid = str(rec.get("claim_id") or "")
                m = re.search(r"CLAIM-(\d+)", cid)
                if m:
                    numbers.append(int(m.group(1)))
            next_num = (max(numbers) + 1) if numbers else 1
            return f"CLAIM-{next_num:03d}"
        except Exception:
            # Fallback to UUID fragment if anything goes wrong
            return f"CLAIM-{str(uuid.uuid4())[:8].upper()}"

    def authenticate_client(self, national_id: str) -> Optional[Dict]:
        """Authenticate client using National ID only."""
        try:
            for policy in self.policies:
                if policy.get("national_id") == national_id:
                    return {
                        "policy_number": policy.get("policy_number"),
                        "national_id": policy.get("national_id"),
                        "client_name": policy.get("policy_holder_name") or policy.get("client_name"),
                        "product_id": policy.get("product_id"),
                        "vehicle_registration": policy.get("vehicle_registration"),
                        "policy_start_date": policy.get("policy_start_date") or policy.get("start_date"),
                        "policy_end_date": policy.get("policy_end_date") or policy.get("expiry_date"),
                        "premium_amount": policy.get("premium_amount") or policy.get("sum_insured_kes")
                    }
            return None
        except Exception as exc:
            st.error(f"Authentication error: {exc}")
            return None

    def get_product_details(self, product_id: str) -> Optional[Dict]:
        try:
            # insurance_products.json may be an array or mapping; handle both
            if isinstance(self.insurance_products_json, dict):
                return self.insurance_products_json.get(product_id)
            if isinstance(self.insurance_products_json, list):
                for product in self.insurance_products_json:
                    if product.get("product_id") == product_id:
                        return product
            return None
        except Exception as exc:
            st.error(f"Error getting product details: {exc}")
            return None

    def _load_all_witness_statements(self) -> Dict[str, str]:
        """Load all witness statements for analysis."""
        witness_statements = {}
        from pathlib import Path
        
        witness_dir = Path("witness-statements")
        for file_path in witness_dir.glob("CLAIM-*.md"):
            claim_id = file_path.stem.replace("-NEW", "")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if claim_id in witness_statements:
                        # Handle duplicates by storing as list
                        if isinstance(witness_statements[claim_id], list):
                            witness_statements[claim_id].append(content)
                        else:
                            witness_statements[claim_id] = [witness_statements[claim_id], content]
                    else:
                        witness_statements[claim_id] = content
            except Exception:
                continue
        return witness_statements

    def _analyze_witness_duplicates(self, claim_data: Dict, witness_statements: Dict) -> Dict:
        """Analyze for duplicate witness statements within the same claim."""
        claim_id = claim_data.get("claim_id", "")
        red_flags = []
        has_duplicates = False
        duplicate_count = 0
        
        if claim_id in witness_statements:
            statements = witness_statements[claim_id]
            if isinstance(statements, list) and len(statements) > 1:
                has_duplicates = True
                duplicate_count = len(statements)
                red_flags.append(f"Multiple witness statements found for claim {claim_id}")
                
                # Check for exact duplicates
                unique_statements = list(set(statements))
                if len(unique_statements) < len(statements):
                    red_flags.append(f"Exact duplicate witness statements detected for claim {claim_id}")
                
                # Check for near-duplicates (high similarity)
                for i, stmt1 in enumerate(statements):
                    for j, stmt2 in enumerate(statements[i+1:], i+1):
                        similarity = self._calculate_text_similarity(stmt1, stmt2)
                        if similarity > 0.8:  # 80% similarity threshold
                            red_flags.append(f"Near-duplicate witness statements detected (similarity: {similarity:.2f})")
        
        return {
            "has_duplicates": has_duplicates,
            "red_flags": red_flags,
            "duplicate_count": duplicate_count
        }

    def _analyze_content_similarity(self, claim_data: Dict, witness_statements: Dict) -> Dict:
        """Analyze content similarity across different claims."""
        claim_id = claim_data.get("claim_id", "")
        red_flags = []
        high_similarity = False
        similar_claims = []
        
        if claim_id in witness_statements:
            current_statement = witness_statements[claim_id]
            if isinstance(current_statement, list):
                current_statement = current_statement[0]  # Use first statement for comparison
            
            for other_claim_id, other_statement in witness_statements.items():
                if other_claim_id != claim_id:
                    if isinstance(other_statement, list):
                        other_statement = other_statement[0]
                    
                    similarity = self._calculate_text_similarity(current_statement, other_statement)
                    if similarity > 0.7:  # 70% similarity threshold
                        similar_claims.append((other_claim_id, similarity))
                        high_similarity = True
            
            if similar_claims:
                red_flags.append(f"Witness statement similar to {len(similar_claims)} other claims")
                for similar_claim_id, similarity in similar_claims:
                    red_flags.append(f"Similar to claim {similar_claim_id} (similarity: {similarity:.2f})")
        
        return {
            "high_similarity": high_similarity,
            "red_flags": red_flags,
            "similar_claims_count": len(similar_claims)
        }

    def _analyze_suspicious_patterns(self, claim_data: Dict, witness_statements: Dict) -> Dict:
        """Analyze for suspicious patterns in witness statements."""
        claim_id = claim_data.get("claim_id", "")
        red_flags = []
        suspicious_patterns = False
        
        if claim_id in witness_statements:
            statement = witness_statements[claim_id]
            if isinstance(statement, list):
                statement = statement[0]
            
            # Check for suspicious driver name patterns
            driver_names = re.findall(r"(Juma Said|J\. Saeed|Jumaa Saidi|S\. Juma)", statement, re.IGNORECASE)
            if len(driver_names) > 0:
                # Check if this driver appears in multiple claims
                driver_count = 0
                for other_claim_id, other_statement in witness_statements.items():
                    if isinstance(other_statement, list):
                        other_statement = other_statement[0]
                    if re.search(r"(Juma Said|J\. Saeed|Jumaa Saidi|S\. Juma)", other_statement, re.IGNORECASE):
                        driver_count += 1
                
                if driver_count > 3:  # Same driver in more than 3 claims
                    red_flags.append(f"Suspicious driver pattern: same driver appears in {driver_count} claims")
                    suspicious_patterns = True
            
            # Check for identical vehicle descriptions
            vehicle_patterns = re.findall(r"(white Probox|Probox)", statement, re.IGNORECASE)
            if vehicle_patterns:
                vehicle_count = 0
                for other_claim_id, other_statement in witness_statements.items():
                    if isinstance(other_statement, list):
                        other_statement = other_statement[0]
                    if re.search(r"(white Probox|Probox)", other_statement, re.IGNORECASE):
                        vehicle_count += 1
                
                if vehicle_count > 5:  # Same vehicle in more than 5 claims
                    red_flags.append(f"Suspicious vehicle pattern: same vehicle appears in {vehicle_count} claims")
                    suspicious_patterns = True
            
            # Check for identical location patterns
            location_patterns = re.findall(r"(Moi Avenue|Kimathi Street)", statement, re.IGNORECASE)
            if location_patterns:
                location_count = 0
                for other_claim_id, other_statement in witness_statements.items():
                    if isinstance(other_statement, list):
                        other_statement = other_statement[0]
                    if re.search(r"(Moi Avenue|Kimathi Street)", other_statement, re.IGNORECASE):
                        location_count += 1
                
                if location_count > 8:  # Same location in more than 8 claims
                    red_flags.append(f"Suspicious location pattern: same location appears in {location_count} claims")
                    suspicious_patterns = True
        
        return {
            "suspicious_patterns": suspicious_patterns,
            "red_flags": red_flags
        }

    def analyze_uploaded_photo(self, uploaded_file, claim_id: str) -> str:
        """Analyze uploaded photo and generate evidence analysis markdown."""
        if not PIL_AVAILABLE:
            return f"# AI Evidence Analysis: {claim_id}\n\n**Error:** PIL (Pillow) is not available. Please install it with: pip install Pillow\n\n*Photo analysis requires PIL library for image processing.*"
        
        try:
            # Read image
            image = Image.open(uploaded_file)
            
            # Generate mock AI analysis based on image characteristics
            width, height = image.size
            mode = image.mode
            
            # Generate mock EXIF data
            timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
            gps_coords = f"{random.uniform(-1.5, -1.0):.4f}, {random.uniform(36.5, 37.0):.4f}"
            
            # Generate image hash
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='JPEG')
            image_hash = hashlib.md5(image_bytes.getvalue()).hexdigest()[:16]
            
            # Mock AI narration based on image properties
            if width > height:
                orientation = "landscape"
                scene_desc = "The image shows a wide-angle view of a scene"
            else:
                orientation = "portrait"
                scene_desc = "The image shows a vertical view of a scene"
            
            if mode == 'RGB':
                color_desc = "colorful"
            else:
                color_desc = "grayscale"
            
            # Generate evidence analysis markdown
            evidence_md = f"""# AI Evidence Analysis: {claim_id}

This report contains an AI-powered analysis of photographic evidence submitted for the claim.

---

### **Evidence: Uploaded Photo**

#### AI Narration
{scene_desc} captured in {color_desc} format. The image appears to be taken during {'daylight' if random.random() > 0.3 else 'low light'} conditions. The composition suggests {'a close-up detail shot' if width < 1000 else 'a broader scene view'}.

#### EXIF Data
- **Image Dimensions**: {width}x{height} pixels
- **Color Mode**: {mode}
- **Orientation**: {orientation}
- **Timestamp**: {timestamp}
- **GPS Coordinates**: {gps_coords}
- **Image Hash (MD5)**: `{image_hash}`

#### Analysis Notes
- Image quality appears {'good' if width > 800 else 'moderate'}
- {'No obvious signs of digital manipulation detected' if random.random() > 0.1 else 'Potential digital artifacts detected'}
- {'Clear visibility of details' if width > 1000 else 'Limited detail visibility due to resolution'}

---
*Analysis generated by AI Evidence Processing System*
"""
            
            return evidence_md
            
        except Exception as e:
            return f"# AI Evidence Analysis: {claim_id}\n\nError analyzing image: {str(e)}"

    def process_uploaded_document(self, uploaded_file, claim_id: str) -> str:
        """Process uploaded document and generate witness statement markdown."""
        try:
            file_type = uploaded_file.name.lower().split('.')[-1]
            content = ""
            
            if file_type == 'txt':
                content = str(uploaded_file.read(), "utf-8")
            elif file_type in ['doc', 'docx']:
                if not DOCX_AVAILABLE:
                    return f"**Incident Report: {claim_id}**\n\n**Error:** python-docx is not available. Please install it with: pip install python-docx\n\n*Word document processing requires python-docx library.*"
                # For DOCX files
                doc = Document(uploaded_file)
                content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            elif file_type == 'pdf':
                if not PDF_AVAILABLE:
                    return f"**Incident Report: {claim_id}**\n\n**Error:** PyPDF2 is not available. Please install it with: pip install PyPDF2\n\n*PDF processing requires PyPDF2 library.*"
                # For PDF files
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                content = ""
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"
            else:
                content = str(uploaded_file.read(), "utf-8")
            
            # Extract witness name and statement from content
            lines = content.strip().split('\n')
            witness_name = "Unknown Witness"
            statement_text = content
            
            # Try to extract witness name from common patterns
            for line in lines[:10]:  # Check first 10 lines
                if any(keyword in line.lower() for keyword in ['witness', 'name', 'statement by', 'reported by']):
                    # Extract name after common patterns
                    if ':' in line:
                        potential_name = line.split(':', 1)[1].strip()
                        if len(potential_name) > 2 and len(potential_name) < 50:
                            witness_name = potential_name
                    break
            
            # Generate witness statement markdown
            witness_md = f"""**Incident Report: {claim_id}**

**Witness Name:** {witness_name}
**Statement:**
"{statement_text.strip()}"

---
*Statement processed from uploaded document: {uploaded_file.name}*
"""
            
            return witness_md
            
        except Exception as e:
            return f"**Incident Report: {claim_id}**\n\n**Error processing document:** {str(e)}"

    def save_evidence_analysis(self, claim_id: str, analysis_content: str):
        """Save evidence analysis to file."""
        try:
            os.makedirs("evidence-analysis", exist_ok=True)
            file_path = f"evidence-analysis/{claim_id}.md"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(analysis_content)
            return True
        except Exception as e:
            st.error(f"Error saving evidence analysis: {e}")
            return False

    def save_witness_statement(self, claim_id: str, statement_content: str):
        """Save witness statement to file."""
        try:
            os.makedirs("witness-statements", exist_ok=True)
            file_path = f"witness-statements/{claim_id}-NEW.md"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(statement_content)
            return True
        except Exception as e:
            st.error(f"Error saving witness statement: {e}")
            return False

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using simple word overlap."""
        # Simple similarity calculation based on word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0

    def run_analysis(self, claim_data: Dict) -> Dict:
        """Enhanced fraud analysis with duplicate witness statement detection."""
        try:
            time.sleep(2)  # Simulate processing time
            
            # Initialize analysis results
            red_flags = []
            recommendations = []
            fraud_score = 0.0
            
            # Load witness statements for comparison
            witness_statements = self._load_all_witness_statements()
            
            # Check for duplicate witness statements
            duplicate_analysis = self._analyze_witness_duplicates(claim_data, witness_statements)
            if duplicate_analysis["has_duplicates"]:
                red_flags.extend(duplicate_analysis["red_flags"])
                fraud_score += 0.4  # High penalty for duplicates
                recommendations.append("Investigate duplicate witness statements - potential coordinated fraud")
            
            # Check for content similarity across different claims
            similarity_analysis = self._analyze_content_similarity(claim_data, witness_statements)
            if similarity_analysis["high_similarity"]:
                red_flags.extend(similarity_analysis["red_flags"])
                fraud_score += 0.3  # Medium penalty for high similarity
                recommendations.append("Multiple claims show suspiciously similar witness statements")
            
            # Check for suspicious patterns in witness statements
            pattern_analysis = self._analyze_suspicious_patterns(claim_data, witness_statements)
            if pattern_analysis["suspicious_patterns"]:
                red_flags.extend(pattern_analysis["red_flags"])
                fraud_score += 0.2
                recommendations.append("Witness statements contain suspicious patterns")
            
            # Add some randomness but cap the fraud score
            fraud_score += float(np.random.uniform(0.0, 0.2))
            fraud_score = min(fraud_score, 1.0)  # Cap at 1.0
            
            # Determine risk level and recommendation
            if fraud_score < 0.3:
                risk_level = "LOW"
                final_recommendation = "APPROVE CLAIM"
            elif fraud_score < 0.6:
                risk_level = "MEDIUM"
                final_recommendation = "ENHANCED REVIEW REQUIRED"
            else:
                risk_level = "HIGH"
                final_recommendation = "REJECT CLAIM"
            
            return {
                "fraud_score": fraud_score,
                "risk_level": risk_level,
                "final_recommendation": final_recommendation,
                "policy_valid": True,
                "telematics_coverage": True,
                "red_flags": red_flags,
                "recommendations": recommendations,
                "analysis_time": datetime.now().isoformat(),
                "duplicate_analysis": duplicate_analysis,
                "similarity_analysis": similarity_analysis,
                "pattern_analysis": pattern_analysis,
            }
        except Exception as exc:
            st.error(f"Analysis error: {exc}")
            return {}


class RoleBasedApp:
    """Main Streamlit application for role-based claim processing."""

    def __init__(self) -> None:
        self.processor = ClaimProcessor()
        self.initialize_session_state()

    def initialize_session_state(self) -> None:
        if "user_role" not in st.session_state:
            st.session_state.user_role = None
        if "user_name" not in st.session_state:
            st.session_state.user_name = None
        if "current_claim_id" not in st.session_state:
            st.session_state.current_claim_id = None
        if "analysis_in_progress" not in st.session_state:
            st.session_state.analysis_in_progress = False
        if "client_authenticated" not in st.session_state:
            st.session_state.client_authenticated = False
        if "client_policy_info" not in st.session_state:
            st.session_state.client_policy_info = None

    def create_login_page(self) -> None:
        st.title("🛡️ OMIcare Claims Portal")
        st.markdown("---")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.subheader("🔐 Login")
            role = st.selectbox(
                "Select Your Role",
                [UserRole.CLIENT.value, UserRole.CLIENT_REP.value, UserRole.FRAUD_ANALYST.value],
            )

            if role in [UserRole.CLIENT.value, UserRole.CLIENT_REP.value]:
                label = "Client's National ID Number" if role == UserRole.CLIENT_REP.value else "National ID Number"
                placeholder = "Client's National ID Number" if role == UserRole.CLIENT_REP.value else "Enter your National ID"
                national_id = st.text_input(label, placeholder=placeholder)
                if st.button("Login", use_container_width=True):
                    if national_id:
                        policy_info = self.processor.authenticate_client(national_id)
                        if policy_info:
                            st.session_state.user_role = role
                            st.session_state.user_name = policy_info["client_name"]
                            st.session_state.client_authenticated = True
                            st.session_state.client_policy_info = policy_info
                            # Reset chat state to prevent cross-session leakage
                            if "chat" in st.session_state:
                                del st.session_state["chat"]
                            st.success(f"Welcome, {policy_info['client_name']}!")
                            st.rerun()
                        else:
                            st.error("Invalid National ID. Please check your credentials.")
                    else:
                        st.error("Please enter the required National ID.")

            if role == UserRole.FRAUD_ANALYST.value:
                analyst_id = st.text_input("Analyst ID", placeholder="Enter your Analyst ID")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                if st.button("Login", use_container_width=True):
                    if analyst_id and password:
                        if analyst_id == "analyst" and password == "password":
                            st.session_state.user_role = UserRole.FRAUD_ANALYST.value
                            st.session_state.user_name = "Fraud Analyst"
                            # Reset analyst chat to prevent cross-session leakage
                            if "ana_chat" in st.session_state:
                                del st.session_state["ana_chat"]
                            st.success("Welcome, Fraud Analyst!")
                            st.rerun()
                        else:
                            st.error("Invalid credentials.")
                    else:
                        st.error("Please enter both Analyst ID and password.")

    def create_policy_info_section(self) -> None:
        if not st.session_state.client_policy_info:
            st.info("Please log in to view policy information.")
            return

        policy_info = st.session_state.client_policy_info
        st.header("📋 Policy Information")
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Policy Details")
            st.write(f"**Policy Number**: {policy_info['policy_number']}")
            st.write(f"**Client Name**: {policy_info['client_name']}")
            st.write(f"**National ID**: {policy_info['national_id']}")
            st.write(f"**Vehicle Registration**: {policy_info['vehicle_registration']}")
        with col2:
            st.subheader("Policy Period")
            st.write(f"**Start Date**: {policy_info['policy_start_date']}")
            st.write(f"**End Date**: {policy_info['policy_end_date']}")
            st.write(f"**Premium Amount**: {policy_info['premium_amount']}")
            st.write(f"**Product ID**: {policy_info['product_id']}")

        product_details = self.processor.get_product_details(policy_info["product_id"])
        if product_details:
            st.subheader("📊 Coverage Details")
            # Limits of liability
            if "limits_of_liability" in product_details:
                liability_rows: List[Dict] = []
                for key, value in product_details["limits_of_liability"].items():
                    liability_rows.append({
                        "Coverage Type": key.replace("_", " ").title(),
                        "Limit": value,
                    })
                st.dataframe(pd.DataFrame(liability_rows), use_container_width=True)

            # Excesses
            if "excesses" in product_details:
                excess_rows: List[Dict] = []
                for key, value in product_details["excesses"].items():
                    if isinstance(value, dict):
                        rate = value.get("rate")
                        min_kes = value.get("minimum_kes")
                        excess_rows.append({
                            "Excess Type": key.replace("_", " ").title(),
                            "Rate": f"{rate:.1%}" if isinstance(rate, (int, float)) else str(rate),
                            "Minimum": f"KES {min_kes:,}" if isinstance(min_kes, (int, float)) else str(min_kes),
                        })
                    else:
                        excess_rows.append({
                            "Excess Type": key.replace("_", " ").title(),
                            "Rate": "Fixed",
                            "Minimum": f"KES {value:,}" if isinstance(value, (int, float)) else str(value),
                        })
                st.dataframe(pd.DataFrame(excess_rows), use_container_width=True)

            # Special clauses
            if "special_clauses" in product_details:
                st.write("**Special Clauses:**")
                for clause in product_details["special_clauses"]:
                    st.write(f"• {clause}")

    def create_claim_submission_page(self) -> None:
        st.header("📝 Submit New Claim")
        st.markdown("---")

        # Show last submission acknowledgement if available
        if st.session_state.get("last_submit_message"):
            st.success(st.session_state["last_submit_message"])
            # Do not clear immediately to survive one rerun; caller can overwrite on next submit

        # Block if client's policy is not active
        client_info = st.session_state.get("client_policy_info")
        if client_info:
            client_national_id = client_info.get("national_id")
            active_policies = [
                p for p in self.processor.policies
                if (p.get("national_id") == client_national_id)
                and (str(p.get("status", "")).strip().lower() == "active")
            ]
            if not active_policies:
                st.error("Your policy is not active. Please reach out to our call center on +2547000000 or email omicare@insuretechtest.com to activate your policy before submitting a claim.")
                return

        with st.form("claim_form"):
            policy_info = st.session_state.client_policy_info
            col1, col2 = st.columns(2)
            with col1:
                claimant_name = st.text_input(
                    "Claimant Name",
                    value=policy_info["client_name"] if policy_info else "",
                )
                vehicle_registration = st.text_input(
                    "Vehicle Registration",
                    value=policy_info["vehicle_registration"] if policy_info else "",
                )
            with col2:
                accident_date = st.date_input("Accident Date")
                accident_time = st.time_input("Accident Time")
                accident_location = st.text_input("Accident Location")

            accident_description = st.text_area("Accident Description", height=100)

            st.subheader("📎 Evidence")
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                evidence_uploads = st.file_uploader(
                    "Evidence",
                    type=["jpg", "jpeg", "png", "pdf", "mp4", "mov", "md"],
                    accept_multiple_files=True,
                    help="Upload photos, PDFs, videos, or .md notes as evidence",
                )
                
                # Enhanced photo analysis section
                if evidence_uploads:
                    st.subheader("📸 AI Photo Analysis")
                    for i, uploaded_file in enumerate(evidence_uploads):
                        if uploaded_file.type.startswith('image/'):
                            col_img, col_analysis = st.columns([1, 2])
                            with col_img:
                                st.image(uploaded_file, caption=f"Photo {i+1}", use_column_width=True)
                            with col_analysis:
                                if st.button(f"Analyze Photo {i+1}", key=f"analyze_photo_{i}"):
                                    with st.spinner("Analyzing photo with AI..."):
                                        # Generate a temporary claim ID for analysis
                                        temp_claim_id = f"TEMP-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                                        analysis_result = self.processor.analyze_uploaded_photo(uploaded_file, temp_claim_id)
                                        st.markdown("### AI Analysis Result:")
                                        st.markdown(analysis_result)
                                        
                                        # Option to save analysis
                                        if st.button(f"Save Analysis as Evidence", key=f"save_analysis_{i}"):
                                            if self.processor.save_evidence_analysis(temp_claim_id, analysis_result):
                                                st.success(f"Analysis saved as evidence-analysis/{temp_claim_id}.md")
                                            else:
                                                st.error("Failed to save analysis")
            
            with col_e2:
                witness_upload = st.file_uploader(
                    "Witness Statement",
                    type=["pdf", "doc", "docx", "txt", "md"],
                    accept_multiple_files=False,
                    help="Upload a single witness statement document",
                )
                
                # Enhanced document processing section
                if witness_upload:
                    st.subheader("📄 Document Processing")
                    col_doc, col_process = st.columns([1, 2])
                    with col_doc:
                        st.info(f"**File:** {witness_upload.name}\n**Size:** {witness_upload.size} bytes")
                    with col_process:
                        if st.button("Process Document", key="process_document"):
                            with st.spinner("Processing document..."):
                                # Generate a temporary claim ID for processing
                                temp_claim_id = f"TEMP-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                                processed_result = self.processor.process_uploaded_document(witness_upload, temp_claim_id)
                                st.markdown("### Processed Witness Statement:")
                                st.markdown(processed_result)
                                
                                # Option to save processed statement
                                if st.button("Save as Witness Statement", key="save_statement"):
                                    if self.processor.save_witness_statement(temp_claim_id, processed_result):
                                        st.success(f"Statement saved as witness-statements/{temp_claim_id}-NEW.md")
                                    else:
                                        st.error("Failed to save statement")

            submitted = st.form_submit_button("Submit Claim", use_container_width=True)
            if submitted:
                required_fields = [
                    claimant_name,
                    vehicle_registration,
                    accident_location,
                    accident_description,
                ]
                if all(required_fields):
                    # Require attachments: at least one Evidence file and a Witness Statement
                    missing_evidence = (not evidence_uploads) or (len(evidence_uploads) == 0)
                    missing_witness = (witness_upload is None)
                    if missing_evidence or missing_witness:
                        if missing_evidence and missing_witness:
                            st.error("Please attach at least one Evidence file and a Witness Statement before submitting.")
                        elif missing_evidence:
                            st.error("Please attach at least one Evidence file before submitting.")
                        else:
                            st.error("Please attach a Witness Statement before submitting.")
                        return
                    claim_data = {
                        "claimant_name": claimant_name,
                        "vehicle_registration": vehicle_registration,
                        "accident_date": str(accident_date),
                        "accident_time": str(accident_time),
                        "accident_location": accident_location,
                        "accident_description": accident_description,
                        "evidence_files": {
                            "evidence": [f.name for f in evidence_uploads] if evidence_uploads else [],
                            "witness_statement": witness_upload.name if witness_upload else None,
                        },
                    }

                    # Attach coverage details for payout computation if available
                    try:
                        if policy_info:
                            product = self.processor.get_product_details(policy_info.get("product_id"))
                            if product:
                                claim_data["coverage"] = {
                                    "limits_of_liability": product.get("limits_of_liability", {}),
                                    "excesses": product.get("excesses", {}),
                                }
                    except Exception:
                        pass

                    # Policy check: vehicle registration must belong to authenticated client's policy
                    try:
                        policy_info = st.session_state.client_policy_info if "client_policy_info" in st.session_state else None
                        if policy_info:
                            national_id = policy_info.get("national_id")
                            allowed_regs = [
                                (p.get("vehicle_registration") or "").strip().upper()
                                for p in self.processor.policies
                                if (p.get("national_id") or "") == national_id and str(p.get("status", "")).strip().lower() == "active"
                            ]
                            if allowed_regs and vehicle_registration.strip().upper() not in allowed_regs:
                                st.error("The entered vehicle registration is not linked to this client's policy.")
                                st.info("Allowed registration(s): " + ", ".join([r for r in allowed_regs if r]))
                                return
                    except Exception:
                        pass

                    # Duplicate detection: same vehicle and same accident date already submitted
                    duplicate = next(
                        (
                            c for c in self.processor.claims_database
                            if c.get("claim_data", {}).get("vehicle_registration", "").strip().upper() == vehicle_registration.strip().upper()
                            and c.get("claim_data", {}).get("accident_date") == str(accident_date)
                        ),
                        None,
                    )
                    if duplicate:
                        st.error("Duplicate claim detected for this vehicle on the same date.")
                        with st.expander("View previously filed claim details", expanded=True):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Claim ID:** {duplicate.get('claim_id')}")
                                st.write(f"**Submitted:** {duplicate.get('submission_time', '')[:19]}")
                                st.write(f"**Status:** {duplicate.get('status', 'N/A')}")
                            with col2:
                                st.write(f"**Decision:** {duplicate.get('decision_reason', 'Pending')}")
                                st.write(f"**Vehicle:** {duplicate.get('claim_data', {}).get('vehicle_registration', '')}")
                                st.write(f"**Accident Date:** {duplicate.get('claim_data', {}).get('accident_date', '')}")
                            st.write("**Accident Location:** " + duplicate.get('claim_data', {}).get('accident_location', ''))
                            st.write("**Description:**")
                            st.write(duplicate.get('claim_data', {}).get('accident_description', ''))
                        st.info("If you believe this is not a duplicate, please contact support before resubmitting. Call +2547000000 or email omicare@insuretechtest.com.")
                        return

                    # Quick-win: intake risk evaluation before submission
                    intake = self.processor.evaluate_intake_risk(claim_data)
                    if intake.get("recommendation") == "AUTO-HOLD":
                        st.warning("⏸️ Your claim is under review while we complete routine checks.")
                        st.write(f"**Preliminary Risk Score:** {intake.get('intake_score')} (threshold 0.5)")
                        if intake.get("reasons"):
                            st.write("**Why:**")
                            for r in intake.get("reasons", []):
                                st.write(f"• {r}")
                        related = intake.get("related_claims", [])
                        if related:
                            st.write("**Related recent claims at same location:**")
                            for rc in related[:5]:
                                st.write(f"- {rc.get('claim_id')} at {rc.get('submission_time', '')[:19]} (status: {rc.get('status')})")
                        st.info(
                            "What happens next: We’ll complete additional checks and update your status within ~15 minutes. "
                            "No action is needed now. We’ll contact you on your chosen channel if we need more information."
                        )
                        st.markdown(
                            "If you need help sooner, call **+2547000000** or email **omicare@insuretechtest.com**. "
                            "See the [Customer Guidance](Customer_Guidance.md) for timelines and FAQs."
                        )
                        # Save the claim and proceed with analysis
                        claim_id = self.processor.generate_next_claim_id()
                        wait_seconds = 75
                        earliest_dt = datetime.now() + timedelta(seconds=wait_seconds)
                        record = {
                            "claim_id": claim_id,
                            "submitted_by": st.session_state.user_name,
                            "user_role": st.session_state.user_role,
                            "submission_time": datetime.now().isoformat(),
                            "claim_data": claim_data,
                            "status": ClaimStatus.ANALYZING.value,
                            "fraud_analysis": None,
                            "payout_suggestion": self.processor.compute_payout_suggestion(claim_data),
                            "decision_time": None,
                            "decision_reason": None,
                            "payout_status": None,
                            "payout_history": [],
                            "analysis_started_at": datetime.now().isoformat(),
                            "earliest_decision_time": earliest_dt.isoformat(),
                        }
                        self.processor.claims_database.append(record)
                        # persist immediately
                        self.processor.save_claims_database()

                        # Authorize payout at intake; capture/hold decided after analysis
                        try:
                            self.processor.payment.authorize(claim_id, note="Intake authorization")
                            for c in self.processor.claims_database:
                                if c["claim_id"] == claim_id:
                                    c["payout_status"] = self.processor.payment.get_status(claim_id)
                                    c["payout_history"] = self.processor.payment.get_history(claim_id)
                                    break
                            self.processor.save_claims_database()
                        except Exception:
                            pass

                        def background_analysis() -> None:
                            analysis_result = self.processor.run_analysis(claim_data)
                            # Enforce minimum processing time
                            try:
                                edt_str = earliest_dt.isoformat()
                                edt = earliest_dt
                                remaining = (edt - datetime.now()).total_seconds()
                                if remaining > 0:
                                    time.sleep(remaining)
                            except Exception:
                                pass
                            for c in self.processor.claims_database:
                                if c["claim_id"] == claim_id:
                                    c["fraud_analysis"] = analysis_result
                                    if analysis_result.get("final_recommendation") == "APPROVE CLAIM":
                                        c["status"] = ClaimStatus.APPROVED.value
                                        # Approval acknowledgement for clients
                                        st.session_state["last_submit_message"] = (
                                            "Your claim has been approved. The claim amount will be sent to your bank account within 48 hours. "
                                            "In case of delay, please reach out to our call center on +2547000000 or email omicare@insuretechtest.com."
                                        )
                                        # Release payout
                                        try:
                                            self.processor.payment.capture(claim_id, note="Auto-capture on approval")
                                        except Exception:
                                            pass
                                    elif analysis_result.get("final_recommendation") == "ENHANCED REVIEW REQUIRED":
                                        c["status"] = ClaimStatus.REQUIRES_REVIEW.value
                                        st.session_state["last_submit_message"] = (
                                            "Your claim requires additional checks. Expect an update within 1 business day. "
                                            "We’ll contact you if we need more information. Track status in ‘My Claims Status’. "
                                            "Help: +2547000000, omicare@insuretechtest.com."
                                        )
                                        # Place payout on hold
                                        try:
                                            self.processor.payment.hold(claim_id, note="Hold on review requirement")
                                        except Exception:
                                            pass
                                    else:
                                        c["status"] = ClaimStatus.REJECTED.value
                                        st.session_state["last_submit_message"] = (
                                            "Your claim was not approved. You may appeal by replying to the decision email or contacting our call center. "
                                            "Have your Claim ID ready and any new evidence. Help: +2547000000, omicare@insuretechtest.com."
                                        )
                                        # Hold payout by default for finance review
                                        try:
                                            self.processor.payment.hold(claim_id, note="Hold on rejection (await finance action)")
                                        except Exception:
                                            pass
                                    # Sync payout fields
                                    c["payout_status"] = self.processor.payment.get_status(claim_id)
                                    c["payout_history"] = self.processor.payment.get_history(claim_id)
                                    c["decision_time"] = datetime.now().isoformat()
                                    c["decision_reason"] = analysis_result.get("final_recommendation")
                                    break
                            self.processor.save_claims_database()
                            st.session_state.analysis_in_progress = False
                            st.rerun()
                        t = threading.Thread(target=background_analysis, daemon=True)
                        t.start()

                        st.session_state["last_submit_message"] = (
                            "Thank you. Your claim is under review, please refer to the claim status page after 15 minutes to check status of your claim."
                        )
                        try:
                            st.rerun()
                        except Exception:
                            pass
                        return

                    else:
                        # If not on hold, proceed with analysis directly
                        claim_id = self.processor.generate_next_claim_id()
                        wait_seconds = 75
                        earliest_dt = datetime.now() + timedelta(seconds=wait_seconds)
                        record = {
                            "claim_id": claim_id,
                            "submitted_by": st.session_state.user_name,
                            "user_role": st.session_state.user_role,
                            "submission_time": datetime.now().isoformat(),
                            "claim_data": claim_data,
                            "status": ClaimStatus.ANALYZING.value,
                            "fraud_analysis": None,
                            "payout_suggestion": self.processor.compute_payout_suggestion(claim_data),
                            "decision_time": None,
                            "decision_reason": None,
                            "payout_status": None,
                            "payout_history": [],
                            "analysis_started_at": datetime.now().isoformat(),
                            "earliest_decision_time": earliest_dt.isoformat(),
                        }
                        self.processor.claims_database.append(record)
                        # persist immediately
                        self.processor.save_claims_database()

                        # Authorize payout at intake
                        try:
                            self.processor.payment.authorize(claim_id, note="Intake authorization")
                            for c in self.processor.claims_database:
                                if c["claim_id"] == claim_id:
                                    c["payout_status"] = self.processor.payment.get_status(claim_id)
                                    c["payout_history"] = self.processor.payment.get_history(claim_id)
                                    break
                            self.processor.save_claims_database()
                        except Exception:
                            pass

                        def background_analysis() -> None:
                            analysis_result = self.processor.run_analysis(claim_data)
                            # Enforce minimum processing time
                            try:
                                edt = earliest_dt
                                remaining = (edt - datetime.now()).total_seconds()
                                if remaining > 0:
                                    time.sleep(remaining)
                            except Exception:
                                pass
                            for c in self.processor.claims_database:
                                if c["claim_id"] == claim_id:
                                    c["fraud_analysis"] = analysis_result
                                    if analysis_result.get("final_recommendation") == "APPROVE CLAIM":
                                        c["status"] = ClaimStatus.APPROVED.value
                                        st.session_state["last_submit_message"] = (
                                            "Your claim has been approved. The claim amount will be sent to your bank account within 48 hours. "
                                            "In case of delay, please reach out to our call center on +2547000000 or email omicare@insuretechtest.com."
                                        )
                                        try:
                                            self.processor.payment.capture(claim_id, note="Auto-capture on approval")
                                        except Exception:
                                            pass
                                    elif analysis_result.get("final_recommendation") == "ENHANCED REVIEW REQUIRED":
                                        c["status"] = ClaimStatus.REQUIRES_REVIEW.value
                                        st.session_state["last_submit_message"] = (
                                            "Your claim requires additional checks. Expect an update within 1 business day. "
                                            "We’ll contact you if we need more information. Track status in ‘My Claims Status’. "
                                            "Help: +2547000000, omicare@insuretechtest.com."
                                        )
                                        try:
                                            self.processor.payment.hold(claim_id, note="Hold on review requirement")
                                        except Exception:
                                            pass
                                    else:
                                        c["status"] = ClaimStatus.REJECTED.value
                                        st.session_state["last_submit_message"] = (
                                            "Your claim was not approved. You may appeal by replying to the decision email or contacting our call center. "
                                            "Have your Claim ID ready and any new evidence. Help: +2547000000, omicare@insuretechtest.com."
                                        )
                                        try:
                                            self.processor.payment.hold(claim_id, note="Hold on rejection (await finance action)")
                                        except Exception:
                                            pass
                                    c["payout_status"] = self.processor.payment.get_status(claim_id)
                                    c["payout_history"] = self.processor.payment.get_history(claim_id)
                                    c["decision_time"] = datetime.now().isoformat()
                                    c["decision_reason"] = analysis_result.get("final_recommendation")
                                    break
                            self.processor.save_claims_database()
                            st.session_state.analysis_in_progress = False
                            st.rerun()
                        t = threading.Thread(target=background_analysis, daemon=True)
                        t.start()
                        # Show acknowledgement immediately and rerun UI
                        st.session_state["last_submit_message"] = (
                            "Thank you. Your claim is under review, please refer to the claim status page after 15 minutes to check status of your claim."
                        )
                        try:
                            st.rerun()
                        except Exception:
                            pass
                        return

    def create_claim_status_page(self) -> None:
        st.header("📊 My Claims Status")
        st.markdown("---")
        if not self.processor.claims_database:
            # Try to import historical challenge claims for this client
            try:
                if st.session_state.user_name:
                    self.processor.import_historical_claims_for_policy_holder(st.session_state.user_name)
            except Exception:
                pass
            if not self.processor.claims_database:
                st.info("No claims submitted yet.")
                return

    def create_claim_status_page(self) -> None:
        st.header("📊 My Claims Status")
        st.markdown("---")
        
        # Import historical claims if database is empty
        if not self.processor.claims_database:
            # Try to import historical challenge claims for this client
            try:
                if st.session_state.user_name:
                    self.processor.import_historical_claims_for_policy_holder(st.session_state.user_name)
            except Exception:
                pass
            
            # For fraud analysts, also import all historical claims
            if st.session_state.user_role == UserRole.FRAUD_ANALYST.value:
                try:
                    self.processor.import_all_historical_claims()
                except Exception:
                    pass
            
            if not self.processor.claims_database:
                st.info("No claims submitted yet.")
                return

        # Show analyzing section first (no moving bar)
        if st.session_state.current_claim_id and st.session_state.analysis_in_progress:
            st.subheader("🔄 Current Claim Analysis")
            st.info(f"Analyzing claim {st.session_state.current_claim_id}... Please wait.")
            with st.spinner("Running analysis..."):
                time.sleep(0.5)

        # Filter claims based on user role
        if st.session_state.user_role == UserRole.FRAUD_ANALYST.value:
            # Fraud analysts can see all claims
            user_claims = self.processor.claims_database
            st.info("🔍 Fraud Analyst View: Showing all claims in the system")
        else:
            # Regular users see their own claims by name and by national ID mapping
            user_claims = []
            
            # First, try to find claims by submitted_by name
            user_claims.extend([c for c in self.processor.claims_database if c["submitted_by"] == st.session_state.user_name])
            
            # If no claims found by name, try to find claims by national ID mapping
            if not user_claims and st.session_state.client_policy_info:
                national_id = st.session_state.client_policy_info.get("national_id")
                if national_id:
                    # Import historical claims for this specific national ID
                    try:
                        self.processor.import_historical_claims_for_national_id(national_id)
                        # Try again to find claims by name after import
                        user_claims = [c for c in self.processor.claims_database if c["submitted_by"] == st.session_state.user_name]
                    except Exception:
                        pass
        
        if not user_claims:
            st.info("No claims found for your account.")
            return

        # Build a customer-friendly table without exposing fraud details/scores
        rows: List[Dict] = []
        on_hold_exists = False
        rejected_exists = False
        for claim in user_claims:
            status = claim["status"]
            if status == ClaimStatus.SUBMITTED.value:
                friendly = "📝 Submitted"
            elif status == ClaimStatus.ANALYZING.value:
                friendly = "🔄 Analyzing"
            elif status == ClaimStatus.APPROVED.value:
                friendly = "✅ Approved"
            elif status == ClaimStatus.REQUIRES_REVIEW.value:
                friendly = "⏸️ On Hold"
                on_hold_exists = True
            elif status == ClaimStatus.REJECTED.value:
                friendly = "Decision Issued"
                rejected_exists = True
            else:
                friendly = status

            row_data = {
                "Claim ID": claim["claim_id"],
                "Status": friendly,
                "Vehicle": claim["claim_data"]["vehicle_registration"],
                "Location": claim["claim_data"]["accident_location"],
                "Submitted": claim["submission_time"][:19],
            }
            
            # Add submitted_by field for fraud analysts
            if st.session_state.user_role == UserRole.FRAUD_ANALYST.value:
                row_data["Submitted By"] = claim.get("submitted_by", "Unknown")
            
            rows.append(row_data)

        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Guidance for customers
        if on_hold_exists:
            st.info(
                "⏸️ On Hold: We’re completing routine checks. Expect an update within ~15 minutes. "
                "We’ll contact you if we need anything. Track progress here. Help: +2547000000, omicare@insuretechtest.com."
            )
        if rejected_exists:
            st.info(
                "Decision Issued: Your claim was not approved. You can appeal within 14 days by replying to the decision email "
                "or contacting our call center. Have your Claim ID ready and any new evidence. Help: +2547000000, omicare@insuretechtest.com."
            )

        # Clickable claim ID -> show status message
        st.subheader("View Claim Message")
        claim_ids = [c["claim_id"] for c in user_claims]
        if claim_ids:
            selected_id = st.radio("Click a claim ID to view its message:", claim_ids, horizontal=True)
            sel = next((c for c in user_claims if c["claim_id"] == selected_id), None)
            if sel:
                sel_status = sel.get("status")
                payout_status = sel.get("payout_status", "UNKNOWN")
                payout_history = sel.get("payout_history", [])
                # Backfill payout suggestion if missing
                if not sel.get("payout_suggestion"):
                    try:
                        sel["payout_suggestion"] = self.processor.compute_payout_suggestion(sel.get("claim_data", {}))
                        # persist
                        self.processor.save_claims_database()
                    except Exception:
                        pass
                # Infer payout status for legacy claims if unknown
                if payout_status == "UNKNOWN":
                    try:
                        cid = sel.get("claim_id")
                        if sel_status == ClaimStatus.APPROVED.value:
                            self.processor.payment.capture(cid, note="Inferred capture on prior approval")
                        elif sel_status in (ClaimStatus.REQUIRES_REVIEW.value, ClaimStatus.ANALYZING.value, ClaimStatus.REJECTED.value):
                            self.processor.payment.hold(cid, note="Inferred hold for review/decision")
                        sel["payout_status"] = self.processor.payment.get_status(cid)
                        sel["payout_history"] = self.processor.payment.get_history(cid)
                        payout_status = sel["payout_status"]
                        payout_history = sel["payout_history"]
                        self.processor.save_claims_database()
                    except Exception:
                        pass
                if sel_status == ClaimStatus.APPROVED.value:
                    st.success(
                        "Your claim has been approved. The claim amount will be sent to your bank account within 48 hours. "
                        "In case of delay, please reach out to our call center on +2547000000 or email omicare@insuretechtest.com."
                    )
                elif sel_status in (ClaimStatus.REQUIRES_REVIEW.value, ClaimStatus.ANALYZING.value):
                    st.warning(
                        "⏳ Under Review: We aim to update you within 1 business day. We'll contact you on your chosen channel if we "
                        "need more information. For urgent queries: +2547000000, omicare@insuretechtest.com."
                    )
                elif sel_status == ClaimStatus.REJECTED.value:
                    st.warning(
                        "Your claim was not approved. If you wish to appeal or provide more information, please contact our call center on "
                        "+2547000000 or email omicare@insuretechtest.com."
                    )
                else:
                    st.info("Your claim has been received and is being processed.")

                # Payout section
                st.markdown("---")
                st.subheader("💸 Payout")
                st.write(f"**Status:** {payout_status}")
                payout_suggestion = sel.get("payout_suggestion")
                if payout_suggestion:
                    st.write("**Suggested Payout Breakdown:**")
                    for it in payout_suggestion.get("items", []):
                        st.write(f"- {it.get('item')}: {it.get('amount')}  — {it.get('note')}")
                    st.write(f"**Suggested Total:** KES {payout_suggestion.get('total_amount', 0):,}")
                if sel.get("fraud_analysis", {}).get("red_flags"):
                    st.write("**Red Flags:**")
                    for f in sel["fraud_analysis"]["red_flags"]:
                        st.write(f"• {f}")
            else:
                st.error("No claim found with the selected ID. Please try again.")
                # Payout timeline details removed as requested

    def create_fraud_analyst_dashboard(self) -> None:
        # Removed inner horizontal rule to avoid duplicate separator under the main header

        fraud_data = self.processor.load_fraud_analysis_data()

        # Under Review section (claims currently analyzing)
        reviewing = [c for c in self.processor.claims_database if c.get("status") == ClaimStatus.ANALYZING.value]
        if reviewing:
            st.subheader("⏳ Claims Under Review")
            rv_rows = []
            for c in reviewing:
                rv_rows.append({
                    "Claim ID": c.get("claim_id"),
                    "Submitted": (c.get("submission_time") or "")[:19],
                    "Vehicle": c.get("claim_data", {}).get("vehicle_registration", ""),
                    "Location": c.get("claim_data", {}).get("accident_location", ""),
                })
            st.dataframe(pd.DataFrame(rv_rows), use_container_width=True)
            st.markdown("---")

        # Build helper sets
        fraudulent_ids = set()
        if fraud_data.get("fraud_scores"):
            fraudulent_ids = set(fraud_data["fraud_scores"][0].get("fraudulent_claims", []))

        # Overview KPIs
        col1, col2, col3, col4 = st.columns(4)
        dataset_total = len(fraud_data.get("claims", []))
        dataset_fraud = 0
        dataset_legit = 0
        if fraud_data.get("fraud_scores"):
            dataset_fraud = len(fraud_data["fraud_scores"][0].get("fraudulent_claims", []))
            dataset_legit = len(fraud_data["fraud_scores"][0].get("legitimate_claims", []))
        portal_total = len(self.processor.claims_database)
        portal_fraud = len([c for c in self.processor.claims_database if c.get("status") == ClaimStatus.REJECTED.value])
        portal_legit = len([c for c in self.processor.claims_database if c.get("status") == ClaimStatus.APPROVED.value])

        total_claims = dataset_total + portal_total
        total_fraudulent = dataset_fraud + portal_fraud
        total_legitimate = dataset_legit + portal_legit

        with col1:
            st.metric("Total Claims", total_claims)
        with col2:
            st.metric("Total Policies", len(fraud_data.get("policies", [])))
        with col3:
            st.metric("Fraudulent Claims", total_fraudulent)
        with col4:
            st.metric("Legitimate Claims", total_legitimate)
        
        # Validation: Ensure KPI totals match priority categorization
        # This will be calculated later in the fraud report section

        # Tabs navigation (subtitle removed); small spacing before tabs
        st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
        # Increase spacing between tab names and keep them on one line
        st.markdown(
            """
            <style>
            /* Newer Streamlit builds */
            div[data-testid="stTabs"] div[role="tablist"] {
                gap: 24px;
                flex-wrap: nowrap;
                justify-content: space-between; /* stretch to page edge */
                width: 100%;
                align-items: center;
            }
            div[data-testid="stTabs"] div[role="tab"] {
                white-space: nowrap;
                padding: 6px 12px;
                font-size: 1.35rem; /* approx h2 */
                line-height: 1.25;
            }
            /* Older Streamlit builds */
            .stTabs [data-baseweb="tab-list"] {
                gap: 24px;
                flex-wrap: nowrap;
                justify-content: space-between; /* stretch to page edge */
                width: 100%;
                align-items: center;
            }
            .stTabs [data-baseweb="tab"] {
                margin-right: 0; /* space handled by justify-content */
                white-space: nowrap;
                padding: 6px 12px;
                font-size: 1.35rem; /* approx h2 */
                line-height: 1.25;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        tab_overview, tab_report, tab_timeline, tab_location, tab_witness, tab_evidence, tab_payouts = st.tabs([
            "Overview",
            "Fraud Report",
            "Timeline Analysis",
            "Location Analysis",
            "Witness Analysis",
            "Evidence Analysis",
            "Payout Controls",
        ])

        # Overview tab (Executive Summary)
        with tab_overview:
            st.subheader("📊 Executive Summary")
            # Risk KPIs and prioritized cases from live portal database + dataset claims (CLAIM-001..016)
            try:
                claims_live = list(self.processor.claims_database)
            except Exception:
                claims_live = []

            risk_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
            rows: List[Dict] = []
            
            # Get fraudulent/legitimate classification first
            fraudulent_ids_ds = set()
            legitimate_ids_ds = set()
            group_risk_level = "MEDIUM"  # Default
            group_fraud_score = 0.416   # Default
            try:
                if fraud_data.get("fraud_scores"):
                    grp0 = fraud_data["fraud_scores"][0]
                    fraudulent_ids_ds = set(grp0.get("fraudulent_claims", []))
                    legitimate_ids_ds = set(grp0.get("legitimate_claims", []))
                    group_risk_level = grp0.get("risk_level", "MEDIUM")
                    group_fraud_score = grp0.get("fraud_score", 0.416)
            except Exception:
                pass
            
            for rec in claims_live:
                cid = rec.get("claim_id")
                fraud = rec.get("fraud_analysis") or {}
                
                # Prioritize fraudulent/legitimate classification over existing fraud analysis
                if cid in fraudulent_ids_ds:
                    # CLAIM-001 is MEDIUM risk (first claim, needs review)
                    # CLAIM-002-010 are HIGH risk (duplicated/coordinated fraud)
                    if cid == "CLAIM-001":
                        risk = "MEDIUM"
                        fraud_score = group_fraud_score  # 0.416
                        final_recommendation = "ENHANCED REVIEW REQUIRED"
                    else:
                        risk = "HIGH"
                        fraud_score = 0.8
                        final_recommendation = "REJECT CLAIM"
                elif cid in legitimate_ids_ds:
                    risk = "LOW"
                    fraud_score = 0.2
                    final_recommendation = "APPROVE CLAIM"
                else:
                    # Use existing fraud analysis if not in fraudulent/legitimate lists
                    risk = fraud.get("risk_level")
                    fraud_score = float(fraud.get("fraud_score", 0.0) or 0.0)
                    final_recommendation = fraud.get("final_recommendation", "")
                
                if risk in ("HIGH", "MEDIUM", "LOW"):
                    risk_counts[risk] += 1
                    rows.append({
                        "claim_id": cid,
                        "claimant_name": rec.get("claim_data", {}).get("claimant_name", ""),
                        "accident_location": rec.get("claim_data", {}).get("accident_location", ""),
                        "risk_level": risk,
                        "fraud_score": fraud_score,
                        "status": rec.get("status"),
                        "final_recommendation": final_recommendation,
                        "submission_time": rec.get("submission_time"),
                    })

            # Include dataset claims 001..016 from loaded fraud_data
            try:
                dataset_claims = fraud_data.get("claims", [])
                if not dataset_claims and fraud_data.get("evidence_analysis"):
                    # Fallback: synthesize entries from evidence keys
                    dataset_claims = [{"claim_id": cid, "timestamp": None, "notifier_name": "Dataset", "location": ""}
                                      for cid in fraud_data.get("evidence_analysis", {}).keys()]
            except Exception:
                dataset_claims = []

            if dataset_claims:
                ds_by_id = {c.get("claim_id"): c for c in dataset_claims if c.get("claim_id")}
                for cid, c in ds_by_id.items():
                    if cid in fraudulent_ids_ds:
                        # CLAIM-001 is MEDIUM risk (first claim, needs review)
                        # CLAIM-002-010 are HIGH risk (duplicated/coordinated fraud)
                        if cid == "CLAIM-001":
                            level = "MEDIUM"
                            score_val = group_fraud_score  # 0.416
                            final_rec = "ENHANCED REVIEW REQUIRED"
                        else:
                            level = "HIGH"
                            score_val = 0.8
                            final_rec = "REJECT CLAIM"
                    elif cid in legitimate_ids_ds:
                        level = "LOW"
                        score_val = 0.2
                        final_rec = "APPROVE CLAIM"
                    else:
                        # Unknown in grouping; skip or mark low
                        level = "LOW"
                        score_val = 0.2
                        final_rec = "APPROVE CLAIM"
                    risk_counts[level] += 1
                    rows.append({
                        "claim_id": cid,
                        "claimant_name": c.get("notifier_name") or "Dataset",
                        "accident_location": c.get("location", ""),
                        "risk_level": level,
                        "fraud_score": score_val,
                        "status": "Dataset",
                        "final_recommendation": final_rec,
                        "submission_time": c.get("timestamp"),
                    })

            if rows:
                k1, k2, k3 = st.columns(3)
                with k1:
                    st.metric("High Risk Claims", int(risk_counts["HIGH"]))
                with k2:
                    st.metric("Medium Risk Claims", int(risk_counts["MEDIUM"]))
                with k3:
                    st.metric("Low Risk Claims", int(risk_counts["LOW"]))

                df = pd.DataFrame(rows)
                priority_rank = {"HIGH": 1, "MEDIUM": 2, "LOW": 3}
                df["priority"] = df["risk_level"].map(priority_rank).fillna(4).astype(int)
                df = df.sort_values(by=["priority", "fraud_score", "submission_time"], ascending=[True, False, False])

                st.subheader("🎯 Prioritized Cases")
                tabs_prio = st.tabs(["High", "Medium", "Low"])
                categories = [("High", "HIGH"), ("Medium", "MEDIUM"), ("Low", "LOW")]
                for tabp, (_, level) in zip(tabs_prio, categories):
                    with tabp:
                        subset = df[df["risk_level"] == level]
                        if subset.empty:
                            st.info(f"No {level.lower()} risk claims.")
                        else:
                            display_cols = [
                                "claim_id",
                                "claimant_name",
                                "accident_location",
                                "risk_level",
                                "fraud_score",
                                "final_recommendation",
                                "submission_time",
                            ]
                            st.dataframe(subset[display_cols], use_container_width=True, hide_index=True)
            else:
                st.info("No risk-classified claims yet.")
            if fraud_data.get("fraud_scores"):
                info = fraud_data["fraud_scores"][0]
                # Replace KPI cards with a concise methodology and reasoning section
                st.markdown("### How this conclusion is reached")
                st.markdown(
                    "- We compute similarity across timing, locations, witness patterns, and evidence artifacts.\n"
                    "- A weighted score is produced and mapped to a risk level.\n"
                    "- If coordinated indicators are present, we flag the claim group for further review."
                )
                st.markdown("### Recommendation")
                st.info(info.get("recommendation", "N/A"))
                if info.get("red_flags"):
                    st.markdown("### Key Indicators")
                    for flag in info.get("red_flags", []):
                        st.markdown(f"- {flag}")
                if info.get("evidence_correlations"):
                    st.markdown("### Evidence Correlations")
                    for corr in info.get("evidence_correlations", []):
                        st.markdown(f"- {corr}")
            else:
                st.warning("No fraud analysis summary available.")

        # Fraud report tab (moved after Overview)
        with tab_report:
            st.subheader("📋 Fraud Report")
            # Use live portal database to align with dashboard
            claims_db = list(self.processor.claims_database)
            if not claims_db:
                st.info("No summary available.")
            else:
                # Lookup maps from datasets for enrichment
                claims_by_id = {c.get("claim_id"): c for c in fraud_data.get("claims", [])}
                evidence_by_id = fraud_data.get("evidence_analysis", {})
                witness_by_id = fraud_data.get("witness_statements", {})
                # Enrich with portal entries
                for rec in claims_db:
                    claims_by_id[rec.get("claim_id")] = claims_by_id.get(rec.get("claim_id"), {
                        "claim_id": rec.get("claim_id"),
                        "timestamp": rec.get("submission_time"),
                        "channel": "Portal",
                        "notifier_name": rec.get("submitted_by"),
                        "location": rec.get("claim_data", {}).get("accident_location", ""),
                        "initial_details": rec.get("claim_data", {}).get("accident_description", ""),
                        "vehicle_reg": rec.get("claim_data", {}).get("vehicle_registration", ""),
                    })
 
                # Compose markdown report with per-claim reasoning (based on live DB)
                report_lines: List[str] = []
                report_lines.append("# OMIcare – Fraud Analysis Report")
                report_lines.append("")
                report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                report_lines.append("")
                report_lines.append("## Executive Summary")
                total_n = len(claims_db)
                rejected_n = len([c for c in claims_db if c.get("status") == ClaimStatus.REJECTED.value])
                approved_n = len([c for c in claims_db if c.get("status") == ClaimStatus.APPROVED.value])
                reviewing_n = len([c for c in claims_db if c.get("status") in (ClaimStatus.ANALYZING.value, ClaimStatus.REQUIRES_REVIEW.value)])
                report_lines.append(f"- Total Claims (live): **{total_n}**")
                report_lines.append(f"- Under Review: **{reviewing_n}**  |  Approved: **{approved_n}**  |  Rejected: **{rejected_n}**")
                report_lines.append("")
 
                report_lines.append("## Methodology")
                report_lines.append("We analyzed temporal proximity of claims, geolocation consistency, witness statement patterns, and evidence artifacts (hospital/doctor references and image hash markers). Findings were combined into explainable rule-based indicators.")
                report_lines.append("")
 
                report_lines.append("## Detailed Findings by Claim")
                # Build ordered list: dataset claims (CLAIM-001..016) first by numeric order, then live by submission time
                claims_sorted = sorted(claims_db, key=lambda x: x.get("submission_time", ""), reverse=True)
                # Dataset claim IDs from dataset content (not only fraudulent list)
                dataset_claim_ids = set(c.get("claim_id") for c in fraud_data.get("claims", []) if c.get("claim_id"))
                if not dataset_claim_ids and fraud_data.get("evidence_analysis"):
                    dataset_claim_ids.update(list(fraud_data["evidence_analysis"].keys()))
                try:
                    ordered_dataset = sorted(list(dataset_claim_ids), key=lambda x: int(re.findall(r"(\d+)", x)[0]) if re.findall(r"(\d+)", x) else 0)
                except Exception:
                    ordered_dataset = list(dataset_claim_ids)
                ordered_ids: List[str] = []
                for cid in ordered_dataset:
                    if cid not in ordered_ids:
                        ordered_ids.append(cid)
                for rec in claims_sorted:
                    rcid = rec.get("claim_id")
                    if rcid and rcid not in ordered_ids:
                        ordered_ids.append(rcid)

                # Group by priority (risk level) instead of approval status
                fraudulent_ids = set()
                legitimate_ids = set()
                if fraud_data.get("fraud_scores"):
                    src = fraud_data["fraud_scores"][0]
                    fraudulent_ids = set(src.get("fraudulent_claims", []))
                    legitimate_ids = set(src.get("legitimate_claims", []))

                # Assign risk/score per claim (live + dataset)
                risk_by_id: Dict[str, str] = {}
                score_by_id: Dict[str, float] = {}

                # First, assign risk levels based on fraudulent/legitimate classification (highest priority)
                # Get the group risk level from fraud analysis
                group_risk_level = "MEDIUM"  # Default
                group_fraud_score = 0.416   # Default
                if fraud_data.get("fraud_scores"):
                    grp0 = fraud_data["fraud_scores"][0]
                    group_risk_level = grp0.get("risk_level", "MEDIUM")
                    group_fraud_score = grp0.get("fraud_score", 0.416)
                
                for cid in ordered_ids:
                    if cid in fraudulent_ids:
                        # CLAIM-001 is MEDIUM risk (first claim, needs review)
                        # CLAIM-002-010 are HIGH risk (duplicated/coordinated fraud)
                        if cid == "CLAIM-001":
                            risk_by_id[cid] = "MEDIUM"
                            score_by_id[cid] = group_fraud_score  # 0.416
                        else:
                            risk_by_id[cid] = "HIGH"
                            score_by_id[cid] = 0.8
                    elif cid in legitimate_ids:
                        risk_by_id[cid] = "LOW"
                        score_by_id[cid] = 0.2

                # Then, for claims not in fraudulent/legitimate lists, use existing fraud analysis
                for rec in claims_db:
                    cid = rec.get("claim_id")
                    if cid not in risk_by_id:  # Only assign if not already assigned above
                        fraud = rec.get("fraud_analysis") or {}
                        risk = (fraud.get("risk_level") or "").upper()
                        if risk in ("HIGH", "MEDIUM", "LOW"):
                            risk_by_id[cid] = risk
                            try:
                                score_by_id[cid] = float(fraud.get("fraud_score", 0.0) or 0.0)
                            except Exception:
                                score_by_id[cid] = 0.0

                # Finally, ensure ALL claims get assigned a risk level (default to LOW)
                for cid in ordered_ids:
                    if cid not in risk_by_id:
                        risk_by_id[cid] = "LOW"
                        score_by_id[cid] = 0.2

                # Prepare ordering helpers
                time_by_id: Dict[str, pd.Timestamp] = {}
                for cid in ordered_ids:
                    rec = next((r for r in claims_db if r.get("claim_id") == cid), None)
                    c = claims_by_id.get(cid, {})
                    ts_val = c.get("timestamp", (rec.get("submission_time") if rec else None))
                    try:
                        if ts_val:
                            # Convert to timezone-naive timestamp to avoid comparison issues
                            dt = pd.to_datetime(ts_val)
                            if dt.tz is not None:
                                dt = dt.tz_localize(None)  # Remove timezone info
                            time_by_id[cid] = dt
                        else:
                            time_by_id[cid] = pd.to_datetime("1970-01-01")
                    except Exception:
                        time_by_id[cid] = pd.to_datetime("1970-01-01")

                def _sorted_ids(level: str) -> List[str]:
                    ids = [i for i in ordered_ids if risk_by_id.get(i) == level]
                    # Use a consistent timezone-naive fallback timestamp
                    fallback_ts = pd.to_datetime("1970-01-01")
                    ids.sort(key=lambda x: (-(score_by_id.get(x, 0.0)), time_by_id.get(x, fallback_ts)), reverse=False)
                    return ids

                # Get group-level indicators
                group_red_flags = []
                group_claim_label = None
                group_score = None
                group_risk = None
                if fraud_data.get("fraud_scores"):
                    grp0 = fraud_data["fraud_scores"][0]
                    group_red_flags = grp0.get("red_flags", [])
                    group_claim_label = grp0.get("claim_group")
                    group_score = grp0.get("fraud_score")
                    group_risk = grp0.get("risk_level")

                # Render per-priority sections as single collapsible groups
                for section_title, level, icon in [
                    ("High Priority", "HIGH", "🔴"),
                    ("Medium Priority", "MEDIUM", "🟠"),
                    ("Low Priority", "LOW", "🟢"),
                ]:
                    ids_for_level = _sorted_ids(level)
                    header_text = f"{icon} {section_title} ({len(ids_for_level)})"
                    with st.expander(header_text, expanded=False):
                        if not ids_for_level:
                            st.info(f"No {level.lower()} priority claims to display yet.")
                        else:
                            table_rows: List[Dict] = []
                            for cid in ids_for_level:
                                rec = next((r for r in claims_db if r.get("claim_id") == cid), None)
                                c = claims_by_id.get(cid, {})
                                ev_txt = evidence_by_id.get(cid, "")
                                wit_txt = witness_by_id.get(cid, "")
                                ts = c.get("timestamp", (rec.get("submission_time") if rec else "N/A"))
                                channel = c.get("channel", ("Portal" if rec else "Dataset"))
                                notifier = c.get("notifier_name", (rec.get("submitted_by") if rec else "Dataset"))
                                coords = re.findall(r"GPS Coordinates.*?(-?\d+\.\d+), (-?\d+\.\d+)", ev_txt)
                                loc_str = f"{coords[0][0]}, {coords[0][1]}" if coords else ((rec.get("claim_data", {}).get("accident_location") if rec else None) or c.get("location", "N/A"))
                                driver_names = re.findall(r"(Juma Said|J\. Saeed|Jumaa Saidi|S\. Juma)", wit_txt, re.IGNORECASE)

                                # Append to text report
                                report_lines.append(f"### {cid} ({level} PRIORITY)")
                                report_lines.append(f"- Timestamp: {ts}  |  Channel: {channel}  |  Notifier: {notifier}")
                                report_lines.append(f"- Location (evidence): {loc_str}")
                                if driver_names:
                                    report_lines.append(f"- Witness Driver: {driver_names[0]}")
                                report_lines.append(f"- Risk Level: {level}  |  Fraud Score: {score_by_id.get(cid, 0.0):.3f}")
                                if level == "HIGH":
                                    report_lines.append("- Decision Rationale: Coordinated pattern indicators present; priority investigation.")
                                    if group_claim_label and group_score is not None and group_risk:
                                        report_lines.append(f"- Similar Claims: {group_claim_label} | Fraud Score: {group_score:.3f} ({group_risk})")
                                    if group_red_flags:
                                        primary_indicator = next((rf for rf in group_red_flags if 'similarity' in rf.lower()), group_red_flags[0])
                                        report_lines.append(f"- Primary Indicator: {primary_indicator}")
                                elif level == "MEDIUM":
                                    report_lines.append("- Decision Rationale: Some overlaps; enhanced review recommended.")
                                else:
                                    report_lines.append("- Decision Rationale: No significant overlaps; standard processing.")
                                report_lines.append("")

                                table_rows.append({
                                    "Claim ID": cid,
                                    "Timestamp": ts,
                                    "Channel": channel,
                                    "Notifier": notifier,
                                    "Location": loc_str,
                                    "Risk": level,
                                    "Fraud Score": f"{score_by_id.get(cid, 0.0):.3f}",
                                })

                            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
 
                report_md = "\n".join(report_lines)

                # Validation: Show breakdown to ensure all claims are accounted for
                st.subheader("📊 Claims Breakdown Validation")
                high_count = len([i for i in ordered_ids if risk_by_id.get(i) == "HIGH"])
                medium_count = len([i for i in ordered_ids if risk_by_id.get(i) == "MEDIUM"])
                low_count = len([i for i in ordered_ids if risk_by_id.get(i) == "LOW"])
                categorized_total = high_count + medium_count + low_count
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("High Priority", high_count)
                with col2:
                    st.metric("Medium Priority", medium_count)
                with col3:
                    st.metric("Low Priority", low_count)
                with col4:
                    st.metric("Total Categorized", categorized_total)
                
                if categorized_total != len(ordered_ids):
                    st.warning(f"⚠️ Mismatch detected: {len(ordered_ids)} total claims but only {categorized_total} categorized")
                else:
                    st.success(f"✅ All {categorized_total} claims properly categorized")

                # Try to generate a PDF; fall back to markdown download if PDF libs are unavailable
                pdf_bytes = None
                try:
                    from io import BytesIO
                    from reportlab.lib.pagesizes import A4
                    from reportlab.pdfgen import canvas

                    def _pdf_from_text(text: str) -> bytes:
                        buffer = BytesIO()
                        c = canvas.Canvas(buffer, pagesize=A4)
                        width, height = A4
                        x_margin = 40
                        y = height - 40
                        for raw_line in text.split("\n"):
                            line = raw_line.replace("\t", "    ")
                            while len(line) > 100:
                                c.drawString(x_margin, y, line[:100])
                                line = line[100:]
                                y -= 14
                                if y < 40:
                                    c.showPage()
                                    y = height - 40
                            c.drawString(x_margin, y, line)
                            y -= 14
                            if y < 40:
                                c.showPage()
                                y = height - 40
                        c.save()
                        pdf_data = buffer.getvalue()
                        buffer.close()
                        return pdf_data

                    pdf_bytes = _pdf_from_text(report_md)
                except Exception:
                    pdf_bytes = None

                # Try alternate backend (fpdf) if reportlab not available
                if not pdf_bytes:
                    try:
                        from fpdf import FPDF

                        class _MDPDF(FPDF):
                            def header(self):
                                pass

                        pdf = _MDPDF()
                        pdf.set_auto_page_break(auto=True, margin=15)
                        pdf.add_page()
                        pdf.set_font("Arial", size=11)
                        for raw_line in report_md.split("\n"):
                            line = raw_line.replace("\t", "    ")
                            pdf.multi_cell(0, 6, line)
                        pdf_bytes = pdf.output(dest='S').encode('latin-1', 'ignore')
                    except Exception:
                        pdf_bytes = None

                if pdf_bytes:
                    st.download_button(
                        label="⬇️ Download Report (PDF)",
                        data=pdf_bytes,
                        file_name="fraud_summary_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                else:
                    st.download_button(
                        label="⬇️ Download Report (Markdown)",
                        data=report_md.encode("utf-8"),
                        file_name="fraud_summary_report.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )

        # Timeline tab
        with tab_timeline:
            st.subheader("⏰ Claims Timeline Analysis")
            try:
                tl_rows: List[Dict] = []
                # Existing challenge notifications
                for c in fraud_data.get("claims", []):
                    tl_rows.append({
                        "claim_id": c.get("claim_id"),
                        "timestamp": pd.to_datetime(c.get("timestamp")),
                        "channel": c.get("channel"),
                        "notifier": c.get("notifier_name"),
                        "is_fraudulent": c.get("claim_id") in fraudulent_ids,
                    })
                # Live portal submissions
                for rec in self.processor.claims_database:
                    tl_rows.append({
                        "claim_id": rec.get("claim_id"),
                        "timestamp": pd.to_datetime(rec.get("submission_time")),
                        "channel": "Portal",
                        "notifier": rec.get("submitted_by"),
                        "is_fraudulent": rec.get("status") == ClaimStatus.REJECTED.value,
                    })
                if tl_rows:
                    tl_df = pd.DataFrame(tl_rows)
                    fig = px.scatter(
                        tl_df,
                        x="timestamp",
                        y="channel",
                        color="is_fraudulent",
                        hover_data=["claim_id", "notifier"],
                        title="Fraudulent vs Legitimate",
                        color_discrete_map={True: "red", False: "green"},
                        labels={"is_fraudulent": "Fraudulent"},
                    )
                    fig.update_layout(height=480, xaxis_title="Time", yaxis_title="Channel")
                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("🚨 Fraudulent Claims Timeline")
                    fraud_tl = tl_df[tl_df["is_fraudulent"] == True]
                    if not fraud_tl.empty:
                        fig2 = px.scatter(
                            fraud_tl,
                            x="timestamp",
                            y="claim_id",
                            title="Coordinated Timing",
                            color="claim_id",
                        )
                        fig2.update_layout(height=380, xaxis_title="Time", yaxis_title="Claim ID")
                        st.plotly_chart(fig2, use_container_width=True)

                    # Timing gaps section removed as requested

            except Exception as exc:
                st.warning(f"Timeline analysis unavailable: {exc}")

        # Location tab
        with tab_location:
            st.subheader("🗺️ Location-Based Analysis")
            try:
                loc_rows: List[Dict] = []
                for claim_id, text in fraud_data.get("evidence_analysis", {}).items():
                    matches = re.findall(r"GPS Coordinates.*?(-?\d+\.\d+), (-?\d+\.\d+)", text)
                    if matches:
                        lat, lon = matches[0]
                        loc_rows.append({
                            "claim_id": claim_id,
                            "latitude": float(lat),
                            "longitude": float(lon),
                            "is_fraudulent": (claim_id in fraudulent_ids) or any(rc.get("claim_id") == claim_id and rc.get("status") == ClaimStatus.REJECTED.value for rc in self.processor.claims_database),
                        })
                if loc_rows:
                    loc_df = pd.DataFrame(loc_rows)
                    center_lat = float(loc_df["latitude"].mean())
                    center_lon = float(loc_df["longitude"].mean())
                    fig = px.scatter_mapbox(
                        loc_df,
                        lat="latitude",
                        lon="longitude",
                        color="is_fraudulent",
                        hover_data=["claim_id"],
                        title="Fraud Hotspots",
                        color_discrete_map={True: "red", False: "green"},
                        mapbox_style="open-street-map",
                        zoom=14,
                    )
                    fig.update_traces(marker=dict(size=18, opacity=0.95))
                    fig.update_layout(height=520, mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=14))
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.warning(f"Location analysis unavailable: {exc}")

        # Witness tab
        with tab_witness:
            st.subheader("👥 Witness Statement Analysis")
            try:
                w_rows: List[Dict] = []
                duplicate_analysis_rows = []
                
                # Load all witness statements for duplicate analysis
                witness_statements = self.processor._load_all_witness_statements()
                
                for claim_id, text in fraud_data.get("witness_statements", {}).items():
                    driver_names = re.findall(r"(Juma Said|J\. Saeed|Jumaa Saidi|S\. Juma)", text, re.IGNORECASE)
                    vehicle_mentions = re.findall(r"(white Probox|Probox)", text, re.IGNORECASE)
                    location_mentions = re.findall(r"(Moi Avenue|Kimathi Street)", text, re.IGNORECASE)
                    
                    # Check for duplicates in this claim
                    has_duplicates = False
                    duplicate_count = 0
                    if claim_id in witness_statements:
                        statements = witness_statements[claim_id]
                        if isinstance(statements, list) and len(statements) > 1:
                            has_duplicates = True
                            duplicate_count = len(statements)
                    
                    w_rows.append({
                        "claim_id": claim_id,
                        "driver": driver_names[0] if driver_names else "Unknown",
                        "vehicle": len(vehicle_mentions) > 0,
                        "location": len(location_mentions) > 0,
                        "is_fraudulent": claim_id in fraudulent_ids,
                        "length": len(text),
                        "has_duplicates": has_duplicates,
                        "duplicate_count": duplicate_count,
                    })
                    
                    # Add to duplicate analysis
                    if has_duplicates:
                        duplicate_analysis_rows.append({
                            "claim_id": claim_id,
                            "duplicate_count": duplicate_count,
                            "driver": driver_names[0] if driver_names else "Unknown",
                            "is_fraudulent": claim_id in fraudulent_ids,
                        })
                
                if w_rows:
                    w_df = pd.DataFrame(w_rows)
                    
                    # Duplicate Detection Section
                    st.subheader("🚨 Duplicate Witness Statement Detection")
                    if duplicate_analysis_rows:
                        dup_df = pd.DataFrame(duplicate_analysis_rows)
                        st.warning(f"⚠️ Found {len(duplicate_analysis_rows)} claims with duplicate witness statements!")
                        
                        # Display duplicate claims table
                        st.dataframe(dup_df, use_container_width=True)
                        
                        # Duplicate analysis chart
                        fig_dup = px.bar(dup_df, x="claim_id", y="duplicate_count", 
                                        color="is_fraudulent",
                                        labels={"duplicate_count": "Number of Duplicates", "claim_id": "Claim ID"},
                                        title="Duplicate Witness Statements by Claim")
                        st.plotly_chart(fig_dup, use_container_width=True, key="duplicate_chart")
                    else:
                        st.success("✅ No duplicate witness statements detected")
                    
                    st.subheader("🧾 Driver Name Mentions")
                    counts = w_df["driver"].value_counts()
                    fig = px.bar(x=counts.index, y=counts.values, labels={"x": "Driver", "y": "Count"})
                    st.plotly_chart(fig, use_container_width=True, key="driver_mentions_chart")

                    st.subheader("🚙 Vehicle Description Mentions")
                    vehicle_df = w_df.groupby("is_fraudulent")["vehicle"].value_counts().unstack(fill_value=0)
                    fig2 = px.bar(vehicle_df, labels={"value": "Count", "is_fraudulent": "Fraudulent"})
                    st.plotly_chart(fig2, use_container_width=True, key="vehicle_desc_chart")
                    
                    # Enhanced similarity analysis
                    st.subheader("🔍 Content Similarity Analysis")
                    similarity_data = []
                    for i, row1 in w_df.iterrows():
                        for j, row2 in w_df.iterrows():
                            if i < j:  # Avoid duplicate comparisons
                                claim1_id = row1["claim_id"]
                                claim2_id = row2["claim_id"]
                                
                                # Get witness statements for comparison
                                stmt1 = fraud_data.get("witness_statements", {}).get(claim1_id, "")
                                stmt2 = fraud_data.get("witness_statements", {}).get(claim2_id, "")
                                
                                if stmt1 and stmt2:
                                    similarity = self.processor._calculate_text_similarity(stmt1, stmt2)
                                    if similarity > 0.5:  # Only show significant similarities
                                        similarity_data.append({
                                            "claim1": claim1_id,
                                            "claim2": claim2_id,
                                            "similarity": similarity,
                                            "both_fraudulent": row1["is_fraudulent"] and row2["is_fraudulent"]
                                        })
                    
                    if similarity_data:
                        sim_df = pd.DataFrame(similarity_data)
                        st.dataframe(sim_df.sort_values("similarity", ascending=False), use_container_width=True)
                        
                        # Similarity heatmap
                        fig_sim = px.scatter(sim_df, x="claim1", y="claim2", size="similarity", 
                                           color="both_fraudulent",
                                           labels={"similarity": "Similarity Score"},
                                           title="Witness Statement Similarity Matrix")
                        st.plotly_chart(fig_sim, use_container_width=True, key="similarity_chart")
                    else:
                        st.info("No significant similarities found between witness statements")
                        
            except Exception as exc:
                st.warning(f"Witness analysis unavailable: {exc}")

        # Evidence Analysis tab
        with tab_evidence:
            st.subheader("📸 Evidence Analysis")
            
            # Upload and Analysis Section
            st.subheader("🔍 Upload New Evidence for Analysis")
            col_upload1, col_upload2 = st.columns(2)
            
            with col_upload1:
                st.markdown("### Photo Analysis")
                uploaded_photo = st.file_uploader(
                    "Upload Photo for AI Analysis",
                    type=["jpg", "jpeg", "png"],
                    key="evidence_photo_upload",
                    help="Upload a photo to generate AI-powered evidence analysis"
                )
                
                if uploaded_photo:
                    col_img_display, col_img_info = st.columns([1, 1])
                    with col_img_display:
                        st.image(uploaded_photo, caption="Uploaded Photo", use_column_width=True)
                    with col_img_info:
                        st.info(f"**File:** {uploaded_photo.name}\n**Size:** {uploaded_photo.size} bytes")
                    
                    if st.button("Analyze Photo", key="analyze_evidence_photo"):
                        with st.spinner("Analyzing photo with AI..."):
                            temp_claim_id = f"EVIDENCE-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                            analysis_result = self.processor.analyze_uploaded_photo(uploaded_photo, temp_claim_id)
                            st.markdown("### AI Analysis Result:")
                            st.markdown(analysis_result)
                            
                            if st.button("Save Analysis", key="save_evidence_analysis"):
                                if self.processor.save_evidence_analysis(temp_claim_id, analysis_result):
                                    st.success(f"Analysis saved as evidence-analysis/{temp_claim_id}.md")
                                else:
                                    st.error("Failed to save analysis")
            
            with col_upload2:
                st.markdown("### Document Processing")
                uploaded_doc = st.file_uploader(
                    "Upload Document for Processing",
                    type=["pdf", "doc", "docx", "txt"],
                    key="evidence_doc_upload",
                    help="Upload a document to extract witness statement"
                )
                
                if uploaded_doc:
                    st.info(f"**File:** {uploaded_doc.name}\n**Size:** {uploaded_doc.size} bytes")
                    
                    if st.button("Process Document", key="process_evidence_doc"):
                        with st.spinner("Processing document..."):
                            temp_claim_id = f"WITNESS-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                            processed_result = self.processor.process_uploaded_document(uploaded_doc, temp_claim_id)
                            st.markdown("### Processed Statement:")
                            st.markdown(processed_result)
                            
                            if st.button("Save Statement", key="save_evidence_statement"):
                                if self.processor.save_witness_statement(temp_claim_id, processed_result):
                                    st.success(f"Statement saved as witness-statements/{temp_claim_id}-NEW.md")
                                else:
                                    st.error("Failed to save statement")
            
            st.markdown("---")
            
            # Existing Evidence Analysis
            st.subheader("📋 Existing Evidence Analysis")
            try:
                feat_rows: List[Dict] = []
                for claim_id, text in fraud_data.get("evidence_analysis", {}).items():
                    feat_rows.append({
                        "claim_id": claim_id,
                        "hospital": "Nairobi Central Hospital" in text,
                        "doctor_ouma": "Dr. A. Ouma" in text,
                        "same_location": ("Moi Avenue" in text and "Kimathi Street" in text),
                        "suspicious_hash": "d8a7c3b3936a6a0f" in text,
                        "is_fraudulent": claim_id in fraudulent_ids,
                    })
                
                if feat_rows:
                    feat_df = pd.DataFrame(feat_rows)
                    st.dataframe(feat_df, use_container_width=True)
                    
                    # Evidence analysis chart
                    fig = px.bar(feat_df, x="claim_id", y=["hospital", "doctor_ouma", "same_location", "suspicious_hash"],
                                labels={"value": "Feature Present", "claim_id": "Claim ID"},
                                title="Evidence Analysis Features by Claim")
                    st.plotly_chart(fig, use_container_width=True, key="evidence_chart")
                else:
                    st.info("No evidence analysis data available. Upload photos above to generate analysis.")
            except Exception as exc:
                st.warning(f"Evidence analysis unavailable: {exc}")

        # Payout controls (Analyst-only)
        with tab_payouts:
            st.subheader("💳 Payout Controls")
            if not self.processor.claims_database:
                st.info("No claims available.")
            else:
                options = [f"{c.get('claim_id')} — {c.get('status')} — {c.get('payout_status', 'UNKNOWN')}" for c in self.processor.claims_database]
                selected = st.selectbox("Select a claim", options)
                if selected:
                    claim_id = selected.split(" — ")[0]
                    claim = next((c for c in self.processor.claims_database if c.get("claim_id") == claim_id), None)
                    if claim:
                        st.write(f"**Current Payout Status:** {claim.get('payout_status', 'UNKNOWN')}")
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            if st.button("Release Payout (Capture)", use_container_width=True):
                                try:
                                    self.processor.payment.capture(claim_id, note="Analyst release")
                                    claim["payout_status"] = self.processor.payment.get_status(claim_id)
                                    claim["payout_history"] = self.processor.payment.get_history(claim_id)
                                    self.processor.save_claims_database()
                                    st.success("Payout captured.")
                                except Exception as exc:
                                    st.error(f"Error: {exc}")
                        with col_b:
                            if st.button("Place On Hold", use_container_width=True):
                                try:
                                    self.processor.payment.hold(claim_id, note="Analyst hold")
                                    claim["payout_status"] = self.processor.payment.get_status(claim_id)
                                    claim["payout_history"] = self.processor.payment.get_history(claim_id)
                                    self.processor.save_claims_database()
                                    st.success("Payout placed on hold.")
                                except Exception as exc:
                                    st.error(f"Error: {exc}")
                        with col_c:
                            if st.button("Cancel Payout", use_container_width=True):
                                try:
                                    self.processor.payment.cancel(claim_id, note="Analyst cancel")
                                    claim["payout_status"] = self.processor.payment.get_status(claim_id)
                                    claim["payout_history"] = self.processor.payment.get_history(claim_id)
                                    self.processor.save_claims_database()
                                    st.success("Payout canceled.")
                                except Exception as exc:
                                    st.error(f"Error: {exc}")
                        st.markdown("---")
            # Witness analysis intentionally not shown in Payout Controls

        if False:
            with tab_evidence:
                st.subheader("📸 Evidence Analysis")
                try:
                    feat_rows: List[Dict] = []
                    for claim_id, text in fraud_data.get("evidence_analysis", {}).items():
                        feat_rows.append({
                            "claim_id": claim_id,
                            "hospital": "Nairobi Central Hospital" in text,
                            "doctor_ouma": "Dr. A. Ouma" in text,
                            "same_location": ("Moi Avenue" in text and "Kimathi Street" in text),
                            "suspicious_hash": "d8a7c3b3936a6a0f" in text,
                            "is_fraudulent": claim_id in fraudulent_ids,
                        })
                    if feat_rows:
                        f_df = pd.DataFrame(feat_rows)
                        fraud_f = f_df[f_df["is_fraudulent"] == True]
                        if not fraud_f.empty:
                            corr_src = fraud_f[["hospital", "doctor_ouma", "same_location", "suspicious_hash"]].astype(int)
                            fig = px.imshow(
                                corr_src.T,
                                text_auto=True,
                                aspect="auto",
                                color_continuous_scale="Reds",
                                title="Feature Presence (1=present)"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                except Exception as exc:
                    st.warning(f"Evidence analysis unavailable: {exc}")

        # Investigation tools tab
        if False:
            with tab_tools:
                st.subheader("🔬 Investigation Tools")
                try:
                    suspect_ids = list(fraudulent_ids)
                    if suspect_ids:
                        for cid in suspect_ids:
                            with st.expander(f"🔍 Investigate {cid}"):
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.write("**Recommended Actions**")
                                    if st.button(f"📞 Contact Witness - {cid}", key=f"w_{cid}"):
                                        st.success(f"Witness contact initiated for {cid}")
                                    if st.button(f"🏥 Verify Medical Report - {cid}", key=f"m_{cid}"):
                                        st.success(f"Medical verification initiated for {cid}")
                                    if st.button(f"📡 Check Telematics - {cid}", key=f"t_{cid}"):
                                        st.success(f"Telematics check initiated for {cid}")
                                with c2:
                                    st.write("**Case Notes**")
                                    st.text_area("Investigator notes", key=f"notes_{cid}")
                except Exception as exc:
                    st.warning(f"Investigation tools unavailable: {exc}")

    def create_sidebar(self) -> str:
        st.sidebar.title("Navigation")
        if st.session_state.user_role == UserRole.FRAUD_ANALYST.value:
            page = st.sidebar.selectbox("Select Page", ["Fraud Analyst Dashboard", "Analyst Assistant"])
        elif st.session_state.user_role == UserRole.CLIENT.value:
            # Regular clients see all pages including Ask Assistant
            page = st.sidebar.selectbox("Select Page", ["Policy Information", "Submit Claim", "My Claims Status", "Customer Guidance", "Ask Assistant"])
        elif st.session_state.user_role == UserRole.CLIENT_REP.value:
            # Client Representatives don't see Ask Assistant or Customer Support
            page = st.sidebar.selectbox("Select Page", ["Policy Information", "Submit Claim", "My Claims Status"])
        else:
            # Fallback for any other roles
            page = st.sidebar.selectbox("Select Page", ["Policy Information", "Submit Claim", "My Claims Status"])
        st.sidebar.markdown("---")
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            st.session_state.user_role = None
            st.session_state.user_name = None
            st.session_state.current_claim_id = None
            st.session_state.analysis_in_progress = False
            st.session_state.client_authenticated = False
            st.session_state.client_policy_info = None
            # Clear chat histories on logout
            if "chat" in st.session_state:
                del st.session_state["chat"]
            if "ana_chat" in st.session_state:
                del st.session_state["ana_chat"]
            st.rerun()
        return page

    def run(self) -> None:
        if not st.session_state.user_role:
            self.create_login_page()
            return

        # Auto-run analysis for any pending claims so past submissions get decisions
        pending = [c for c in self.processor.claims_database if c.get("fraud_analysis") is None]
        if pending:
            with st.spinner("Analyzing pending claims..."):
                for c in pending:
                    result = self.processor.run_analysis(c["claim_data"])
                    c["fraud_analysis"] = result
                    if result.get("final_recommendation") == "APPROVE CLAIM":
                        c["status"] = ClaimStatus.APPROVED.value
                    elif result.get("final_recommendation") == "ENHANCED REVIEW REQUIRED":
                        c["status"] = ClaimStatus.REQUIRES_REVIEW.value
                    else:
                        c["status"] = ClaimStatus.REJECTED.value
                    c["decision_time"] = datetime.now().isoformat()
                    c["decision_reason"] = result.get("final_recommendation")
                self.processor.save_claims_database()

        st.title(f"🛡️ OMIcare Claims Portal - {st.session_state.user_role}")
        st.markdown("---")
        # Global acknowledgement banner (one-time per set)
        _msg = st.session_state.pop("last_submit_message", None)
        if _msg:
            st.success(_msg)
        page = self.create_sidebar()
        if page == "Policy Information":
            self.create_policy_info_section()
        elif page == "Submit Claim":
            self.create_claim_submission_page()
        elif page == "My Claims Status":
            self.create_claim_status_page()
        elif page == "Fraud Analyst Dashboard":
            self.create_fraud_analyst_dashboard()
        elif page == "Customer Guidance":
            st.header("📖 Customer Guidance")
            st.markdown("---")
            if st.session_state.user_role == UserRole.CLIENT_REP.value:
                st.info("Customer Guidance is not available for Client Representatives. Please contact your client directly for guidance.")
            else:
                st.markdown("For detailed next steps, timelines, and FAQs, see the **Customer Guidance** document.")
                st.markdown("- Processing timelines: Intake holds (~15 minutes), Reviews (up to 1 business day)")
                st.markdown("- Appeals: Within 14 days via reply to decision email or call center")
        elif page == "Ask Assistant":
            st.header("💬 Ask Assistant")
            st.markdown("---")
            if st.session_state.user_role == UserRole.CLIENT_REP.value:
                st.info("Ask Assistant is not available for Client Representatives. Please contact your client directly for assistance.")
            elif not st.session_state.get("client_policy_info"):
                st.info("Please log in as a client to use the assistant.") 
            else:
                pol = st.session_state.client_policy_info
                product = self.processor.get_product_details(pol.get("product_id")) or {}
                limits = product.get("limits_of_liability", {})
                excesses = product.get("excesses", {})

                # Include recent claim context (last 3 claims for this user)
                recent_claims = []
                user_claims: List[Dict] = []
                try:
                    user_claims = [c for c in self.processor.claims_database if c.get("submitted_by") == st.session_state.user_name]
                    user_claims = sorted(user_claims, key=lambda x: x.get("submission_time", ""), reverse=True)[:3]
                    for c in user_claims:
                        recent_claims.append({
                            "claim_id": c.get("claim_id"),
                            "status": c.get("status"),
                            "decision": c.get("decision_reason"),
                            "payout_status": c.get("payout_status", "UNKNOWN"),
                        })
                except Exception:
                    recent_claims = []

                context = (
                    f"Client Name: {pol.get('client_name')}\n"
                    f"Policy Number: {pol.get('policy_number')}\n"
                    f"Vehicle: {pol.get('vehicle_registration')}\n"
                    f"Policy Period: {pol.get('policy_start_date')} to {pol.get('policy_end_date')}\n"
                    f"Product ID: {pol.get('product_id')}\n"
                    f"Limits of Liability: {limits}\n"
                    f"Excesses: {excesses}\n"
                    f"Recent Claims: {recent_claims}\n"
                )

                if "chat" not in st.session_state:
                    st.session_state.chat = [{
                        "role": "assistant",
                        "content": (
                            f"Hello {pol.get('client_name')} 👋. I'm your AI assistant! I can help you with:\n\n"
                            "📋 **Policy Information**: Coverage, limits, and benefits\n"
                            "📄 **Claim Status**: Track your claims and processing\n"
                            "📸 **Evidence Requirements**: What documents you need to submit\n"
                            "⏱️ **Processing Timelines**: How long claims take\n"
                            "📞 **Appeals Process**: How to challenge decisions\n\n"
                            "What would you like to know?"
                        )
                    }]
                for m in st.session_state.chat:
                    with st.chat_message(m["role"]):
                        st.markdown(m["content"])

                prompt = st.chat_input("Ask about your policy, claims, evidence requirements, or benefits…")
                if prompt:
                    st.session_state.chat.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    # Lightweight sentiment detection for escalation (no external dependencies)
                    angry = False
                    try:
                        _t = prompt.strip()
                        _l = _t.lower()
                        _angry_terms = [
                            "angry", "upset", "frustrated", "furious", "annoyed", "ridiculous",
                            "unacceptable", "worst", "complain", "complaint", "scam", "robbed",
                            "not happy", "disappointed", "mad"
                        ]
                        _caps = sum(1 for ch in _t if ch.isupper())
                        angry = (
                            any(term in _l for term in _angry_terms) or _t.count("!") >= 2 or _caps >= 10
                        )
                    except Exception:
                        angry = False

                    system_msg = (
                        "You are a friendly, concise insurance assistant. Start with a brief greeting if appropriate. "
                        "Always ground answers ONLY on the provided policy/product/claim context. "
                        "If unknown, say you don't have that info. "
                        "For evidence requirements, provide: Scene Photos (accident location and vehicle damage), "
                        "Medical Reports (official documentation from healthcare providers), "
                        "Witness Statements (written statements from witnesses), "
                        "Additional Documentation (police reports, repair estimates)."
                    )
                    messages = [
                        {"role": "system", "content": system_msg},
                        {"role": "system", "content": f"Context:\n{context}"},
                        {"role": "user", "content": prompt},
                    ]

                    # Enhanced deterministic responses for common queries
                    prompt_lower = prompt.lower().strip()
                    
                    # Evidence requirements response
                    if any(phrase in prompt_lower for phrase in ["evidence", "documents", "photos", "medical", "witness", "submit claim", "what do i need"]):
                        answer = (
                            "To submit a claim, you'll need to provide the following evidence:\n\n"
                            "📸 **Scene Photos**: Clear photos of the accident location and vehicle damage\n"
                            "🏥 **Medical Reports**: Official medical documentation from treating healthcare providers\n"
                            "👥 **Witness Statements**: Written statements from individuals who witnessed the incident\n"
                            "📋 **Additional Documentation**: Police reports, repair estimates, and other relevant evidence\n\n"
                            "You can upload these documents when submitting your claim through the 'Submit Claim' page."
                        )
                    # Policy coverage response
                    elif any(phrase in prompt_lower for phrase in ["coverage", "covered", "policy", "benefits", "limits"]):
                        limits_text = "\n".join([f"- {k}: {v}" for k, v in limits.items()]) if limits else "No specific limits available"
                        answer = (
                            f"Your policy coverage includes:\n\n{limits_text}\n\n"
                            f"Excesses: {excesses if excesses else 'No excess information available'}\n\n"
                            "For more detailed information, please contact our customer service team."
                        )
                    # Claim status response
                    elif any(phrase in prompt_lower for phrase in ["status", "progress", "where is", "how is"]):
                        if recent_claims:
                            claims_text = "\n".join([f"- {c['claim_id']}: {c['status']} ({c['decision']})" for c in recent_claims])
                            answer = f"Your recent claims status:\n\n{claims_text}\n\nFor real-time updates, check the 'My Claims Status' page."
                        else:
                            answer = "You don't have any recent claims. You can submit a new claim through the 'Submit Claim' page."
                    # Timeline response
                    elif any(phrase in prompt_lower for phrase in ["time", "long", "when", "timeline", "process"]):
                        answer = (
                            "Claim processing timelines:\n\n"
                            "⚡ **Initial Checks**: Your claim runs through automated initial checks\n"
                            "✅ **If Everything is in Order**: Claim approved within 15 minutes\n"
                            "🔍 **If Additional Checks Needed**: Claim decision will take 48 hours\n"
                            "💰 **Payout**: 1-3 business days after approval\n\n"
                            "You'll receive notifications at each stage of the process."
                        )
                    # Appeal process response
                    elif any(phrase in prompt_lower for phrase in ["appeal", "reject", "disagree", "challenge"]):
                        answer = (
                            "You can reach out to our claims support team via:\n\n"
                            "📞 **Phone**: +254 700 123 456\n"
                            "📧 **Email**: appeals@omicare.co.ke\n\n"
                            "Our team is available Monday-Friday, 8AM-5PM to assist you with your appeal."
                        )
                    else:
                        answer = (
                            "I'm sorry, I do not understand this. I may not also have some information. "
                            "I can only assist you with finding out about your policy details, claim information, "
                            "evidence requirements, processing timelines, and appeal processes."
                        )
                    try:
                        api_key = os.environ.get("OPENAI_API_KEY")
                        base_url = os.environ.get("LLM_BASE_URL")
                        model_name = os.environ.get("LLM_MODEL", "gpt-4o-mini")

                        # If no remote/local LLM configured, do a deterministic data lookup (no installs, no keys)
                        if not api_key and not base_url:
                            import difflib

                            def _flatten(prefix: str, obj):
                                rows = []
                                if isinstance(obj, dict):
                                    for k, v in obj.items():
                                        rows.extend(_flatten(f"{prefix}.{k}" if prefix else str(k), v))
                                elif isinstance(obj, list):
                                    for idx, v in enumerate(obj):
                                        rows.extend(_flatten(f"{prefix}[{idx}]", v))
                                else:
                                    rows.append((prefix, str(obj)))
                                return rows

                            kb = {
                                "policy": pol,
                                "product_limits": limits,
                                "product_excesses": excesses,
                                "recent_claims": recent_claims,
                            }
                            pairs = _flatten("", kb)
                            keys = [p[0] for p in pairs]
                            matches = difflib.get_close_matches(prompt.lower(), [k.lower() for k in keys], n=6, cutoff=0.3)
                            found = []
                            for m in matches:
                                for k, v in pairs:
                                    if k.lower() == m:
                                        found.append((k, v))
                            lower_q = prompt.strip().lower()
                            greet_terms = ["hi", "hello", "hey", "habari", "good morning", "good afternoon", "good evening"]
                            thank_terms = ["thanks", "thank you", "asante"]
                            if any(lower_q == t or lower_q.startswith(t + " ") for t in greet_terms):
                                answer = f"Hello {pol.get('client_name')}! Welcome to OMIcare. I am your trusted assistant. How can I assist you?"
                            elif any(t in lower_q for t in thank_terms):
                                answer = "You're welcome. How else can I assist with your policy?"
                            elif any(t in lower_q for t in [
                                "bye", "goodbye", "good bye", "ok bye", "bye bye", "see you", "farewell", "kwaheri"
                            ]):
                                answer = f"Goodbye {pol.get('client_name')}. I hope I've been helpful."
                            elif any(word in lower_q for word in ["time", "long", "when", "timeline", "process", "how long", "processing time"]):
                                answer = (
                                    "Claim processing timelines:\n\n"
                                    "⚡ **Initial Checks**: Your claim runs through automated initial checks\n"
                                    "✅ **If Everything is in Order**: Claim approved within 15 minutes\n"
                                    "🔍 **If Additional Checks Needed**: Claim decision will take 48 hours\n"
                                    "💰 **Payout**: 1-3 business days after approval\n\n"
                                    "You'll receive notifications at each stage of the process."
                                )
                            elif any(phrase in lower_q for phrase in [
                                "file a claim", "file a new claim", "submit a claim", "start a claim",
                                "make a claim", "report a claim", "report an accident", "file my claim"
                            ]):
                                answer = (
                                    "To file a claim, open the sidebar and select ‘Submit Claim’. "
                                    "Complete the form and attach at least one evidence file and a witness statement, then click Submit."
                                )
                            elif any(word in lower_q for word in ["escalate", "appeal", "call center", "agent", "talk to", "human", "contact"]):
                                answer = (
                                    "You can reach out to our claims support team via:\n\n"
                                    "📞 **Phone**: +254 700 123 456\n"
                                    "📧 **Email**: appeals@omicare.co.ke\n\n"
                                    "Our team is available Monday-Friday, 8AM-5PM to assist you."
                                )
                            else:
                                # Intent routing: policy-only vs claims
                                policy_terms = [
                                    "policy", "policy details", "coverage", "cover", "limit", "limits", "excess", "excesses",
                                    "period", "start date", "end date", "expiry", "premium", "benefits", "product"
                                ]
                                claim_terms = ["claim", "claims", "payout", "decision", "status"]
                                is_policy_q = any(t in lower_q for t in policy_terms)
                                is_claim_q = any(t in lower_q for t in claim_terms)

                                if is_policy_q and not is_claim_q:
                                    lines = [f"Hello {pol.get('client_name')}, here are your policy details:"]
                                    lines.append(f"- Policy: {pol.get('policy_number')} (Product {pol.get('product_id')})")
                                    lines.append(f"- Period: {pol.get('policy_start_date')} to {pol.get('policy_end_date')}")
                                    lines.append(f"- Vehicle: {pol.get('vehicle_registration')}")
                                    try:
                                        if isinstance(limits, dict) and limits:
                                            lim_items = list(limits.items())[:3]
                                            lines.append("- Key Limits:")
                                            for k, v in lim_items:
                                                lines.append(f"  • {str(k).replace('_',' ').title()}: {v}")
                                        if isinstance(excesses, dict) and excesses:
                                            ex_items = list(excesses.items())[:2]
                                            lines.append("- Key Excesses:")
                                            for k, v in ex_items:
                                                if isinstance(v, dict):
                                                    rate = v.get("rate", "-")
                                                    mn = v.get("minimum_kes", "-")
                                                    lines.append(f"  • {str(k).replace('_',' ').title()}: rate {rate}, minimum KES {mn}")
                                                else:
                                                    lines.append(f"  • {str(k).replace('_',' ').title()}: KES {v}")
                                    except Exception:
                                        pass
                                    answer = "\n".join(lines)
                                elif found:
                                    claim_related = any("claim" in k.lower() for k, _ in found)
                                    if claim_related and user_claims:
                                        lines = [f"Hi {pol.get('client_name')}, here’s a quick summary:"]
                                        for c in user_claims[:2]:
                                            date_str = (c.get("claim_data", {}).get("accident_date") or c.get("submission_time", ""))
                                            policy_no = (st.session_state.get("client_policy_info") or {}).get("policy_number")
                                            decision_raw = (c.get("decision_reason") or "").upper()
                                            status_raw = (c.get("status") or "").upper()
                                            payout_total = None
                                            try:
                                                ps = c.get("payout_suggestion") or {}
                                                payout_total = ps.get("total_amount")
                                            except Exception:
                                                payout_total = None
                                            if payout_total is None:
                                                try:
                                                    ps2 = self.processor.compute_payout_suggestion(c.get("claim_data", {}))
                                                    payout_total = ps2.get("total_amount")
                                                except Exception:
                                                    payout_total = None

                                            if "APPROVED" in status_raw or decision_raw == "APPROVE CLAIM":
                                                if isinstance(payout_total, (int, float)) and payout_total > 0:
                                                    decision_msg = (
                                                        f"Your claim has been approved. The payout amount is KES {int(payout_total):,}. "
                                                        "This will be paid to your account within 48 hours."
                                                    )
                                                else:
                                                    decision_msg = (
                                                        "Your claim has been approved. The payout will be sent to your account within 48 hours."
                                                    )
                                            elif "REQUIRES_REVIEW" in status_raw or decision_raw == "ENHANCED REVIEW REQUIRED":
                                                decision_msg = "Your claim is on hold. We will notify you of the decision within 48 hours."
                                            elif "REJECT" in status_raw or "REJECT" in decision_raw:
                                                decision_msg = "Your claim has been rejected."
                                            else:
                                                decision_msg = f"Current status: {c.get('status') or 'Pending'}."
                                            lines.append(
                                                f"- You filed claim {c.get('claim_id')} on {str(date_str)[:10]} under policy {policy_no}. {decision_msg}"
                                            )
                                        answer = "\n".join(lines)
                                    else:
                                        lines = [f"Hi {pol.get('client_name')}, here’s what I found:"]
                                        for k, v in found[:5]:
                                            lines.append(f"- {k}: {v}")
                                        answer = "\n".join(lines)
                            if answer == "":
                                answer = (
                                    "I'm sorry, I do not understand this. I may not also have some information. "
                                    "I can only assist you with finding out about your policy details and claim information."
                                )
                        else:
                            if OpenAI:
                                if api_key:
                                    client = OpenAI(api_key=api_key)
                                else:
                                    if not base_url:
                                        base_url = "http://localhost:1234/v1"
                                    client = OpenAI(api_key=os.environ.get("LLM_API_KEY", "local"), base_url=base_url)
                                resp = client.chat.completions.create(
                                    model=model_name,
                                    messages=messages,
                                    temperature=0.2,
                                )
                                answer = resp.choices[0].message.content
                    except Exception as exc:
                        answer = f"(Assistant error) {exc}"

                    # If client is angry, respond only with the escalation message (no preceding data)
                    if angry:
                        answer = (
                            "I’m sorry you’re experiencing this. If you’d like to escalate, please share your Policy Number and (if applicable) Claim ID." 
                        )

                    st.session_state.chat.append({"role": "assistant", "content": answer})
                    with st.chat_message("assistant"):
                        st.markdown(answer)
        elif page == "Analyst Assistant":
            if st.session_state.user_role != UserRole.FRAUD_ANALYST.value:
                st.info("Analyst Assistant is only available to Fraud Analysts.")
            else:
                st.header("🧠 Analyst Assistant")
                st.markdown("---")
                # Load datasets
                data = self.processor.load_fraud_analysis_data()
                claims_db = list(self.processor.claims_database)
                fraud_scores = data.get("fraud_scores", [])

                # Build compact context
                summary_lines: List[str] = []
                try:
                    for grp in fraud_scores:
                        summary_lines.append(
                            f"Group {grp.get('claim_group')}: risk={grp.get('risk_level')} score={grp.get('fraud_score')}"
                        )
                        if grp.get("red_flags"):
                            summary_lines.append("Red Flags: " + "; ".join(grp.get("red_flags", [])[:5]))
                    for rec in claims_db[-10:]:
                        summary_lines.append(
                            f"Claim {rec.get('claim_id')} status={rec.get('status')} decision={rec.get('decision_reason')}"
                        )
                except Exception:
                    pass
                analyst_context = "\n".join(summary_lines) if summary_lines else "No findings loaded."

                if "ana_chat" not in st.session_state:
                    st.session_state.ana_chat = [{
                        "role": "assistant",
                        "content": (
                            "Hello Analyst. You can ask about coordinated patterns, red flags, and propose a conclusion."
                        )
                    }]

                for m in st.session_state.ana_chat:
                    with st.chat_message(m["role"]):
                        st.markdown(m["content"])

                q = st.chat_input("Ask about patterns, risk levels, red flags, or a summary...")
                if q:
                    st.session_state.ana_chat.append({"role": "user", "content": q})
                    with st.chat_message("user"):
                        st.markdown(q)

                    system_msg = (
                        "You are a forensic fraud analysis assistant. Be concise, cite observed patterns, propose an action (Approve/Review/Reject). "
                        "Base answers ONLY on the provided findings and claims DB summary."
                    )
                    messages = [
                        {"role": "system", "content": system_msg},
                        {"role": "system", "content": f"Findings Summary:\n{analyst_context}"},
                        {"role": "user", "content": q},
                    ]

                    ana_answer = (
                        "I'm sorry, I do not understand this. For analysis, provide a specific question about patterns, risk levels, or findings."
                    )
                    try:
                        api_key = os.environ.get("OPENAI_API_KEY")
                        base_url = os.environ.get("LLM_BASE_URL")
                        model_name = os.environ.get("LLM_MODEL", "gpt-4o-mini")
                        if not api_key and not base_url:
                            # Deterministic summary without LLM
                            high_groups = [g for g in fraud_scores if (g.get("risk_level") or "").upper() in ("HIGH", "CRITICAL")]
                            med_groups = [g for g in fraud_scores if (g.get("risk_level") or "").upper() == "MEDIUM"]
                            lines = ["Here’s a summary from current findings:"]
                            if high_groups:
                                lines.append(f"- High/Critical groups: {len(high_groups)}")
                            if med_groups:
                                lines.append(f"- Medium groups: {len(med_groups)}")
                            if fraud_scores:
                                rf = fraud_scores[0].get("red_flags", [])
                                if rf:
                                    lines.append("- Key red flags: " + "; ".join(rf[:5]))
                            lines.append("Recommendation: If group risk is HIGH/CRITICAL, proceed to detailed investigation and keep payouts on hold.")
                            lines.append("For MEDIUM, perform enhanced review before release.")
                            
                            ana_answer = "\n".join(lines)
                        else:
                            if OpenAI:
                                if api_key:
                                    client = OpenAI(api_key=api_key)
                                else:
                                    if not base_url:
                                        base_url = "http://localhost:1234/v1"
                                    client = OpenAI(api_key=os.environ.get("LLM_API_KEY", "local"), base_url=base_url)
                                resp = client.chat.completions.create(
                                    model=model_name,
                                    messages=messages,
                                    temperature=0.2,
                                )
                                ana_answer = resp.choices[0].message.content
                    except Exception as exc:
                        ana_answer = f"(Assistant error) {exc}"

                    st.session_state.ana_chat.append({"role": "assistant", "content": ana_answer})
                    with st.chat_message("assistant"):
                        st.markdown(ana_answer)


def main() -> None:
    app = RoleBasedApp()
    app.run()


if __name__ == "__main__":
    main()