"""
Global Claim State Definition Module.

This module defines the central `ClaimState` TypedDict used by LangGraph to pass
state, evidence analysis, RAG policy matches, anomaly scores, and synthesis decisions
across parallel agent nodes.
"""

import operator
from typing import TypedDict, List, Dict, Any, Optional, Annotated


class ClaimState(TypedDict, total=False):
    """
    TypedDict representing the global state of a motor insurance claim packet throughout
    the multi-agent adjudication workflow.
    """

    # --- Raw Input Claim Packet ---
    claim_id: str
    policy_number: str
    vehicle_details: Dict[str, Any]
    incident_narrative: str
    uploaded_images: List[str]
    estimate_line_items: List[Dict[str, Any]]
    customer_history: List[Dict[str, Any]]
    held_policy_endorsements: List[str]

    # --- Multimodal Evidence Agent Outputs ---
    vision_findings: Dict[str, Any]
    missing_evidence_flags: List[str]

    # --- Policy & Limit Matcher Agent Outputs (RAG) ---
    retrieved_policy_clauses: List[Dict[str, Any]]
    applicable_depreciation_rates: Dict[str, float]
    compulsory_deductible: float
    policy_warning_flags: List[str]

    # --- Cost & History Anomaly Agent Outputs ---
    cost_variance_flags: List[Dict[str, Any]]
    frequency_risk_score: float
    history_anomalies: List[str]

    # --- Synthesis & Adjudication Verdict Outputs ---
    cross_modal_consistency: bool
    consistency_notes: str
    adjudication_verdict: str
    adjudication_rationale: str
    claimed_amount: float
    approved_amount: float
    deductions_breakdown: Dict[str, float]
    mandatory_citations: List[str]
    # --- Execution Logs & Data Provenance ---
    execution_logs: Annotated[List[Dict[str, Any]], operator.add]


