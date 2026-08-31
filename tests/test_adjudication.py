"""
Integration Tests for Motor Claim Adjudication Edge Cases (Phase 7).
"""

import pytest
from src.core.state import ClaimState
from src.core.graph import run_claim_adjudication


def test_missing_evidence_triggers_escalation():
    """Verify missing mandatory damage photo triggers ESCALATE verdict."""
    state: ClaimState = {
        "claim_id": "TEST-CLM-ESC-01",
        "policy_number": "POL-ESC-01",
        "vehicle_details": {"make": "Hyundai", "model": "i20", "age_years": 1.0, "cubic_capacity": 1197.0},
        "incident_narrative": "Major frontal collision on highway.",
        "uploaded_images": [],
        "estimate_line_items": [{"part_name": "Front Bumper", "claimed_cost": 5000.0}],
        "customer_history": [],
        "held_policy_endorsements": [],
    }

    final_state = run_claim_adjudication(state)
    assert final_state["adjudication_verdict"] == "ESCALATE"
    assert len(final_state["missing_evidence_flags"]) > 0


def test_metal_depreciation_adjusted_approval():
    """Verify claim without Zero-Dep rider applies metal depreciation and approves adjusted amount."""
    state: ClaimState = {
        "claim_id": "TEST-CLM-ADJ-02",
        "policy_number": "POL-ADJ-02",
        "vehicle_details": {"make": "Honda", "model": "City", "age_years": 3.0, "cubic_capacity": 1498.0},
        "incident_narrative": "Scratched bonnet hood during storm.",
        "uploaded_images": [],
        "estimate_line_items": [{"part_name": "Bonnet Hood", "category": "metal", "claimed_cost": 10000.0}],
        "customer_history": [],
        "held_policy_endorsements": [],
    }

    final_state = run_claim_adjudication(state)
    assert final_state["claimed_amount"] == 10000.0
    assert final_state["deductions_breakdown"]["depreciation"] == 1500.0
    assert final_state["deductions_breakdown"]["compulsory_deductible"] == 1000.0
