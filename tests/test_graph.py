"""
Automated Pytest Suite for LangGraph Multi-Agent Workflows (Phase 7).
"""

import pytest
from src.core.state import ClaimState
from src.core.graph import build_adjudication_graph, run_claim_adjudication


def test_stategraph_compilation():
    """Verify LangGraph StateGraph builds and compiles without syntax/edge errors."""
    graph = build_adjudication_graph()
    assert graph is not None


def test_end_to_end_adjudication_workflow_mock():
    """Verify end-to-end claim execution across parallel agent nodes to final decision."""
    initial_state: ClaimState = {
        "claim_id": "TEST-CLM-500",
        "policy_number": "POL-500",
        "vehicle_details": {"make": "Maruti", "model": "Swift", "age_years": 2.0, "cubic_capacity": 1197.0},
        "incident_narrative": "Vehicle bumper collided with parking pillar.",
        "uploaded_images": [],
        "estimate_line_items": [
            {"part_name": "Front Bumper", "category": "plastic", "claimed_cost": 4000.0},
        ],
        "customer_history": [],
        "held_policy_endorsements": ["ZERO_DEP"],
    }

    final_state = run_claim_adjudication(initial_state)

    assert "adjudication_verdict" in final_state
    assert final_state["adjudication_verdict"] in ["APPROVE", "APPROVE_ADJUSTED", "ESCALATE", "REJECT"]
    assert "approved_amount" in final_state
    assert final_state["claimed_amount"] == 4000.0
