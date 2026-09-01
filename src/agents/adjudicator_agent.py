from typing import Dict, Any, List
from pydantic import BaseModel, Field
from src.strategies.depreciation_strategy import DepreciationStrategy
from src.strategies.deductible_strategy import DeductibleStrategy
from src.utils.llm_utils import invoke_gemini_json
from src.utils.logger import create_log_entry


class SynthesisVerdict(BaseModel):
    """Pydantic schema for Synthesis & Adjudication Agent decision response."""

    adjudication_verdict: str = Field(
        description="Final verdict decision: 'APPROVE', 'APPROVE_ADJUSTED', 'ESCALATE', or 'REJECT'"
    )
    adjudication_rationale: str = Field(
        description="Audit-ready narrative rationale explaining the adjudication verdict and calculation breakdown"
    )
    mandatory_citations: List[str] = Field(
        description="IRDAI and India Motor Tariff policy clause citations supporting the verdict"
    )
    investigation_triggers: List[str] = Field(
        default_factory=list,
        description="Specific triggers requiring human adjuster investigation or document submission",
    )


def adjudicator_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    estimate_items = state.get("estimate_line_items", [])
    vehicle = state.get("vehicle_details", {})
    endorsements = state.get("held_policy_endorsements", [])
    vision_findings = state.get("vision_findings", {})
    cross_modal_consistency = state.get("cross_modal_consistency", True)
    missing_evidence = state.get("missing_evidence_flags", [])
    policy_warnings = state.get("policy_warning_flags", [])
    citations = state.get("mandatory_citations", [])
    cost_flags = state.get("cost_variance_flags", [])
    risk_score = state.get("frequency_risk_score", 0.0)
    logs = list(state.get("execution_logs", []))

    total_claimed = sum(float(item.get("claimed_cost", 0.0)) for item in estimate_items)

    dep_strategy = DepreciationStrategy()
    dep_results = dep_strategy.calculate(
        estimate_items=estimate_items,
        vehicle_details=vehicle,
        held_endorsements=endorsements,
    )

    ded_strategy = DeductibleStrategy()
    ded_results = ded_strategy.calculate(
        estimate_items=dep_results["item_breakdown"],
        vehicle_details=vehicle,
        held_endorsements=endorsements,
    )

    dep_deduction = dep_results["total_depreciation_deduction"]
    compulsory_deductible = ded_results["compulsory_deductible"]
    approved_payout = ded_results["net_payable_amount"]

    has_fraud_escalation = any(
        flag.get("action_flag") == "FRAUD_ESCALATION" for flag in cost_flags
    )
    has_cost_adjustment = any(
        flag.get("action_flag") == "ADJUSTMENT_CLAMP" for flag in cost_flags
    )

    investigation_triggers: List[str] = []

    if missing_evidence or not cross_modal_consistency or has_fraud_escalation or risk_score > 0.70:
        recommended_verdict = "ESCALATE"
        if missing_evidence:
            investigation_triggers.extend(missing_evidence)
        if not cross_modal_consistency:
            investigation_triggers.append("CROSS_MODAL_NARRATIVE_MISMATCH")
        if has_fraud_escalation:
            investigation_triggers.append("EXCESSIVE_BENCHMARK_COST_INFLATION")
        if risk_score > 0.70:
            investigation_triggers.append("HIGH_CLAIM_VELOCITY_RISK")
    elif dep_deduction > 0.0 or compulsory_deductible > 0.0 or has_cost_adjustment or policy_warnings:
        recommended_verdict = "APPROVE_ADJUSTED"
    else:
        recommended_verdict = "APPROVE"

    data_sources = [
        "Multimodal Evidence Agent: Photo damage findings & cross-modal consistency",
        "Policy & Limit Matcher Agent: ChromaDB policy clause citations & coverage warnings",
        "Cost & History Anomaly Agent: Cost variance benchmarks & claim velocity risk score",
        f"Statutory IRDAI Depreciation Rules (Depreciation: ₹{dep_deduction:,.2f})",
        f"India Motor Tariff Deductible Schedule (Compulsory Deductible: ₹{compulsory_deductible:,.2f})",
        "Model Engine: Google Gemini 1.5 Pro (gemini-3.6-flash)"
    ]

    prompt = f"""You are the Chief Adjudication Arbitrator for Motor Insurance Claims.
Synthesize the final claim decision based on findings from Evidence, Policy, and Anomaly agents.

Claim Summary:
- Total Claimed: ₹{total_claimed:,.2f}
- Calculated Approved Net Payout: ₹{approved_payout:,.2f}
- Deductions: Depreciation = ₹{dep_deduction:,.2f}, Compulsory Deductible = ₹{compulsory_deductible:,.2f}
- Cross-Modal Consistency: {cross_modal_consistency}
- Missing Evidence Flags: {missing_evidence}
- Cost Variance Flags: {cost_flags}
- Recommended Verdict: {recommended_verdict}
- Mandatory Policy Citations: {citations}
- Active Policy Riders: {endorsements}

Evaluation Tasks:
1. Provide the final adjudication_verdict ('APPROVE', 'APPROVE_ADJUSTED', 'ESCALATE', or 'REJECT').
2. Write a professional, audit-ready rationale explaining the breakdown of approved vs deducted amounts.
3. List explicit investigation triggers if verdict is ESCALATE.
"""

    try:
        synthesis: SynthesisVerdict = invoke_gemini_json(
            prompt=prompt,
            response_schema=SynthesisVerdict,
            temperature=0.2,
        )

        final_verdict = synthesis.adjudication_verdict
        final_rationale = synthesis.adjudication_rationale
        final_citations = list(set(citations + synthesis.mandatory_citations))
        final_triggers = list(set(investigation_triggers + synthesis.investigation_triggers))

        status_flag = "WARNING" if final_verdict in ["ESCALATE", "REJECT", "APPROVE_ADJUSTED"] else "SUCCESS"
        summary_text = (
            f"Verdict: {final_verdict}. "
            f"Gross Claimed: ₹{total_claimed:,.2f}, Approved Payout: ₹{approved_payout:,.2f}. "
            f"Depreciation Deduction: ₹{dep_deduction:,.2f}, Compulsory Excess: ₹{compulsory_deductible:,.2f}. "
            f"Triggers: {', '.join(final_triggers) or 'None'}"
        )

        success_log = create_log_entry(
            agent="Synthesis & Adjudication Agent",
            step="Multi-Agent Result Convergence & Final Payout Arbitration",
            summary=summary_text,
            data_sources=data_sources,
            status=status_flag
        )
        logs.append(success_log)

    except Exception as exc:
        print(f"[ADJUDICATOR AGENT FALLBACK] Synthesis LLM call failed: {exc}")
        final_verdict = recommended_verdict
        final_rationale = (
            f"Claim processed with decision verdict {recommended_verdict}. "
            f"Claimed: ₹{total_claimed:,.2f}, Approved: ₹{approved_payout:,.2f}. "
            f"Depreciation: ₹{dep_deduction:,.2f}, Deductible: ₹{compulsory_deductible:,.2f}."
        )
        final_citations = citations
        final_triggers = investigation_triggers

        fallback_log = create_log_entry(
            agent="Synthesis & Adjudication Agent",
            step="Multi-Agent Result Convergence & Final Payout Arbitration",
            summary=f"Adjudicator synthesis fallback triggered: {exc}",
            data_sources=data_sources,
            status="FALLBACK"
        )
        logs.append(fallback_log)

    return {
        "adjudication_verdict": final_verdict,
        "adjudication_rationale": final_rationale,
        "claimed_amount": round(total_claimed, 2),
        "approved_amount": round(approved_payout, 2),
        "deductions_breakdown": {
            "depreciation": round(dep_deduction, 2),
            "compulsory_deductible": round(compulsory_deductible, 2),
        },
        "mandatory_citations": final_citations,
        "investigation_triggers": final_triggers,
        "execution_logs": logs,
    }

