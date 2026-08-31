"""
Cost & History Anomaly Agent Node Module.
"""

from typing import Dict, Any, List
from pydantic import BaseModel, Field
from src.config import config
from src.utils.file_utils import read_json_file
from src.utils.llm_utils import invoke_gemini_json


class AnomalyAnalysis(BaseModel):
    """Pydantic schema for Cost & History Anomaly Agent response."""

    frequency_risk_score: float = Field(
        description="Risk score (0.0 to 1.0) assessing claim frequency and recurrence velocity"
    )
    cost_variance_flags: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Line items exceeding benchmark bounds with variance percentages",
    )
    history_anomalies: List[str] = Field(
        default_factory=list,
        description="Detailed anomaly flags regarding repeat claims or suspicious cost spikes",
    )
    fir_verification_required: bool = Field(
        description="True if claim exceeds loss threshold (₹50,000) or involves third-party injury requiring FIR"
    )


def anomaly_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    estimate_items = state.get("estimate_line_items", [])
    history = state.get("customer_history", [])
    vehicle = state.get("vehicle_details", {})
    narrative = state.get("incident_narrative", "").lower()

    segment = str(vehicle.get("segment", "hatchback")).lower()
    if segment not in ["hatchback", "sedan", "suv", "luxury"]:
        segment = "hatchback"

    try:
        matrix_data = read_json_file(config.BENCHMARKS_PATH)
        segment_benchmarks = matrix_data.get("segments", {}).get(segment, {})
    except Exception:
        segment_benchmarks = {}

    total_claimed = sum(float(item.get("claimed_cost", 0.0)) for item in estimate_items)

    cost_flags: List[Dict[str, Any]] = []
    for item in estimate_items:
        part_key = item.get("part_name", "").lower().replace(" ", "_")
        claimed_cost = float(item.get("claimed_cost", 0.0))

        bench = segment_benchmarks.get(part_key)
        if bench:
            median_cost = float(bench.get("benchmark_median", claimed_cost))
            if claimed_cost > median_cost:
                variance_pct = (claimed_cost - median_cost) / median_cost
                if variance_pct >= config.COST_ADJUSTMENT_THRESHOLD:
                    severity = "FRAUD_ESCALATION" if variance_pct >= config.FRAUD_ESCALATION_THRESHOLD else "ADJUSTMENT_CLAMP"
                    cost_flags.append(
                        {
                            "part_name": item.get("part_name", ""),
                            "claimed_cost": claimed_cost,
                            "benchmark_median": median_cost,
                            "variance_pct": round(variance_pct * 100, 1),
                            "action_flag": severity,
                        }
                    )

    is_third_party = "third party" in narrative or "tp injury" in narrative or "pedestrian" in narrative
    requires_fir = total_claimed >= config.FIR_REQUIRED_LOSS_THRESHOLD or is_third_party

    prompt = f"""You are an expert Insurance Fraud Auditor & Data Scientist.
Audit the following claim estimate items and 6-month claim history for cost inflation and recurrence anomalies.

Vehicle Segment: {segment}
Claimed Line Items: {estimate_items}
Prior Claim History (Past 6 Months): {history}
Calculated Cost Variance Flags: {cost_flags}

Evaluation Tasks:
1. Assign a frequency_risk_score between 0.0 (low risk) and 1.0 (high risk) based on 6-month velocity and prior claim count.
2. Formulate history anomaly descriptions.
3. Verify FIR requirement flag.
"""

    try:
        analysis: AnomalyAnalysis = invoke_gemini_json(
            prompt=prompt,
            response_schema=AnomalyAnalysis,
            temperature=0.2,
        )

        return {
            "cost_variance_flags": cost_flags if cost_flags else analysis.cost_variance_flags,
            "frequency_risk_score": analysis.frequency_risk_score,
            "history_anomalies": analysis.history_anomalies,
        }

    except Exception as exc:
        print(f"[ANOMALY AGENT FALLBACK] Anomaly synthesis call failed: {exc}")
        return {
            "cost_variance_flags": cost_flags,
            "frequency_risk_score": 0.3 if len(history) > 1 else 0.1,
            "history_anomalies": ["FALLBACK_EVALUATION"],
        }
