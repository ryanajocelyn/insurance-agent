"""
Policy & Limit Matcher Agent Node Module (ChromaDB RAG).
"""

from typing import Dict, Any, List
from pydantic import BaseModel, Field
from src.rag.retriever import PolicyRetriever
from src.utils.llm_utils import invoke_gemini_json


class PolicyMatcherAnalysis(BaseModel):
    """Pydantic schema for Policy & Limit Matcher Agent response."""

    matched_clause_summaries: List[str] = Field(
        description="Summaries of applicable policy clauses and IRDAI rules"
    )
    mandatory_citations: List[str] = Field(
        description="Exact policy clause and circular citations"
    )
    non_covered_items: List[str] = Field(
        default_factory=list,
        description="Line items excluded from policy coverage",
    )
    policy_warning_flags: List[str] = Field(
        default_factory=list,
        description="Warnings regarding missing riders or partial coverage",
    )


def policy_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    narrative = state.get("incident_narrative", "")
    estimate_items = state.get("estimate_line_items", [])
    endorsements = state.get("held_policy_endorsements", [])
    vehicle = state.get("vehicle_details", {})

    retriever = PolicyRetriever()

    query_text = f"{narrative} " + " ".join([item.get("part_name", "") for item in estimate_items])
    retrieved_clauses = retriever.retrieve_relevant_clauses(query=query_text, n_results=4)

    has_consumables_rider = any(
        r.upper() in ["CONSUMABLES_COVER", "CONSUMABLES"] for r in endorsements
    )
    consumables_keywords = ["oil", "coolant", "nuts", "bolts", "ac gas", "fluid", "filter", "grease"]

    warnings: List[str] = []
    non_covered: List[str] = []

    for item in estimate_items:
        part_name = item.get("part_name", "").lower()
        if any(kw in part_name for kw in consumables_keywords):
            if not has_consumables_rider:
                non_covered.append(item.get("part_name", ""))
                if "CONSUMABLES_NOT_COVERED_WARNING" not in warnings:
                    warnings.append("CONSUMABLES_NOT_COVERED_WARNING")

    prompt = f"""You are an expert Indian Motor Insurance Policy Underwriter & Claims Auditor.
Analyze the retrieved policy clauses against the claimed items, vehicle specifications, and active policy rider endorsements.

Active Policy Riders: {endorsements}
Vehicle Details: {vehicle}
Claimed Line Items: {estimate_items}

Retrieved IRDAI Policy & Regulatory Context:
{retrieved_clauses}

Evaluation Tasks:
1. Identify all applicable policy coverage clauses and specific section citations.
2. Determine if any claimed items are excluded under standard Own Damage (OD) terms.
3. Generate exact clause citations.
4. List any policy warning flags or non-covered items.
"""

    try:
        analysis: PolicyMatcherAnalysis = invoke_gemini_json(
            prompt=prompt,
            response_schema=PolicyMatcherAnalysis,
            temperature=0.2,
        )

        all_warnings = list(set(warnings + analysis.policy_warning_flags))
        all_non_covered = list(set(non_covered + analysis.non_covered_items))

        return {
            "retrieved_policy_clauses": retrieved_clauses,
            "mandatory_citations": analysis.mandatory_citations,
            "policy_warning_flags": all_warnings,
        }

    except Exception as exc:
        print(f"[POLICY AGENT FALLBACK] RAG synthesis call failed: {exc}")
        return {
            "retrieved_policy_clauses": retrieved_clauses,
            "mandatory_citations": ["IMT Standard Own Damage Section I"],
            "policy_warning_flags": warnings,
        }
