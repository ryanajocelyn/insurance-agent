from typing import Dict, Any, List
from pydantic import BaseModel, Field
from src.rag.retriever import PolicyRetriever
from src.utils.llm_utils import invoke_gemini_json
from src.utils.logger import create_log_entry


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

    # Format retrieved sources for log provenance
    retrieved_sources_desc = []
    for rc in retrieved_clauses:
        doc_title = rc.get("document_title") or rc.get("source_file", "IRDAI Policy")
        score = rc.get("similarity_score", 0.0)
        page = rc.get("page_number", 1)
        retrieved_sources_desc.append(f"[DOC] '{doc_title}' (Page {page}, Similarity: {score:.2f})")

    data_sources = [
        "Vector Database Store: ChromaDB (Collection: 'motor_policy_clauses', Embedding: 'gemini-embedding-001')",
        f"Retrieved Documents ({len(retrieved_clauses)} matches): {'; '.join(retrieved_sources_desc) if retrieved_sources_desc else 'None'}",
        f"Active Policy Endorsements / Riders: {endorsements if endorsements else 'None'}",
        "Model Engine: Google Gemini 1.5 Pro (gemini-3.6-flash)"
    ]

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

        summary_text = (
            f"Retrieved {len(retrieved_clauses)} relevant clauses. "
            f"Mandatory Citations: {', '.join(analysis.mandatory_citations)}. "
            f"Warnings: {', '.join(all_warnings) or 'None'}. "
            f"Excluded Items: {', '.join(all_non_covered) or 'None'}"
        )

        status_flag = "WARNING" if all_warnings or all_non_covered else "SUCCESS"

        success_log = create_log_entry(
            agent="Policy & Limit Matcher Agent",
            step="ChromaDB RAG Retrieval & Policy Coverage Verification",
            summary=summary_text,
            data_sources=data_sources,
            status=status_flag
        )

        return {
            "retrieved_policy_clauses": retrieved_clauses,
            "mandatory_citations": analysis.mandatory_citations,
            "policy_warning_flags": all_warnings,
            "execution_logs": [success_log],
        }

    except Exception as exc:
        print(f"[POLICY AGENT FALLBACK] RAG synthesis call failed: {exc}")
        fallback_log = create_log_entry(
            agent="Policy & Limit Matcher Agent",
            step="ChromaDB RAG Retrieval & Policy Coverage Verification",
            summary=f"Policy RAG fallback triggered due to service exception: {exc}",
            data_sources=data_sources,
            status="FALLBACK"
        )

        return {
            "retrieved_policy_clauses": retrieved_clauses,
            "mandatory_citations": ["IMT Standard Own Damage Section I"],
            "policy_warning_flags": warnings,
            "execution_logs": [fallback_log],
        }

