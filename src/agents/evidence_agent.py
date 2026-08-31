"""
Multimodal Evidence Agent Node Module.
"""

from typing import Dict, Any, List
from pydantic import BaseModel, Field
from src.utils.llm_utils import invoke_gemini_json
from src.utils.image_utils import encode_image_base64


class VisionEvidenceAnalysis(BaseModel):
    """Pydantic schema for structured Multimodal Evidence Agent response."""

    detected_damage_areas: List[str] = Field(
        description="Identified vehicle damage zones"
    )
    impact_mechanism: str = Field(
        description="Assessed impact mechanism and direction of force"
    )
    cross_modal_consistency: bool = Field(
        description="True if damage imagery matches claimed incident narrative; False if there is a discrepancy"
    )
    consistency_notes: str = Field(
        description="Detailed justification for cross-modal consistency assessment"
    )
    missing_evidence_flags: List[str] = Field(
        default_factory=list,
        description="List of missing required damage evidence photos",
    )


def evidence_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    narrative = state.get("incident_narrative", "")
    images = state.get("uploaded_images", [])
    estimate_items = state.get("estimate_line_items", [])

    if not images:
        return {
            "vision_findings": {
                "detected_damage_areas": [],
                "impact_mechanism": "UNVERIFIED_NO_IMAGES",
            },
            "cross_modal_consistency": False,
            "consistency_notes": "No photographic evidence was provided with the claim packet.",
            "missing_evidence_flags": ["MANDATORY_DAMAGE_PHOTO_MISSING"],
        }

    image_payloads: List[Dict[str, str]] = []
    for img_item in images:
        try:
            b64_payload = encode_image_base64(img_item)
            image_payloads.append(b64_payload)
        except Exception as exc:
            print(f"[EVIDENCE AGENT WARNING] Image encoding failed for '{img_item}': {exc}")

    prompt = f"""You are an expert Motor Insurance Forensic Claims Assessor.
Analyze the attached vehicle damage photographs and cross-verify them against the customer's accident narrative and claimed repair estimate line items.

Customer Narrative:
"{narrative}"

Claimed Estimate Line Items:
{estimate_items}

Evaluation Tasks:
1. Identify all visible vehicle damage zones.
2. Determine the assessed impact mechanism and direction of force.
3. Check cross-modal consistency: Does the photographic damage align with the claimed narrative?
4. Identify any missing damage photos required to substantiate all claimed line items.
"""

    try:
        analysis: VisionEvidenceAnalysis = invoke_gemini_json(
            prompt=prompt,
            response_schema=VisionEvidenceAnalysis,
            images=image_payloads if image_payloads else None,
            temperature=0.2,
        )

        return {
            "vision_findings": {
                "detected_damage_areas": analysis.detected_damage_areas,
                "impact_mechanism": analysis.impact_mechanism,
            },
            "cross_modal_consistency": analysis.cross_modal_consistency,
            "consistency_notes": analysis.consistency_notes,
            "missing_evidence_flags": analysis.missing_evidence_flags,
        }

    except Exception as exc:
        print(f"[EVIDENCE AGENT FALLBACK] Gemini vision analysis call failed: {exc}")
        return {
            "vision_findings": {
                "detected_damage_areas": ["UNKNOWN"],
                "impact_mechanism": "FALLBACK_TIMEOUT",
            },
            "cross_modal_consistency": True,
            "consistency_notes": f"Fallback mode active due to vision service timeout: {exc}",
            "missing_evidence_flags": ["VISION_AGENT_TIMEOUT"],
        }
