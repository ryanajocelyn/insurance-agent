# GEMINI.md - Google Gemini 1.5 Pro & GenAI Integration Reference Guide

## 1. Executive Summary & Overview
This repository utilizes **Google Gemini** (`gemini-3.6-flash` / `gemini-1.5-pro`) and **Google GenAI Text Embedding** (`gemini-embedding-001`) for multimodal motor claim adjudication, policy interpretation, visual damage inspection, and decision synthesis.

This document serves as the authoritative reference guide for prompt engineering conventions, API usage parameters, structured JSON output rules, image preprocessing, error handling, and rate-limit mitigation strategies across all agent nodes.

---

## 2. Model Specifications & Parameters

| Operational Component | Model Target | Default Setting | Purpose |
| :--- | :--- | :--- | :--- |
| **Multimodal Vision & Reasoning** | `gemini-3.6-flash` | `temperature=0.2` | Damage photo analysis, narrative consistency, policy interpretation, claim adjudication synthesis |
| **Vector Embeddings** | `gemini-embedding-001` | `output_dimensionality=768` | Policy clause and regulatory document vector embeddings for ChromaDB retrieval |

### Key Model Configuration Rules:
1. **Temperature (`0.2`)**: Controls deterministic reasoning while allowing subtle contextual flexibility when comparing unstructured accident narratives with photographic damage.
2. **Native JSON Output (`response_mime_type="application/json"`)**: All Gemini 1.5 Pro agent calls MUST strictly specify `response_mime_type="application/json"` paired with explicit Pydantic schema models to guarantee structured, parseable responses.
3. **Safety Settings**: Configured with standard enterprise safety thresholds to prevent false positive triggers on accident injury text or damage descriptions.

---

## 3. Multimodal Damage Image Handling

Gemini 1.5 Pro accepts native image inputs (Base64 encoded string or raw bytes) alongside text prompts.

### 3.1 Preprocessing & Downsampling Rules (`src/utils/image_utils.py`)
To prevent API payload inflation and minimize response latency, all input damage photographs must pass through the `compress_and_downsample_image()` utility before transmission:
- **Maximum Resolution**: Downsample high-res photos to max `1024x1024` pixels while preserving aspect ratio.
- **Compression**: Save compressed JPEG images at `85%` quality.
- **Format**: Convert PNG/WEBP to JPEG prior to Base64 encoding.

### 3.2 Prompting Pattern for Vision Verification
When passing damage images to Gemini 1.5 Pro in `src/agents/evidence_agent.py`:
- Always pair images with the claimed `incident_narrative` and estimated repair line items.
- Instruct Gemini to evaluate:
  1. Primary damage location (e.g., front bumper vs rear tailgate).
  2. Severity classification (`MINOR_SCRATCH`, `MODERATE_CRUMPLE`, `SEVERE_STRUCTURAL`).
  3. Direction of impact force.
  4. Cross-modal consistency boolean flag (`true`/`false`) comparing narrative claims against visual evidence.

---

## 4. Structured JSON Output Guidelines & Pydantic Schema Enforcement

To eliminate unpredictable markdown formatting or raw text parsing errors, all Gemini invocations use the centralized helper `invoke_gemini_json()` in `src/utils/llm_utils.py`.

### 4.1 Invocation Pattern Example (`src/utils/llm_utils.py`)
```python
from pydantic import BaseModel, Field
from typing import List, Optional

class VisionEvidenceAnalysis(BaseModel):
    detected_damage_areas: List[str] = Field(description="Identified vehicle damage zones")
    impact_mechanism: str = Field(description="Assessed impact mechanism and force direction")
    cross_modal_consistency: bool = Field(description="True if damage matches narrative claims")
    consistency_notes: str = Field(description="Detailed rationale for consistency evaluation")
    missing_evidence_flags: List[str] = Field(default_factory=list, description="Missing required photos")

# Invocation via Gemini API wrapper
def analyze_damage_evidence(narrative: str, image_b64_list: List[str]) -> VisionEvidenceAnalysis:
    prompt = f"Analyze the following vehicle accident narrative and damage images...\nNarrative: {narrative}"
    return invoke_gemini_json(
        prompt=prompt,
        images=image_b64_list,
        response_schema=VisionEvidenceAnalysis,
        temperature=0.2
    )
```

### 4.2 Error Handling & Fallback Strategy
1. **Exponential Backoff**: Retry API calls up to 3 times on rate-limit (`429`) or temporary API errors with exponential backoff (`2s`, `4s`, `8s`).
2. **Schema Parsing Validation**: If output JSON fails Pydantic validation, re-attempt invocation with an explicit schema correction prompt.
3. **Graceful Agent Failure**: If Gemini calls fail completely, set a safe agent fallback state (e.g., `missing_evidence_flags: ["VISION_AGENT_TIMEOUT"]`, triggering `ESCALATE` verdict) to ensure graph execution never crashes.

---

## 5. RAG Prompting & Policy Citation Rules

When retrieving IRDAI regulations and policy terms in `src/agents/policy_agent.py`:
1. **Strict Context Grounding**: System prompts MUST instruct Gemini to cite specific clause numbers (e.g., *"IRDAI OD Section 1.b Exclusion"*, *"IMT Endorsement 22"*) from the retrieved ChromaDB context.
2. **Zero Hallucination Policy**: If a specific repair item or damage condition is not covered in the retrieved policy context, Gemini MUST flag it as non-covered rather than assuming coverage.
3. **Consumables Handling**: If line items contain consumables (oil, coolant, nuts/bolts) and the policy lacks a "Consumables Cover" rider, emit an explicit non-coverage flag.

---

## 6. Code Style & Integration Best Practices

1. **Centralized Client**: Access the Gemini client strictly through singleton configurations in `src/config.py` using `GOOGLE_API_KEY`.
2. **MLflow Tracing**: Ensure LLM invocation spans are wrapped with MLflow autologging (`mlflow.langchain.autolog()`) to capture execution latency and token metrics.
3. **No Direct Key Hardcoding**: Never hardcode API keys in scripts or markdown files; load exclusively from `.env`.
