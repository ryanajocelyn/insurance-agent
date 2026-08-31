# TECHNICAL DESIGN DOCUMENT (TDD)
## Project: Multi-Agent Motor Insurance Claim Adjudication Assistant
**Framework**: LangGraph | **Vector Database**: ChromaDB | **LLM & Vision**: Gemini 1.5 Pro | **UI**: Streamlit | **Observability**: MLflow Tracing

---

## 1. Executive Summary & Business Objective
Insurance carriers face severe friction during claim adjudication due to fragmented, multimodal evidentiary packets (unstructured accident narratives, photographic damage evidence, garage repair invoices, First Information Reports (FIRs), and historical claim records). Manual assessment is inherently slow, error-prone, and susceptible to fraud or inconsistent claim handling.

This project delivers an automated, explainable **Multi-Agent Claim Adjudication Engine** for Motor Insurance built upon a cyclically and conditionally routed **LangGraph** architecture. The solution combines:
1. **Multimodal Evidence Analysis** via Gemini 1.5 Pro to verify photographic evidence against incident narratives.
2. **Policy RAG Grounding** via ChromaDB to map claimed damages to IRDAI (Insurance Regulatory and Development Authority of India) standard terms, depreciation schedules, and rider endorsements.
3. **Statistical Anomaly & Fraud Pattern Detection** against repair benchmark matrices and historical claim frequencies.
4. **Structured Decision Synthesis** yielding an audit-ready verdict (`APPROVE`, `APPROVE_ADJUSTED`, `ESCALATE`, `REJECT`) backed by clause-level citations and transparent arithmetic.

---

## 2. System Architecture & Component Design

```
                                    +-----------------------------------------+
                                    |        Streamlit Web Interface          |
                                    |  (Packet Upload, Claim Trace, UI View)  |
                                    +--------------------+--------------------+
                                                         |
                                                         v
                                    +-----------------------------------------+
                                    |      LangGraph StateGraph Engine        |
                                    |     (State Management & Routing)        |
                                    +--------------------+--------------------+
                                                         |
                         +-------------------------------+-------------------------------+
                         |                               |                               |
                         v                               v                               v
          +-----------------------------+ +-----------------------------+ +-----------------------------+
          |  Multimodal Evidence Agent  | | Policy Retrieval & Matcher  | |   Cost & History Anomaly    |
          |      (Gemini 1.5 Pro)       | |     (ChromaDB + Gemini)     | |      Evaluation Agent       |
          |  - Damage Severity & Type   | |  - IRDAI Policy Clauses     | |  - Parts Benchmark Lookup   |
          |  - Cross-Modal Consistency  | |  - Deductibles & Riders     | |  - Velocity & Recurrence    |
          |  - Narrative vs Photo Check | |  - Depreciation Schedules   | |  - Fraud Pattern Scoring    |
          +--------------+--------------+ +--------------+--------------+ +--------------+--------------+
                         |                               |                               |
                         +-------------------------------+-------------------------------+
                                                         |
                                                         v
                                    +-----------------------------------------+
                                    |     Synthesis & Adjudication Agent      |
                                    |        - Cross-Modal Verification       |
                                    |        - Conflict & Margin Arbitration  |
                                    |        - Deductible Calculation Engine  |
                                    +--------------------+--------------------+
                                                         |
                                                         v
                                    +-----------------------------------------+
                                    |        Final Decision Output            |
                                    |  [APPROVE / ADJUST / ESCALATE / REJECT] |
                                    |      + Policy Citations & Payout        |
                                    +--------------------+--------------------+
                                                         |
                                                         v
                                    +-----------------------------------------+
                                    |          MLflow Observability           |
                                    |  (Token Latency, Spans, Graph Tracing)  |
                                    +-----------------------------------------+
```

---

## 3. Data Sources & Ingestion Pipeline

### 3.1 Policy & Regulatory Corpus Ingestion (ChromaDB)
* **Regulatory & Policy Source Directory (`docs/`)**:
  The regulatory corpus and policy documents are stored as PDF files across categorized subdirectories within the `docs/` folder:
  * **`docs/guidelines/`**: IRDAI circulars and regulatory guidelines (e.g., Motor Insurance Service Provider, Bima Vahak Guidelines, POS Person rules, Information & Cyber Security for Insurers, Insurance Repositories).
  * **`docs/policy_forms/`**: Standard motor policy forms, rider wordings, and circulars (e.g., Customer Information Sheet (CIS) for Private Car Package Policy, Return to Invoice (RTI) Add-On cover, IMT-29 Compulsory coverage, Motor TP liability rate orders).
  * **`docs/exposure_draft/`**: Regulatory framework drafts (e.g., IRDAI Insurance Fraud Monitoring Framework Guidelines 2024).
  * **`docs/rules/`**: Statutory motor vehicle rules (e.g., Motor Vehicles (Third Party Insurance Base Premium and Liability) Rules 2022).
* **Ingestion Workflow & Execution Model**:
  * **Manual Standalone CLI Execution**: Ingestion is executed manually as a standalone Python CLI program (`python src/rag/ingest_policies.py`).
  * **PDF Parsing & Extraction**: Uses `PyPDFLoader` to read and extract structured text across all PDF files in the four `docs/` subdirectories (`guidelines/`, `policy_forms/`, `exposure_draft/`, `rules/`).
  * **Metadata Tagging**: Each extracted chunk is enriched with metadata properties (`category`, `source_file`, `document_title`, `page_number`) to support category-filtered RAG similarity searches.
  * **Semantic Chunking**: Recursive character text splitting with a chunk size of 800 tokens and 150 token overlap.
  * **Vector Embeddings & Persistence**: Generates embeddings via Google's `models/text-embedding-004` API and persists vectors into local ChromaDB collections (`motor_policy_clauses`, `irdai_guidelines`, `irdai_rules`).

### 3.2 Vehicle Damage Image Corpus (Multimodal)
* **Visual Benchmarking**: Kaggle Car Damage Severity datasets, open vehicle collision libraries (`CarDD`).
* Ingests high-resolution JPG/PNG files covering frontal, side-profile, rear bumper, windshield, undercarriage, and wheel assembly damage.

### 3.3 Repair Cost Benchmarks & Historical Claim Data
* **Cost Lookup Tables (`data/benchmarks/motor_repair_matrix.json`)**:
  * Vehicle segments: Hatchback, Sedan, Compact SUV, Luxury.
  * Line item bounds for OEM Part Replacement vs. Tinkering/Denting vs. Painting charges.
* **Claim History Schema**:
  * Customer ID, Vehicle Reg Number, Policy Inception Date, Prior Claim Dates, Prior Claim Types, Prior Claim Amounts, Loss-to-Premium Ratio (LPR).

---

## 4. LangGraph Multi-Agent Workflow Specification

### 4.1 Global State Definition (`ClaimState`)
```python
from typing import TypedDict, List, Dict, Any, Optional

class ClaimState(TypedDict):
    # Raw Inputs
    claim_id: str
    policy_number: str
    vehicle_details: Dict[str, Any]      # make, model, variant, age, idv
    incident_narrative: str
    uploaded_images: List[str]          # file paths or base64 data
    estimate_line_items: List[Dict[str, Any]] # part, labor, cost, claimed_action
    customer_history: List[Dict[str, Any]]    # prior claims in past 36 months
    held_policy_endorsements: List[str] # e.g. ["ZERO_DEP", "ENGINE_PROTECT"]
    
    # Multimodal Vision Extractions
    vision_findings: Dict[str, Any]      # detected_damage_areas, severity, consistency_score
    missing_evidence_flags: List[str]
    
    # RAG Extractions
    retrieved_policy_clauses: List[Dict[str, Any]]
    applicable_depreciation_rates: Dict[str, float]
    compulsory_deductible: float
    
    # Anomaly Extractions
    cost_variance_flags: List[Dict[str, Any]]
    frequency_risk_score: float         # 0.0 to 1.0
    history_anomalies: List[str]
    
    # Synthesis & Adjudication Verdict
    cross_modal_consistency: bool
    consistency_notes: str
    adjudication_verdict: str           # "APPROVE", "APPROVE_ADJUSTED", "ESCALATE", "REJECT"
    adjudication_rationale: str
    claimed_amount: float
    approved_amount: float
    deductions_breakdown: Dict[str, float]
    mandatory_citations: List[str]
    investigation_triggers: List[str]
```

### 4.2 Agent Breakdown & Responsibilities

#### Agent 1: Multimodal Evidence Agent
* **Role**: Analyzes damage imagery using Gemini 1.5 Pro multimodal capabilities; checks cross-consistency against `incident_narrative`.
* **Outputs**:
  * Identified damage zones (e.g., `["front_bumper", "radiator_grille", "left_headlight"]`).
  * Assessed impact mechanism (e.g., `"head-on impact at low-to-medium speed"`).
  * Narrative alignment verification.
  * Graceful handling of missing images: sets `missing_evidence_flags: ["REAR_DAMAGE_PHOTO_MISSING"]` without pipeline failure.

#### Agent 2: Policy & Limit Matcher Agent (ChromaDB RAG)
* **Role**: Evaluates coverage, active riders, policy limitations, and depreciation schedules.
* **Outputs**:
  * Semantic matching of damage items to policy clauses.
  * Clause exclusion checks.
  * Compulsory deductible application (e.g., ₹1,000 for engines <= 1500cc; ₹2,000 for > 1500cc).

#### Agent 3: Cost & History Anomaly Agent
* **Role**: Audits financial metrics and claim recurrence velocity.
* **Outputs**:
  * Cost inflation flagging: Compares estimate line items against benchmark bounds.
  * Velocity scoring: Evaluates short inter-claim intervals.

#### Agent 4: Synthesis & Adjudication Engine (Consensus & Settlement)
* **Role**: Arbitrates agent findings, executes mathematical settlement formulas, and renders the formal claim decision.
* **Decision Matrix Rules**:
  * `REJECT`: Direct exclusion trigger.
  * `ESCALATE`: Severe cross-modal mismatch, suspicious repeat claims, or unverified cost spikes (>+100%).
  * `APPROVE_ADJUSTED`: Legitimate incident, but claimed items exceed benchmarks (+30%), lack Nil Dep rider, or include non-covered consumables.
  * `APPROVE`: Total consistency, valid coverage, within cost tolerances.

---

## 5. Technical Stack & Observability

| Layer | Technology | Version / Specifics |
| :--- | :--- | :--- |
| **Language** | Python | `>= 3.10` |
| **Agent Framework** | LangGraph & LangChain Core | `langgraph>=0.2.0`, `langchain-core>=0.3.0` |
| **LLM & Vision** | Google Gemini | `gemini-3.6-flash` / `gemini-1.5-pro` |
| **Relational Database** | SQLite (Abstracted Repository) | `sqlite3` + `jinjasql>=0.1.8` for dynamic SQL templating |
| **Vector Database** | ChromaDB (Abstracted Store) | Persistent local client (`chromadb>=0.5.0`) + `gemini-embedding-001` (768 dims) |
| **Observability** | MLflow | `mlflow>=2.15.0` with `mlflow.langchain.autolog()` |
| **Evaluation Framework** | RAGAS & Custom Metrics | `ragas>=0.1.0` (Context Precision, Recall, Faithfulness, Verdict Accuracy) |
| **Frontend UI** | Streamlit (with Event Streaming) | `streamlit>=1.38.0` with `astream_events` real-time state feedback |
| **Containerization** | Docker & Docker Compose | `Dockerfile` + `docker-compose.yml` |
| **CI/CD Pipeline** | GitHub Actions | `.github/workflows/ci.yml` |
| **Tool Protocol (MCP)** | Model Context Protocol | `mcp>=0.1.0` integration for external API tools |
| **Code Quality & Format** | Black, Pylint, Flake8 | `black>=24.0.0`, `pylint>=3.0.0`, `flake8>=7.0.0` |
| **Testing Framework** | Pytest Suite | `pytest>=8.0.0`, `pytest-mock>=3.12.0`, `pytest-asyncio` |

---

## 6. Project Directory Structure

```text
insurance-adj/
│
├── src/                            # Source package directory
│   ├── app.py                      # Streamlit entry point (real-time streaming, trace visualizer, HITL override)
│   ├── config.py                   # Singleton configuration & environment settings
│   │
│   ├── core/                       # Core engine state, factory, graph & tracer
│   │   ├── __init__.py
│   │   ├── state.py                # TypedDict ClaimState definition
│   │   ├── graph.py                # LangGraph StateGraph pipeline & parallel edges
│   │   ├── tracer.py               # MLflow logging & span manager
│   │   └── factory.py              # Factory Pattern for Agent Node Instantiation
│   │
│   ├── agents/                     # Specialized Multi-Agent Nodes
│   │   ├── __init__.py
│   │   ├── evidence_agent.py       # Gemini 1.5 Pro multimodal extractor & narrative verifier
│   │   ├── policy_agent.py         # Vector RAG & policy clause interpreter
│   │   ├── anomaly_agent.py        # Benchmark cost checker & velocity scoring agent
│   │   └── adjudicator_agent.py    # Consensus arbitration & payout calculation engine
│   │
│   ├── db/                         # Database & Vector Store abstraction
│   │   ├── __init__.py
│   │   ├── base_repository.py      # Abstract Base Class (Repository Pattern for RDBMS)
│   │   ├── sqlite_repository.py    # SQLite Repository implementation
│   │   ├── sql_executor.py         # JinjaSql query execution & abstraction engine
│   │   ├── vector_base.py          # Abstract Base Class for Vector Store
│   │   └── chroma_repository.py    # ChromaDB Vector Store Repository implementation
│   │
│   ├── strategies/                 # Settlement calculation strategies
│   │   ├── __init__.py
│   │   ├── base_strategy.py        # Abstract Strategy for Settlement Calculations
│   │   ├── depreciation_strategy.py# IMT Metal vs Zero-Dep rider calculation strategies
│   │   └── deductible_strategy.py  # Compulsory & voluntary deductible strategy
│   │
│   ├── utils/                      # Reusable Utility Helpers
│   │   ├── __init__.py
│   │   ├── llm_utils.py            # Dedicated Gemini 1.5 Pro invocation & JSON parsing helper
│   │   ├── image_utils.py          # Dedicated image downsampling, compression & encoding helper
│   │   └── file_utils.py           # Dedicated file I/O & PDF/text document parser helper
│   │
│   ├── rag/                        # Policy RAG & Ingestion CLI
│   │   ├── __init__.py
│   │   ├── ingest_policies.py      # Standalone CLI program to ingest PDFs from docs/ subdirectories into ChromaDB
│   │   └── retriever.py            # Clause retrieval engine with similarity thresholds and category filters
│   │
│   ├── eval/                       # Evaluation & Quality Framework
│   │   ├── __init__.py
│   │   ├── run_eval.py             # Automated evaluation runner for RAGAS and custom claim metrics
│   │   ├── ragas_evaluator.py      # RAGAS metrics evaluator (Context Precision, Recall, Faithfulness)
│   │   ├── claim_metrics.py        # Custom claim evaluation metrics (Verdict Precision, MAE)
│   │   └── eval_dataset.json       # Ground-truth evaluation dataset
│   │
│   └── mcp/                        # Model Context Protocol (MCP) Server
│       ├── __init__.py
│       ├── server.py               # MCP Server integration module
│       └── tools.py                # External tool definitions (Vehicle Registry API, FIR Verification)
│
├── tests/                          # Automated Pytest suite
│   ├── conftest.py                 # Pytest shared fixtures & path configuration
│   ├── test_env.py                 # Tests for Config singleton & paths
│   ├── test_utils.py               # Unit tests for image downsampling, LLM helper, file I/O
│   ├── test_repositories.py        # Tests for SQLite & ChromaDB abstraction layers
│   ├── test_strategies.py          # Unit tests for settlement & depreciation strategies
│   ├── test_graph.py               # Automated pytest suite for LangGraph workflows
│   └── test_adjudication.py        # Integration tests for end-to-end claim packets
│
├── docs/                           # Documentation & Raw Regulatory Policy Corpus
│   ├── architecture/               # Technical Architecture Documentation
│   │   ├── technical_design_doc.md
│   │   ├── step_by_step_implementation_plan.md
│   │   ├── walkthrough.md
│   │   └── user_guide.md
│   ├── exposure_draft/             # Raw PDF circulars
│   ├── guidelines/                 # IRDAI guidelines PDFs
│   ├── policy_forms/               # Policy forms & CIS PDFs
│   └── rules/                      # Statutory rules PDFs
│
├── data/                           # Data storage & benchmark tables
│   ├── sql_templates/              # JinjaSql template files (.sql)
│   ├── benchmarks/                 # Spare parts & labor cost reference ranges
│   ├── motor_claims.db             # Local SQLite database
│   └── chroma_db/                  # Local ChromaDB vector database
│
├── Dockerfile                      # Production Docker container definition
├── docker-compose.yml              # Multi-container orchestration (Streamlit UI, MLflow)
├── README.md                       # Quickstart & Repository documentation
├── GEMINI.md                       # Google Gemini reference guide
└── requirements.txt                # Production dependencies
```
