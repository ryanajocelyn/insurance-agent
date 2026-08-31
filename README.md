# Multi-Agent Motor Insurance Claim Adjudication Assistant

[![CI/CD Pipeline](https://github.com/your-org/insurance-adj/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/insurance-adj/actions)

An automated, explainable **Multi-Agent Claim Adjudication Engine** for Motor Insurance built upon a cyclically and conditionally routed **LangGraph** architecture, powered by **Google Gemini 1.5 Pro**, **ChromaDB RAG**, **SQLite + JinjaSql**, **Streamlit**, and **MLflow**.

---

## 🌟 Key Features

- **Multimodal Damage Verification**: Gemini 1.5 Pro damage photo inspection & narrative consistency check.
- **Policy RAG Grounding**: ChromaDB vector search against IRDAI circulars and policy terms (`models/text-embedding-004`).
- **Statistical Anomaly Audit**: Repair benchmark cost variance (+30% clamp, >+100% fraud escalation) and 6-month claim velocity risk scoring.
- **Settlement Mathematics**: Automatic India Motor Tariff (IMT) metal part depreciation scale and compulsory excess calculation.
- **Interactive Streamlit Web UI**: Real-time event streaming, claim trace visualizer, and Human-in-the-Loop (HITL) adjuster payout override.
- **Observability & RAGAS Evaluation**: MLflow tracking and RAGAS metrics evaluation (Context Precision, Recall, Faithfulness).
- **Enterprise Tooling**: Docker containerization (`docker-compose.yml`), GitHub Actions CI/CD (`.github/workflows/ci.yml`), and Model Context Protocol (`mcp/`) server integration.

---

## 📁 Repository Structure

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

---

## 🚀 Quickstart Guide

For detailed execution instructions, see the [End-User Manual](file:///d:/Abiz/Technical/code/insurance-adj/docs/architecture/user_guide.md).

```bash
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Configure API Key
cp .env.example .env

# 3. Run Policy Ingestion CLI (Uses python -m or PYTHONPATH=. to resolve src module)
python -m src.rag.ingest_policies

# 4. Launch Streamlit Web UI
streamlit run src/app.py

# 5. Run Evaluation Benchmarks
python -m src.eval.run_eval --dataset src/eval/eval_dataset.json
```

---

## 📄 Documentation References
- [Technical Design Document](file:///d:/Abiz/Technical/code/insurance-adj/docs/architecture/technical_design_doc.md)
- [Architecture Walkthrough](file:///d:/Abiz/Technical/code/insurance-adj/docs/architecture/walkthrough.md)
- [End-User Manual](file:///d:/Abiz/Technical/code/insurance-adj/docs/architecture/user_guide.md)
- [Gemini Integration Reference](file:///d:/Abiz/Technical/code/insurance-adj/GEMINI.md)
