# Step-by-Step Implementation Plan: Multi-Agent Motor Insurance Claim Adjudication Assistant

## Executive Summary
This document provides a modular, step-by-step implementation plan for building the **Multi-Agent Motor Insurance Claim Adjudication Assistant** using **LangGraph**, **Google Gemini 1.5 Pro**, **ChromaDB**, **SQLite + JinjaSql**, **Streamlit**, and **MLflow**.

---

## Phase 1: Base Environment & Configuration Setup
- **Goal**: Establish project directory structure, dependencies, configuration loader, and baseline pytest environment.
- **Files Created**:
  - `requirements.txt`
  - `src/config.py`
  - `.env.example`
  - `tests/test_env.py`

## Phase 2: Core Data Models & Reusable Utility Helpers
- **Goal**: Implement graph state representation and dedicated helper modules for LLM invocation, image downsampling, and file I/O.
- **Files Created**:
  - `src/core/state.py` (`ClaimState` TypedDict definition)
  - `src/utils/llm_utils.py` (`invoke_gemini_json()` with exponential backoff retries)
  - `src/utils/image_utils.py` (`compress_and_downsample_image()` and `encode_image_base64()`)
  - `src/utils/file_utils.py` (JSON, text, and PDF page-by-page text parsing)
  - `tests/test_utils.py`

## Phase 3: Relational DB & Vector Store Abstraction Layer
- **Goal**: Abstract database interactions using the Repository Pattern and JinjaSql query templating.
- **Files Created**:
  - `src/db/base_repository.py` & `src/db/vector_base.py` (Abstract interfaces)
  - `src/db/sql_executor.py` (JinjaSql query template renderer)
  - `src/db/sqlite_repository.py` & `data/sql_templates/`
  - `src/db/chroma_repository.py` (ChromaDB wrapper with `models/text-embedding-004`)
  - `data/benchmarks/motor_repair_matrix.json` (Benchmark repair bounds matrix)
  - `tests/test_repositories.py`

## Phase 4: Regulatory Policy Document Ingestion CLI Pipeline
- **Goal**: Provide a standalone CLI program to parse, chunk, embed, and index raw regulatory PDF documents into ChromaDB.
- **Files Created**:
  - `src/rag/ingest_policies.py` (CLI ingestion script scanning `docs/` subdirectories: `guidelines`, `policy_forms`, `exposure_draft`, `rules`)
  - `src/rag/retriever.py` (Policy RAG retriever engine)

## Phase 5: Business Rule Calculation Strategies
- **Goal**: Encapsulate settlement mathematics into Strategy Pattern implementations.
- **Files Created**:
  - `src/strategies/base_strategy.py` (Abstract strategy interface)
  - `src/strategies/depreciation_strategy.py` (IMT metal scale vs Zero-Dep rider override)
  - `src/strategies/deductible_strategy.py` (Compulsory excess subtraction)
  - `tests/test_strategies.py`

## Phase 6: Specialized Multi-Agent Nodes
- **Goal**: Construct dedicated agent node handlers utilizing Agent Factory instantiation.
- **Files Created**:
  - `src/core/factory.py` (`AgentFactory` pattern)
  - `src/agents/evidence_agent.py` (Gemini 1.5 Pro multimodal damage analysis)
  - `src/agents/policy_agent.py` (ChromaDB policy clause RAG matcher)
  - `src/agents/anomaly_agent.py` (Cost benchmark auditor & claim velocity risk scorer)
  - `src/agents/adjudicator_agent.py` (Synthesis arbitrator & payout math calculator)

## Phase 7: LangGraph Orchestration & MLflow Observability
- **Goal**: Compile StateGraph workflow with parallel fan-out execution and log execution runs to MLflow.
- **Files Created**:
  - `src/core/tracer.py` (MLflow tracking manager)
  - `src/core/graph.py` (`StateGraph` pipeline definition)
  - `tests/test_graph.py` & `tests/test_adjudication.py`

## Phase 8: Evaluation & Quality Benchmarking Framework
- **Goal**: Implement automated quality benchmarking with RAGAS metrics and domain claim metrics.
- **Files Created**:
  - `src/eval/eval_dataset.json` (Ground-truth dataset)
  - `src/eval/ragas_evaluator.py` (Context Precision, Recall, Faithfulness)
  - `src/eval/claim_metrics.py` (Verdict Precision & Payout MAE)
  - `src/eval/run_eval.py` (CLI evaluation runner)

## Phase 9: Streamlit Web UI with Live Event Streaming & HITL
- **Goal**: Build an interactive web frontend featuring live trace visualization and adjuster override functionality.
- **Files Created**:
  - `src/app.py` (Streamlit web application)

## Phase 10: Enterprise Tooling, Containerization & CI/CD
- **Goal**: Package application for multi-container deployment and setup GitHub Actions workflow.
- **Files Created**:
  - `src/mcp/server.py` & `src/mcp/tools.py` (Model Context Protocol server)
  - `Dockerfile` & `docker-compose.yml`
  - `.github/workflows/ci.yml`

## Phase 11: Final Architecture Documentation & User Guide
- **Goal**: Finalize architecture documentation, operator guide, and repository README.
- **Files Created**:
  - `docs/architecture/technical_design_doc.md`
  - `docs/architecture/walkthrough.md`
  - `docs/architecture/user_guide.md`
  - `README.md`
