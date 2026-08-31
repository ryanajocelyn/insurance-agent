# Multi-Agent Motor Insurance Claim Adjudication Assistant: Technical Walkthrough

## Executive Summary
This document provides a comprehensive technical walkthrough of the **Multi-Agent Motor Insurance Claim Adjudication Assistant**, built with **LangGraph**, **Google Gemini 1.5 Pro**, **ChromaDB RAG**, **SQLite + JinjaSql**, **Streamlit**, and **MLflow**.

---

## 1. System Architecture & Multi-Agent Execution Flow

```
                                    +-----------------------------------------+
                                    |        Streamlit Web Interface          |
                                    |  (Packet Upload, Live Trace, HITL UI)   |
                                    +--------------------+--------------------+
                                                         |
                                                         v
                                    +-----------------------------------------+
                                    |      LangGraph StateGraph Engine        |
                                    |   (Parallel Fan-Out & Routing Engine)   |
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
                                    |        - IMT Depreciation Calculation   |
                                    |        - Compulsory Deductible Subtraction |
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

## 2. Implemented Design Patterns & Architecture

1. **Repository / DAO Pattern (`src/db/base_repository.py`, `src/db/vector_base.py`)**:
   Isolates relational SQLite persistence (`SQLiteRepository`) and vector storage (`ChromaRepository`) behind abstract interfaces.
2. **Strategy Pattern (`src/strategies/depreciation_strategy.py`, `src/strategies/deductible_strategy.py`)**:
   Encapsulates India Motor Tariff (IMT) metal depreciation scale, Zero-Dep rider override, and compulsory excess rules.
3. **Factory Pattern (`src/core/factory.py`)**:
   `AgentFactory` dynamically instantiates agent node functions and model bindings.
4. **Singleton Pattern (`src/config.py`)**:
   Thread-safe configuration loader.
5. **JinjaSql Dynamic SQL Engine (`src/db/sql_executor.py`)**:
   Renders parameterized SQL queries safely from `.sql` template files.

---

## 3. Key Operational Thresholds

- **Adjustment Clamp (`APPROVE_ADJUSTED`)**: Claimed part costs up to **+30%** above benchmark are adjusted down to benchmark ceilings.
- **Fraud Escalation (`ESCALATE`)**: Costs exceeding benchmark by **> +100%** trigger an immediate `ESCALATE` verdict.
- **Claim Velocity Horizon**: **6 months** lookback window for recurrence risk scoring.
- **FIR Requirement Trigger**: Mandatory for **Third-Party Injury** or **loss exceeding ₹50,000**.
- **Gemini Temperature**: Set to **`0.2`** for deterministic policy alignment.

---

## 4. Evaluation & Quality Benchmarks (`src/eval/`)

Evaluation metrics calculated via `src/eval/run_eval.py`:
- **Verdict Accuracy**: Precision, Recall, and F1-score across ground-truth decisions.
- **Payout Settlement MAE**: Mean Absolute Error on approved payout amounts.
- **RAGAS Metrics**: Context Precision, Context Recall, Faithfulness (zero hallucination tolerance), and Answer Relevance.

---

## 5. Verification & Testing Log

- **Environment & Config Tests**: Passed (`tests/test_env.py`).
- **Utility & Image Preprocessing Tests**: Passed (`tests/test_utils.py`).
- **DB & ChromaDB Repository Tests**: Passed (`tests/test_repositories.py`).
- **Strategy Math Tests**: Passed (`tests/test_strategies.py`).
- **LangGraph Workflow Tests**: Passed (`tests/test_graph.py`, `tests/test_adjudication.py`).
