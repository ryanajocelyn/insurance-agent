"""
LangGraph Multi-Agent StateGraph Engine Module.

Defines and compiles the cyclically and conditionally routed LangGraph workflow.
Executes Multimodal Evidence Agent, Policy RAG Agent, and Anomaly Agent concurrently
in parallel before converging at the Synthesis Adjudication Node.
"""

import time
from typing import Dict, Any, List
from langgraph.graph import StateGraph, START, END
from src.core.state import ClaimState
from src.core.factory import AgentFactory
from src.core.tracer import MLflowTracer


def build_adjudication_graph() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph pipeline.

    Parallel Execution Flow:
                    +--------> Evidence Agent --------+
                    |                                 |
        START ------+--------> Policy RAG Agent ------+------> Synthesis Adjudicator -----> END
                    |                                 |
                    +--------> Anomaly Agent ---------+

    Returns:
        StateGraph: Compiled LangGraph workflow pipeline.
    """
    builder = StateGraph(ClaimState)

    # Add agent node handlers via AgentFactory
    builder.add_node("evidence_agent", AgentFactory.get_agent_node("evidence"))
    builder.add_node("policy_agent", AgentFactory.get_agent_node("policy"))
    builder.add_node("anomaly_agent", AgentFactory.get_agent_node("anomaly"))
    builder.add_node("adjudicator_agent", AgentFactory.get_agent_node("adjudicator"))

    # Parallel Fan-Out Edges from START to Evidence, Policy, and Anomaly agents
    builder.add_edge(START, "evidence_agent")
    builder.add_edge(START, "policy_agent")
    builder.add_edge(START, "anomaly_agent")

    # Convergence Edges from parallel agents into Synthesis Adjudicator node
    builder.add_edge("evidence_agent", "adjudicator_agent")
    builder.add_edge("policy_agent", "adjudicator_agent")
    builder.add_edge("anomaly_agent", "adjudicator_agent")

    # Terminal Edge from Adjudicator to END
    builder.add_edge("adjudicator_agent", END)

    # Compile StateGraph
    compiled_graph = builder.compile()
    return compiled_graph


def run_claim_adjudication(initial_state: ClaimState) -> ClaimState:
    """
    Execute compiled LangGraph multi-agent workflow against an input claim state packet.

    Args:
        initial_state (ClaimState): Raw input claim packet data.

    Returns:
        ClaimState: Final state containing adjudication verdict, rationale, and approved payout metrics.
    """
    claim_id = initial_state.get("claim_id", "CLM-UNKNOWN")
    print(f"\n========================================================", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] [LANGGRAPH START] Executing Multi-Agent Adjudication Workflow for '{claim_id}'...", flush=True)
    print(f"========================================================", flush=True)

    graph = build_adjudication_graph()

    start_time = time.time()
    final_state: ClaimState = graph.invoke(initial_state)
    latency = time.time() - start_time

    verdict = final_state.get("adjudication_verdict", "UNKNOWN")
    logs_count = len(final_state.get("execution_logs", []))
    print(f"\n========================================================", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] [LANGGRAPH END] Workflow Completed in {latency:.2f}s | Verdict: {verdict} | Step Logs: {logs_count}", flush=True)
    print(f"========================================================\n", flush=True)

    # Log run to MLflow observability tracer
    MLflowTracer.log_claim_adjudication_run(final_state, execution_time_seconds=latency)

    return final_state


