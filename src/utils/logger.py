"""
Logging Utility Module for Multi-Agent Workflow and Data Lineage Tracking.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Union

# Configure standard Python logger for console and server visibility
logger = logging.getLogger("insurance_agent")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def create_log_entry(
    agent: str,
    step: str,
    summary: str,
    data_sources: List[Union[str, Dict[str, Any]]],
    status: str = "SUCCESS"
) -> Dict[str, Any]:
    """
    Create a structured log entry dictionary for agent execution state tracking.

    Args:
        agent (str): Name of the executing agent (e.g. 'Evidence Agent').
        step (str): Title of the execution step.
        summary (str): Detailed summary of what was evaluated/executed.
        data_sources (List[Union[str, Dict[str, Any]]]): Information sources used.
        status (str): Step status ('SUCCESS', 'WARNING', 'FALLBACK').

    Returns:
        Dict[str, Any]: Structured log dictionary.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{agent}] ({step}) - {summary} | Status: {status} | Sources: {len(data_sources)}"
    
    if status == "FALLBACK" or status == "WARNING":
        logger.warning(log_msg)
    else:
        logger.info(log_msg)

    return {
        "timestamp": timestamp,
        "agent": agent,
        "step": step,
        "status": status,
        "summary": summary,
        "data_sources": data_sources,
    }
