import sys
import logging
from datetime import datetime
from typing import Dict, Any, List, Union

# Configure standard Python logger for console and server visibility
logger = logging.getLogger("insurance_agent")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
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
    Create a structured log entry dictionary for agent execution state tracking
    and print immediately to stdout for real-time terminal console visibility.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time_short = datetime.now().strftime("%H:%M:%S")
    log_msg = f"[{agent}] ({step}) - {summary} | Status: {status} | Sources: {len(data_sources)}"
    
    # Print directly to stdout with flush=True to ensure real-time terminal logging in Streamlit
    print(f"[{time_short}] [{status}] [{agent}] ({step}) - {summary} | Sources: {len(data_sources)}", flush=True)

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

