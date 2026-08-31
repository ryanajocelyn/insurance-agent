"""
Agent Factory Pattern Module.

Provides the `AgentFactory` class for dynamically instantiating agent node handlers
and binding model dependencies based on global configuration.
"""

from typing import Dict, Any, Callable
from src.config import config


class AgentFactory:
    """Factory Pattern class creating agent node functions."""

    @staticmethod
    def get_agent_node(agent_name: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """
        Dynamically retrieve an agent node execution function by name.

        Args:
            agent_name (str): Identifier name ('evidence', 'policy', 'anomaly', 'adjudicator').

        Returns:
            Callable[[Dict[str, Any]], Dict[str, Any]]: Node handler function taking ClaimState.
        """
        name = agent_name.lower().strip()
        if name in ("evidence", "evidence_agent"):
            from src.agents.evidence_agent import evidence_agent_node
            return evidence_agent_node
        elif name in ("policy", "policy_agent"):
            from src.agents.policy_agent import policy_agent_node
            return policy_agent_node
        elif name in ("anomaly", "anomaly_agent"):
            from src.agents.anomaly_agent import anomaly_agent_node
            return anomaly_agent_node
        elif name in ("adjudicator", "adjudicator_agent"):
            from src.agents.adjudicator_agent import adjudicator_agent_node
            return adjudicator_agent_node
        else:
            raise ValueError(f"Unknown agent node requested: '{agent_name}'")
