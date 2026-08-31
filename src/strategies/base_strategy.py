"""
Abstract Settlement Calculation Strategy Interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseSettlementStrategy(ABC):
    """Abstract Base Class for claim settlement calculation strategies."""

    @abstractmethod
    def calculate(
        self,
        estimate_items: List[Dict[str, Any]],
        vehicle_details: Dict[str, Any],
        held_endorsements: List[str],
    ) -> Dict[str, Any]:
        pass
