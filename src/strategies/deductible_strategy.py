"""
Compulsory Deductible & Voluntary Excess Strategy Module.
"""

from typing import Dict, Any, List
from src.strategies.base_strategy import BaseSettlementStrategy


class DeductibleStrategy(BaseSettlementStrategy):
    """Compulsory and Voluntary Deductible calculation strategy."""

    @staticmethod
    def get_compulsory_deductible(cubic_capacity: float) -> float:
        if cubic_capacity <= 1500.0:
            return 1000.0
        else:
            return 2000.0

    def calculate(
        self,
        estimate_items: List[Dict[str, Any]],
        vehicle_details: Dict[str, Any],
        held_endorsements: List[str],
    ) -> Dict[str, Any]:
        cc = float(vehicle_details.get("cubic_capacity", vehicle_details.get("cc", 1200.0)))
        compulsory_deductible = self.get_compulsory_deductible(cc)
        voluntary_deductible = float(vehicle_details.get("voluntary_deductible", 0.0))

        total_deductible = compulsory_deductible + voluntary_deductible

        gross_adjusted_amount = sum(
            float(item.get("adjusted_cost", item.get("claimed_cost", 0.0))) for item in estimate_items
        )

        net_payable_amount = max(0.0, gross_adjusted_amount - total_deductible)

        return {
            "engine_cc": cc,
            "compulsory_deductible": compulsory_deductible,
            "voluntary_deductible": voluntary_deductible,
            "total_deductible": total_deductible,
            "gross_adjusted_amount": round(gross_adjusted_amount, 2),
            "net_payable_amount": round(net_payable_amount, 2),
        }
