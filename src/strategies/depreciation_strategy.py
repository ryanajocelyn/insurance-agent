"""
Depreciation Calculation Strategy Module.
"""

from typing import Dict, Any, List
from src.strategies.base_strategy import BaseSettlementStrategy


class DepreciationStrategy(BaseSettlementStrategy):
    """IMT Depreciation calculation strategy with Nil Dep rider support."""

    @staticmethod
    def get_metal_depreciation_rate(vehicle_age_years: float) -> float:
        if vehicle_age_years <= 0.5:
            return 0.00
        elif vehicle_age_years <= 1.0:
            return 0.05
        elif vehicle_age_years <= 2.0:
            return 0.10
        elif vehicle_age_years <= 3.0:
            return 0.15
        elif vehicle_age_years <= 4.0:
            return 0.25
        elif vehicle_age_years <= 5.0:
            return 0.35
        elif vehicle_age_years <= 10.0:
            return 0.40
        else:
            return 0.50

    def calculate(
        self,
        estimate_items: List[Dict[str, Any]],
        vehicle_details: Dict[str, Any],
        held_endorsements: List[str],
    ) -> Dict[str, Any]:
        has_zero_dep = any(
            rider.upper() in ["ZERO_DEP", "NIL_DEP", "ZERO_DEPRECIATION"]
            for rider in held_endorsements
        )
        vehicle_age = float(vehicle_details.get("age_years", 2.0))
        metal_rate = self.get_metal_depreciation_rate(vehicle_age)

        total_depreciation_deduction = 0.0
        applied_rates: Dict[str, float] = {
            "metal": metal_rate if not has_zero_dep else 0.0,
            "plastic_rubber": 0.50 if not has_zero_dep else 0.0,
            "fiberglass": 0.30 if not has_zero_dep else 0.0,
            "glass": 0.00,
        }

        item_breakdown: List[Dict[str, Any]] = []

        for item in estimate_items:
            part_name = item.get("part_name", "Unknown Part")
            category = str(item.get("category", "metal")).lower()
            claimed_cost = float(item.get("claimed_cost", 0.0))

            if has_zero_dep:
                rate = 0.0
            else:
                if "glass" in category and "fiber" not in category:
                    rate = 0.00
                elif "fiber" in category:
                    rate = 0.30
                elif "rubber" in category or "plastic" in category or "nylon" in category:
                    rate = 0.50
                elif "labor" in category or "painting" in category or "tinkering" in category:
                    rate = 0.00
                else:
                    rate = metal_rate

            dep_amount = round(claimed_cost * rate, 2)
            adjusted_cost = round(claimed_cost - dep_amount, 2)
            total_depreciation_deduction += dep_amount

            item_breakdown.append(
                {
                    "part_name": part_name,
                    "category": category,
                    "claimed_cost": claimed_cost,
                    "depreciation_rate": rate,
                    "depreciation_deduction": dep_amount,
                    "adjusted_cost": adjusted_cost,
                }
            )

        return {
            "has_zero_dep_rider": has_zero_dep,
            "applied_rates": applied_rates,
            "total_depreciation_deduction": round(total_depreciation_deduction, 2),
            "item_breakdown": item_breakdown,
        }
