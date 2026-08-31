"""
Unit Tests for Settlement Calculation Strategies (Phase 5).
"""

import pytest
from src.strategies.depreciation_strategy import DepreciationStrategy
from src.strategies.deductible_strategy import DeductibleStrategy


def test_imt_metal_depreciation_scale():
    """Verify age-based metal depreciation rates per IMT schedule."""
    assert DepreciationStrategy.get_metal_depreciation_rate(0.4) == 0.00
    assert DepreciationStrategy.get_metal_depreciation_rate(0.8) == 0.05
    assert DepreciationStrategy.get_metal_depreciation_rate(1.5) == 0.10
    assert DepreciationStrategy.get_metal_depreciation_rate(2.5) == 0.15
    assert DepreciationStrategy.get_metal_depreciation_rate(3.5) == 0.25
    assert DepreciationStrategy.get_metal_depreciation_rate(4.5) == 0.35
    assert DepreciationStrategy.get_metal_depreciation_rate(7.0) == 0.40
    assert DepreciationStrategy.get_metal_depreciation_rate(12.0) == 0.50


def test_depreciation_calculation_without_zero_dep():
    """Verify part-level depreciation calculation without Zero Dep rider."""
    strategy = DepreciationStrategy()
    items = [
        {"part_name": "Front Bumper", "category": "plastic", "claimed_cost": 4000.0},
        {"part_name": "Bonnet Hood", "category": "metal", "claimed_cost": 10000.0},
        {"part_name": "Windshield Glass", "category": "glass", "claimed_cost": 5000.0},
        {"part_name": "Tinkering Labor", "category": "labor", "claimed_cost": 2000.0},
    ]
    vehicle = {"age_years": 2.5}

    result = strategy.calculate(items, vehicle, held_endorsements=[])
    assert not result["has_zero_dep_rider"]
    assert result["total_depreciation_deduction"] == 3500.0


def test_depreciation_calculation_with_zero_dep_rider():
    """Verify Zero Depreciation rider overrides part depreciation to 0."""
    strategy = DepreciationStrategy()
    items = [
        {"part_name": "Front Bumper", "category": "plastic", "claimed_cost": 4000.0},
        {"part_name": "Bonnet Hood", "category": "metal", "claimed_cost": 10000.0},
    ]
    vehicle = {"age_years": 4.0}

    result = strategy.calculate(items, vehicle, held_endorsements=["ZERO_DEP"])
    assert result["has_zero_dep_rider"]
    assert result["total_depreciation_deduction"] == 0.0


def test_compulsory_deductible_by_engine_cc():
    """Verify compulsory excess based on engine cc limits."""
    assert DeductibleStrategy.get_compulsory_deductible(1197.0) == 1000.0
    assert DeductibleStrategy.get_compulsory_deductible(1498.0) == 1000.0
    assert DeductibleStrategy.get_compulsory_deductible(1998.0) == 2000.0


def test_deductible_strategy_settlement_calculation():
    """Verify compulsory deductible subtraction after depreciation."""
    strategy = DeductibleStrategy()
    items = [
        {"part_name": "Front Bumper", "adjusted_cost": 3000.0},
        {"part_name": "Headlight", "adjusted_cost": 5000.0},
    ]
    vehicle = {"cubic_capacity": 1200.0, "voluntary_deductible": 0.0}

    result = strategy.calculate(items, vehicle, held_endorsements=[])
    assert result["compulsory_deductible"] == 1000.0
    assert result["gross_adjusted_amount"] == 8000.0
    assert result["net_payable_amount"] == 7000.0
