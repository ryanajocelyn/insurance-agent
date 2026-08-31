"""
Unit Tests for Base Environment and Configuration Setup (Phase 1).
"""

import os
import pytest
from src.config import Config, config


def test_config_singleton():
    """Verify Config follows Singleton design pattern."""
    config_a = Config()
    config_b = Config()
    assert config_a is config_b
    assert config_a is config


def test_config_default_thresholds():
    """Verify required business rule defaults."""
    assert config.COST_ADJUSTMENT_THRESHOLD == 0.30
    assert config.FRAUD_ESCALATION_THRESHOLD == 1.00
    assert config.CLAIM_VELOCITY_LOOKBACK_MONTHS == 6
    assert config.FIR_REQUIRED_LOSS_THRESHOLD == 50000.0
    assert config.GEMINI_TEMPERATURE == 0.2
    assert config.GEMINI_MODEL_NAME == "gemini-3.6-flash"
    assert config.EMBEDDING_MODEL_NAME == "gemini-embedding-001"


def test_config_summary_dictionary():
    """Verify get_summary returns dict with essential configuration fields."""
    summary = config.get_summary()
    assert isinstance(summary, dict)
    assert "gemini_model" in summary
    assert summary["gemini_model"] == "gemini-3.6-flash"
