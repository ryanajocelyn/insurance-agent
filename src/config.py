"""
Global Configuration Module for Multi-Agent Motor Claim Adjudication Engine.

This module provides a thread-safe Singleton `Config` class that manages
environment settings, LLM parameters, database connection paths, threshold limits,
and operational parameters across all agent nodes and execution pipelines.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Ensure src and project root are in sys.path
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file if present
load_dotenv(PROJECT_ROOT / ".env")


class Config:
    """Singleton Configuration class storing runtime parameters and defaults."""

    _instance: Optional["Config"] = None

    def __new__(cls) -> "Config":
        """Ensure singleton pattern instance creation."""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize configuration properties and environment defaults."""
        # Project Root Path
        self.BASE_DIR: Path = PROJECT_ROOT
        self.SRC_DIR: Path = SRC_DIR

        # Google Gemini API & Model Settings
        self.GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
        self.GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-3.6-flash")
        self.GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
        self.EMBEDDING_MODEL_NAME: str = os.getenv(
            "EMBEDDING_MODEL_NAME", "gemini-embedding-001"
        )

        # Database Storage Paths
        self.DATA_DIR: Path = self.BASE_DIR / "data"
        self.SQLITE_DB_PATH: Path = self.DATA_DIR / "motor_claims.db"
        self.CHROMA_DB_PATH: Path = self.DATA_DIR / "chroma_db"
        self.SQL_TEMPLATES_DIR: Path = self.DATA_DIR / "sql_templates"

        # Benchmarks & Documents Source Directory
        self.DOCS_DIR: Path = self.BASE_DIR / "docs"
        self.DOCS_ARCHITECTURE_DIR: Path = self.DOCS_DIR / "architecture"
        self.BENCHMARKS_PATH: Path = (
            self.DATA_DIR / "benchmarks" / "motor_repair_matrix.json"
        )

        # Adjudication & Fraud Threshold Settings
        self.COST_ADJUSTMENT_THRESHOLD: float = float(
            os.getenv("COST_ADJUSTMENT_THRESHOLD", "0.30")
        )  # +30% benchmark variance triggers APPROVE_ADJUSTED clamp
        self.FRAUD_ESCALATION_THRESHOLD: float = float(
            os.getenv("FRAUD_ESCALATION_THRESHOLD", "1.00")
        )  # > +100% benchmark variance triggers ESCALATE
        self.CLAIM_VELOCITY_LOOKBACK_MONTHS: int = int(
            os.getenv("CLAIM_VELOCITY_LOOKBACK_MONTHS", "6")
        )  # 6-month claim velocity window
        self.FIR_REQUIRED_LOSS_THRESHOLD: float = float(
            os.getenv("FIR_REQUIRED_LOSS_THRESHOLD", "50000.0")
        )  # Loss > ₹50,000 mandates FIR verification

        # MLflow & Observability Settings
        self.MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
        self.MLFLOW_EXPERIMENT_NAME: str = os.getenv(
            "MLFLOW_EXPERIMENT_NAME", "Motor_Claim_Adjudication_Engine"
        )

    def validate(self) -> bool:
        """
        Validate critical environment parameters.

        Returns:
            bool: True if configuration is valid, False otherwise.
        """
        if not self.GOOGLE_API_KEY:
            print("[CONFIG WARNING] GOOGLE_API_KEY environment variable is missing.")
            return False
        return True

    def get_summary(self) -> Dict[str, Any]:
        """
        Return a summary dictionary of active configuration settings.

        Returns:
            Dict[str, Any]: Configuration properties summary.
        """
        return {
            "gemini_model": self.GEMINI_MODEL_NAME,
            "gemini_temperature": self.GEMINI_TEMPERATURE,
            "embedding_model": self.EMBEDDING_MODEL_NAME,
            "cost_adjustment_threshold": f"{self.COST_ADJUSTMENT_THRESHOLD * 100:.0f}%",
            "fraud_escalation_threshold": f">{self.FRAUD_ESCALATION_THRESHOLD * 100:.0f}%",
            "claim_velocity_lookback": f"{self.CLAIM_VELOCITY_LOOKBACK_MONTHS} months",
            "fir_loss_threshold": f"₹{self.FIR_REQUIRED_LOSS_THRESHOLD:,.0f}",
            "sqlite_db_path": str(self.SQLITE_DB_PATH),
            "chroma_db_path": str(self.CHROMA_DB_PATH),
        }


# Global singleton configuration accessor
config = Config()
