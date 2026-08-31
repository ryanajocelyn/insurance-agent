"""
MLflow Observability & Tracer Manager Module.

Provides helper methods for initializing MLflow autologging, tracking graph execution spans,
and logging token metrics and final claim decisions.
"""

import mlflow
from typing import Dict, Any, Optional
from src.config import config


class MLflowTracer:
    """Manager class for initializing and managing MLflow experiment tracking."""

    _initialized: bool = False

    @classmethod
    def setup_mlflow(cls) -> None:
        """Initialize MLflow tracking URI and experiment setting once."""
        if not cls._initialized:
            try:
                mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
                mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
                cls._initialized = True
            except Exception as exc:
                print(f"[MLFLOW WARNING] Failed to setup MLflow tracking: {exc}")

    @classmethod
    def log_claim_adjudication_run(
        self,
        claim_state: Dict[str, Any],
        execution_time_seconds: float = 0.0,
    ) -> None:
        """
        Log claim adjudication execution metrics and final verdict to MLflow.

        Args:
            claim_state (Dict[str, Any]): Final ClaimState dict.
            execution_time_seconds (float): Total workflow execution latency.
        """
        self.setup_mlflow()
        try:
            with mlflow.start_run(run_name=f"Claim_{claim_state.get('claim_id', 'UNKNOWN')}"):
                mlflow.log_metric("claimed_amount", float(claim_state.get("claimed_amount", 0.0)))
                mlflow.log_metric("approved_amount", float(claim_state.get("approved_amount", 0.0)))
                mlflow.log_metric("execution_time_seconds", float(execution_time_seconds))
                mlflow.log_metric("frequency_risk_score", float(claim_state.get("frequency_risk_score", 0.0)))

                mlflow.log_param("claim_id", str(claim_state.get("claim_id", "")))
                mlflow.log_param("verdict", str(claim_state.get("adjudication_verdict", "")))
                mlflow.log_param("cross_modal_consistency", str(claim_state.get("cross_modal_consistency", True)))
                mlflow.log_param("gemini_model", config.GEMINI_MODEL_NAME)

        except Exception as exc:
            print(f"[MLFLOW WARNING] Failed to log run to MLflow: {exc}")
