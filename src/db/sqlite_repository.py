"""
SQLite Database Repository Implementation.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional
from src.config import config
from src.db.base_repository import BaseDatabaseRepository
from src.db.sql_executor import SqlExecutor


class SQLiteRepository(BaseDatabaseRepository):
    """SQLite implementation of the BaseDatabaseRepository interface."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path if db_path else config.SQLITE_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.sql_executor = SqlExecutor()
        self.initialize_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def initialize_schema(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS claims_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    claim_id TEXT UNIQUE NOT NULL,
                    customer_id TEXT NOT NULL,
                    policy_number TEXT NOT NULL,
                    claim_date TEXT NOT NULL,
                    claimed_amount REAL NOT NULL,
                    approved_amount REAL NOT NULL,
                    claim_type TEXT NOT NULL,
                    adjudication_verdict TEXT NOT NULL
                );
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS adjudication_audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    claim_id TEXT NOT NULL,
                    policy_number TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    claimed_amount REAL NOT NULL,
                    approved_amount REAL NOT NULL,
                    adjudication_verdict TEXT NOT NULL,
                    cross_modal_consistency INTEGER NOT NULL,
                    rationale TEXT NOT NULL,
                    deductions_json TEXT,
                    citations_json TEXT,
                    investigation_triggers_json TEXT
                );
                """
            )

            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_customer_date ON claims_history(customer_id, claim_date);"
            )
            conn.commit()

    def get_customer_history(self, customer_id: str, lookback_months: int = 6) -> List[Dict[str, Any]]:
        query, params = self.sql_executor.prepare_query(
            "get_customer_claims.sql",
            {"customer_id": customer_id, "lookback_months": lookback_months},
        )

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def save_adjudication_result(self, claim_record: Dict[str, Any]) -> str:
        params = {
            "claim_id": claim_record.get("claim_id", ""),
            "policy_number": claim_record.get("policy_number", ""),
            "claimed_amount": float(claim_record.get("claimed_amount", 0.0)),
            "approved_amount": float(claim_record.get("approved_amount", 0.0)),
            "adjudication_verdict": claim_record.get("adjudication_verdict", "ESCALATE"),
            "cross_modal_consistency": 1 if claim_record.get("cross_modal_consistency", True) else 0,
            "rationale": claim_record.get("adjudication_rationale", ""),
            "deductions_json": json.dumps(claim_record.get("deductions_breakdown", {})),
            "citations_json": json.dumps(claim_record.get("mandatory_citations", [])),
            "investigation_triggers_json": json.dumps(
                claim_record.get("investigation_triggers", [])
            ),
        }

        query, bind_params = self.sql_executor.prepare_query("save_adjudication.sql", params)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, bind_params)
            conn.commit()

        return str(params["claim_id"])

    def execute_template_query(self, template_name: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        query, bind_params = self.sql_executor.prepare_query(template_name, params)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, bind_params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
