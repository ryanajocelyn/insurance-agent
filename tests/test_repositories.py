"""
Unit Tests for Database & Vector Store Repositories (Phase 3).
"""

import pytest
from src.db.sqlite_repository import SQLiteRepository
from src.db.chroma_repository import ChromaRepository
from src.utils.file_utils import read_json_file
from src.config import config


def test_sqlite_repository_crud(tmp_path):
    """Test SQLite schema initialization, record insertion, and customer history retrieval."""
    test_db_path = tmp_path / "test_claims.db"
    repo = SQLiteRepository(db_path=test_db_path)

    with repo._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO claims_history (
                claim_id, customer_id, policy_number, claim_date, claimed_amount, approved_amount, claim_type, adjudication_verdict
            ) VALUES (
                'CLM-001', 'CUST-100', 'POL-100', datetime('now'), 15000.0, 12000.0, 'OWN_DAMAGE', 'APPROVE_ADJUSTED'
            );
            """
        )
        conn.commit()

    history = repo.get_customer_history(customer_id="CUST-100", lookback_months=6)
    assert len(history) == 1
    assert history[0]["claim_id"] == "CLM-001"
    assert history[0]["claimed_amount"] == 15000.0


def test_sqlite_save_adjudication(tmp_path):
    """Test saving a synthesized adjudication decision into audit logs."""
    test_db_path = tmp_path / "test_claims.db"
    repo = SQLiteRepository(db_path=test_db_path)

    claim_record = {
        "claim_id": "CLM-002",
        "policy_number": "POL-200",
        "claimed_amount": 25000.0,
        "approved_amount": 22000.0,
        "adjudication_verdict": "APPROVE_ADJUSTED",
        "cross_modal_consistency": True,
        "adjudication_rationale": "Metal depreciation applied.",
        "deductions_breakdown": {"depreciation": 3000.0, "deductible": 1000.0},
        "mandatory_citations": ["IMT Section I"],
        "investigation_triggers": [],
    }

    claim_id = repo.save_adjudication_result(claim_record)
    assert claim_id == "CLM-002"


def test_chroma_repository_indexing_and_search(tmp_path):
    """Test indexing document chunks into ChromaDB and performing similarity search."""
    chroma_dir = tmp_path / "chroma_test"
    repo = ChromaRepository(chroma_path=chroma_dir)

    docs = [
        "Depreciation on metal parts is 15% for vehicles between 2 and 3 years old.",
        "Compulsory excess for engine capacity above 1500cc is Rupees 2000.",
    ]
    metas = [
        {"category": "guidelines", "source_file": "imt_rules.pdf"},
        {"category": "policy_forms", "source_file": "cis_policy.pdf"},
    ]

    added_ids = repo.add_documents(documents=docs, metadatas=metas, collection_name="test_collection")
    assert len(added_ids) == 2

    results = repo.similarity_search(
        query="metal depreciation rate for 3 year old vehicle",
        n_results=1,
        collection_name="test_collection",
    )
    assert len(results) == 1
    assert "metadata" in results[0]


def test_repair_benchmark_matrix():
    """Verify loading and structure of motor_repair_matrix.json benchmark file."""
    matrix = read_json_file(config.BENCHMARKS_PATH)
    assert "segments" in matrix
    assert "hatchback" in matrix["segments"]
    assert "front_bumper" in matrix["segments"]["hatchback"]
    assert "benchmark_median" in matrix["segments"]["hatchback"]["front_bumper"]
