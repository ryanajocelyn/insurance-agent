"""
Unit Tests for Policy Document Ingestion & File Filtering.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.rag.ingest_policies import ingest_policy_documents
from src.config import config


def test_ingest_policy_documents_file_list(tmp_path):
    """Verify that ingestion processes only the files listed in the file list file."""
    # Create mock PDF files
    docs_dir = tmp_path / "docs"
    guidelines_dir = docs_dir / "guidelines"
    guidelines_dir.mkdir(parents=True)

    pdf1 = guidelines_dir / "file1.pdf"
    pdf2 = guidelines_dir / "file2.pdf"
    pdf1.write_bytes(b"%PDF-1.4 mock content 1")
    pdf2.write_bytes(b"%PDF-1.4 mock content 2")

    # Create file_list containing only file1.pdf
    file_list_path = tmp_path / "ingestion_files.txt"
    file_list_path.write_text(f"docs/guidelines/{pdf1.name}\n", encoding="utf-8")

    mock_pages = [{"page_number": 1, "text": "This is a test policy page content for guidelines."}]

    with patch("src.rag.ingest_policies.extract_pdf_text_by_page", return_value=mock_pages), \
         patch("src.rag.ingest_policies.ChromaRepository") as mock_chroma_cls:
        
        mock_chroma_repo = MagicMock()
        mock_chroma_cls.return_value = mock_chroma_repo
        mock_chroma_repo.add_documents.return_value = ["id1"]

        with patch.object(config, "BASE_DIR", tmp_path):
            stats = ingest_policy_documents(
                docs_dir_path=str(docs_dir),
                file_list_path=str(file_list_path),
            )

        assert stats["total_pdfs"] == 1
        assert stats["total_chunks"] == 1
        mock_chroma_repo.add_documents.assert_called_once()
        call_kwargs = mock_chroma_repo.add_documents.call_args.kwargs
        assert call_kwargs["metadatas"][0]["source_file"] == "file1.pdf"


def test_ingest_policy_documents_all(tmp_path):
    """Verify that passing file_list_path=None processes all PDF files in subdirectories."""
    docs_dir = tmp_path / "docs"
    guidelines_dir = docs_dir / "guidelines"
    guidelines_dir.mkdir(parents=True)

    pdf1 = guidelines_dir / "file1.pdf"
    pdf2 = guidelines_dir / "file2.pdf"
    pdf1.write_bytes(b"%PDF-1.4 mock content 1")
    pdf2.write_bytes(b"%PDF-1.4 mock content 2")

    mock_pages = [{"page_number": 1, "text": "Sample text for chunking."}]

    with patch("src.rag.ingest_policies.extract_pdf_text_by_page", return_value=mock_pages), \
         patch("src.rag.ingest_policies.ChromaRepository") as mock_chroma_cls:
        
        mock_chroma_repo = MagicMock()
        mock_chroma_cls.return_value = mock_chroma_repo
        mock_chroma_repo.add_documents.return_value = ["id1", "id2"]

        stats = ingest_policy_documents(
            docs_dir_path=str(docs_dir),
            file_list_path=None,
        )

        assert stats["total_pdfs"] == 2
        assert stats["total_chunks"] == 2
