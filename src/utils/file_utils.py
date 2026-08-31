"""
File I/O and Document Parsing Utility Module.

Provides reusable methods for reading and writing local text files, JSON benchmark matrices,
PDF documents, and safe file path handling across the ingestion and evaluation pipelines.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Union
from pypdf import PdfReader


def read_json_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Read and parse a local JSON dataset or configuration file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found at path: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def write_json_file(file_path: Union[str, Path], data: Dict[str, Any], indent: int = 2) -> None:
    """Write a dictionary dataset to a local JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def read_text_file(file_path: Union[str, Path]) -> str:
    """Read text content from a plain text or markdown file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Text file not found at path: {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return content


def extract_pdf_text_by_page(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Extract text content from a PDF document page-by-page."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found at path: {path}")

    reader = PdfReader(str(path))
    pages_output: List[Dict[str, Any]] = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages_output.append(
            {
                "page_number": i + 1,
                "text": text.strip(),
            }
        )

    return pages_output
