"""
Policy & Regulatory Document Ingestion CLI Script.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import config
from src.db.chroma_repository import ChromaRepository
from src.utils.file_utils import extract_pdf_text_by_page


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest regulatory PDF documents from docs/ into ChromaDB vector store."
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default=str(config.DOCS_DIR),
        help="Path to docs directory containing PDF subdirectories.",
    )
    parser.add_argument(
        "--file-list",
        type=str,
        default=str(config.BASE_DIR / "tests" / "ingestion_files.txt"),
        help="Path to text file containing specific list of PDF file paths to process.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ignore file list and process all PDF files in docs directory.",
    )
    return parser.parse_args()


def ingest_policy_documents(
    docs_dir_path: str = str(config.DOCS_DIR),
    file_list_path: Optional[str] = str(config.BASE_DIR / "tests" / "ingestion_files.txt"),
) -> Dict[str, int]:
    """
    Ingest regulatory PDF documents into ChromaDB vector store.

    Args:
        docs_dir_path (str): Path to root docs directory.
        file_list_path (Optional[str]): Optional path to a file containing a list of specific
            PDF file paths to process for testing/cost optimization. If None or non-existent,
            scans all PDFs under subdirectories in docs_dir_path.

    Returns:
        Dict[str, int]: Ingestion summary statistics (total_pdfs, total_chunks).
    """
    docs_root = Path(docs_dir_path)
    if not docs_root.exists():
        raise FileNotFoundError(f"Specified docs directory does not exist: {docs_root}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chroma_repo = ChromaRepository()

    stats: Dict[str, int] = {"total_pdfs": 0, "total_chunks": 0}

    # Determine files to process
    pdf_files_to_process: List[Path] = []

    if file_list_path:
        fl_path = Path(file_list_path)
        if not fl_path.is_absolute():
            fl_path = config.BASE_DIR / fl_path

        if fl_path.exists():
            print(f"--> Reading ingestion target files from '{fl_path}'...")
            with open(fl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str or line_str.startswith("#"):
                        continue
                    pdf_path = Path(line_str)
                    if not pdf_path.is_absolute():
                        pdf_path = config.BASE_DIR / pdf_path
                    if pdf_path.exists():
                        pdf_files_to_process.append(pdf_path)
                    else:
                        print(f"[INGEST WARNING] Target PDF file specified in list not found: {pdf_path}")

    if not pdf_files_to_process:
        if file_list_path and Path(file_list_path).exists():
            print("[INGEST INFO] No valid PDF files found in file list. Exiting ingestion.")
            return stats
        print(f"--> File list not used or empty. Scanning all PDFs in '{docs_root}' subdirectories...")
        subdirs = ["guidelines", "policy_forms", "exposure_draft", "rules"]
        for category in subdirs:
            category_dir = docs_root / category
            if not category_dir.exists():
                print(f"[INGEST WARNING] Directory '{category_dir}' not found. Skipping...")
                continue
            pdf_files_to_process.extend(category_dir.glob("*.pdf"))

    print(f"--> Processing {len(pdf_files_to_process)} PDF document(s)...")

    # Group files by category for structured ChromaDB batch insertion
    categorized_files: Dict[str, List[Path]] = {}
    for pdf_path in pdf_files_to_process:
        category = pdf_path.parent.name if pdf_path.parent != docs_root else "general"
        categorized_files.setdefault(category, []).append(pdf_path)

    for category, pdf_files in categorized_files.items():
        category_chunks: List[str] = []
        category_metadatas: List[Dict[str, Any]] = []
        category_ids: List[str] = []

        for pdf_path in pdf_files:
            stats["total_pdfs"] += 1
            doc_title = pdf_path.stem
            print(f"    - Ingesting PDF [{category}]: {pdf_path.name}")
            try:
                pages = extract_pdf_text_by_page(pdf_path)
            except Exception as exc:
                print(f"[INGEST ERROR] Failed to parse PDF '{pdf_path.name}': {exc}")
                continue

            for page_info in pages:
                page_num = page_info["page_number"]
                page_text = page_info["text"]
                if not page_text.strip():
                    continue

                chunks = text_splitter.split_text(page_text)
                clean_stem = pdf_path.stem.replace(" ", "_")
                for chunk_idx, chunk_text in enumerate(chunks):
                    chunk_id = f"{category}_{clean_stem}_p{page_num}_c{chunk_idx}"
                    metadata = {
                        "category": category,
                        "source_file": pdf_path.name,
                        "document_title": doc_title,
                        "page_number": page_num,
                        "chunk_index": chunk_idx,
                    }
                    category_chunks.append(chunk_text)
                    category_metadatas.append(metadata)
                    category_ids.append(chunk_id)

        if category_chunks:
            print(f"--> Indexing {len(category_chunks)} chunks for category '{category}' into ChromaDB...")
            chroma_repo.add_documents(
                documents=category_chunks,
                metadatas=category_metadatas,
                ids=category_ids,
                collection_name="motor_policy_clauses",
            )
            stats["total_chunks"] += len(category_chunks)

    print("\n========================================================")
    print(f"   PDF Ingestion Complete!")
    print(f"   Total PDFs Processed: {stats['total_pdfs']}")
    print(f"   Total Chunks Indexed: {stats['total_chunks']}")
    print("========================================================\n")

    return stats


if __name__ == "__main__":
    args = parse_args()
    file_list = None if args.all else args.file_list
    ingest_policy_documents(args.docs_dir, file_list_path=file_list)
