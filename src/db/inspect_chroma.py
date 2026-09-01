"""
ChromaDB Inspection & Verification CLI Tool.

Run this script to check:
1. Total document chunk count in ChromaDB collection ('motor_policy_clauses').
2. List of all unique ingested source files & document titles.
3. Sample chunk content and metadata.
4. Raw query similarity search results with distance and similarity scores.
"""

import argparse
from typing import Optional
from src.db.chroma_repository import ChromaRepository
from src.rag.retriever import PolicyRetriever


def inspect_chroma(query: Optional[str] = None):
    repo = ChromaRepository()
    collection = repo._get_collection("motor_policy_clauses")
    
    count = collection.count()
    print("\n========================================================")
    print("        ChromaDB Vector Store Inspection Report")
    print("========================================================")
    print(f"Collection Name: 'motor_policy_clauses'")
    print(f"Total Indexed Chunks: {count}")

    if count == 0:
        print("\nWARNING: ChromaDB collection is EMPTY!")
        print("   Run ingestion to populate vector database:")
        print("   python -m src.rag.ingest_policies --all")
        print("========================================================\n")
        return

    # Retrieve all metadatas to show unique source files
    all_data = collection.get(include=["metadatas"])
    metadatas = all_data.get("metadatas", [])
    
    sources = set()
    categories = set()
    for meta in metadatas:
        if meta:
            src = meta.get("source_file") or meta.get("document_title") or "Unknown"
            cat = meta.get("category") or "general"
            sources.add(src)
            categories.add(cat)

    print("\nIngested Source Documents:")
    for src in sorted(sources):
        print(f"   - [DOC] {src}")

    print("\nIngested Categories:", ", ".join(sorted(categories)))

    # Run query test if provided
    if query:
        print(f"\nExecuting Test Query: '{query}'")
        raw_matches = repo.similarity_search(query=query, n_results=4)
        print(f"   Found {len(raw_matches)} raw match(es) from similarity_search:")
        
        for idx, match in enumerate(raw_matches, 1):
            score = match.get("similarity_score", 0.0)
            dist = match.get("distance", 0.0)
            meta = match.get("metadata", {})
            src = meta.get("source_file", "Unknown")
            page = meta.get("page_number", 1)
            content_preview = match.get("content", "")[:120].replace("\n", " ")
            
            print(f"\n   Match #{idx}:")
            print(f"     - Source: {src} (Page {page})")
            print(f"     - Similarity Score: {score:.4f} (Cosine Distance: {dist:.4f})")
            print(f"     - Preview: \"{content_preview}...\"")
            
        # Test PolicyRetriever
        retriever = PolicyRetriever(repo)
        filtered_default = retriever.retrieve_relevant_clauses(query=query, n_results=4, min_similarity_score=0.40)
        filtered_relaxed = retriever.retrieve_relevant_clauses(query=query, n_results=4, min_similarity_score=0.15)
        
        print(f"\nPolicyRetriever Results Comparison:")
        print(f"   - At min_similarity_score=0.40 (Default): {len(filtered_default)} clause(s) returned")
        print(f"   - At min_similarity_score=0.15 (Relaxed): {len(filtered_relaxed)} clause(s) returned")

    print("\n========================================================\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect ChromaDB vector store documents and query matches.")
    parser.add_argument("--query", type=str, default="collided with pillar damage to bumper", help="Test query string to search.")
    args = parser.parse_args()
    
    inspect_chroma(query=args.query)
