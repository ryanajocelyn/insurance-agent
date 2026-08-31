"""
Policy RAG Retriever Module.
"""

from typing import Dict, Any, List, Optional
from src.db.chroma_repository import ChromaRepository


class PolicyRetriever:
    """Retriever engine querying vector store for relevant IRDAI regulations and policy clauses."""

    def __init__(self, chroma_repo: Optional[ChromaRepository] = None):
        self.repo = chroma_repo if chroma_repo else ChromaRepository()

    def retrieve_relevant_clauses(
        self,
        query: str,
        n_results: int = 5,
        category_filter: Optional[str] = None,
        min_similarity_score: float = 0.40,
    ) -> List[Dict[str, Any]]:
        raw_matches = self.repo.similarity_search(
            query=query,
            n_results=n_results,
            category_filter=category_filter,
            collection_name="motor_policy_clauses",
        )

        filtered_clauses: List[Dict[str, Any]] = []
        for match in raw_matches:
            score = match.get("similarity_score", 1.0)
            if score >= min_similarity_score:
                filtered_clauses.append(
                    {
                        "clause_text": match["content"],
                        "source_file": match["metadata"].get("source_file", "IRDAI Policy Document"),
                        "document_title": match["metadata"].get("document_title", ""),
                        "category": match["metadata"].get("category", ""),
                        "page_number": match["metadata"].get("page_number", 1),
                        "similarity_score": score,
                    }
                )

        return filtered_clauses
