"""
Abstract Vector Store Repository Interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BaseVectorStoreRepository(ABC):
    """Abstract Base Class defining the contract for Vector Store Repositories."""

    @abstractmethod
    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        collection_name: str = "motor_policy_clauses",
    ) -> List[str]:
        pass

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        n_results: int = 5,
        category_filter: Optional[str] = None,
        collection_name: str = "motor_policy_clauses",
    ) -> List[Dict[str, Any]]:
        pass
