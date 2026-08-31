"""
Abstract Base Repository Interface for Relational Database Persistence.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BaseDatabaseRepository(ABC):
    """Abstract Base Class defining the contract for Relational Database Repositories."""

    @abstractmethod
    def initialize_schema(self) -> None:
        pass

    @abstractmethod
    def get_customer_history(self, customer_id: str, lookback_months: int = 6) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def save_adjudication_result(self, claim_record: Dict[str, Any]) -> str:
        pass

    @abstractmethod
    def execute_template_query(self, template_name: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        pass
