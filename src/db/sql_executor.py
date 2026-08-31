"""
JinjaSql Engine Utility Module.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from jinjasql import JinjaSql
from src.config import config


class SqlExecutor:
    """Utility class for loading, preparing, and rendering JinjaSql queries."""

    def __init__(self, templates_dir: Optional[Path] = None):
        self.templates_dir = templates_dir if templates_dir else config.SQL_TEMPLATES_DIR
        self.j2sql = JinjaSql(param_style="qmark")

    def load_template(self, template_name: str) -> str:
        template_path = self.templates_dir / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"SQL template file not found at path: {template_path}")

        with open(template_path, "r", encoding="utf-8") as f:
            template_str = f.read()
        return template_str

    def prepare_query(self, template_name: str, params: Dict[str, Any]) -> Tuple[str, List[Any]]:
        template_str = self.load_template(template_name)
        query, bind_params = self.j2sql.prepare_query(template_str, params)
        return query, bind_params
