"""
Model Context Protocol (MCP) Server Integration Module.
"""

from typing import Dict, Any, List
from src.mcp.tools import verify_vehicle_registration, verify_police_fir, fetch_oem_part_catalog


class MCPServer:
    """Server handler exposing standardized MCP tool definitions."""

    def __init__(self):
        self.tools = {
            "verify_vehicle_registration": verify_vehicle_registration,
            "verify_police_fir": verify_police_fir,
            "fetch_oem_part_catalog": fetch_oem_part_catalog,
        }

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "verify_vehicle_registration",
                "description": "Verify vehicle specs from state registration transport database.",
                "parameters": {"type": "object", "properties": {"reg_no": {"type": "string"}}},
            },
            {
                "name": "verify_police_fir",
                "description": "Verify authenticity of filed Police FIR document.",
                "parameters": {"type": "object", "properties": {"fir_number": {"type": "string"}}},
            },
            {
                "name": "fetch_oem_part_catalog",
                "description": "Fetch OEM list price for replacement spare parts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "make": {"type": "string"},
                        "model": {"type": "string"},
                        "part_name": {"type": "string"},
                    },
                },
            },
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' not registered in MCP Server.")
        tool_fn = self.tools[name]
        return tool_fn(**arguments)
