"""
Model Context Protocol (MCP) External Tool Definitions.
"""

from typing import Dict, Any


def verify_vehicle_registration(reg_no: str) -> Dict[str, Any]:
    return {
        "reg_no": reg_no.upper(),
        "status": "ACTIVE",
        "make": "Maruti Suzuki",
        "model": "Swift VXi",
        "fuel_type": "PETROL",
        "cubic_capacity": 1197.0,
        "fitness_valid_until": "2030-05-15",
    }


def verify_police_fir(fir_number: str) -> Dict[str, Any]:
    return {
        "fir_number": fir_number.upper(),
        "is_verified": True,
        "police_station": "Central Traffic Station",
        "incident_date": "2026-08-28",
        "injury_reported": False,
    }


def fetch_oem_part_catalog(make: str, model: str, part_name: str) -> Dict[str, Any]:
    return {
        "make": make,
        "model": model,
        "part_name": part_name,
        "oem_list_price": 4500.0,
        "in_stock": True,
    }
