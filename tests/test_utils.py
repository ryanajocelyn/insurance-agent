"""
Unit Tests for Reusable Utility Helpers (Phase 2).
"""

import io
import json
import pytest
from PIL import Image
from src.core.state import ClaimState
from src.utils.image_utils import compress_and_downsample_image, encode_image_base64
from src.utils.file_utils import read_json_file, write_json_file, read_text_file


def test_claim_state_structure():
    """Verify ClaimState dictionary keys and structure."""
    state: ClaimState = {
        "claim_id": "CLM-1001",
        "policy_number": "POL-2024-88",
        "vehicle_details": {"make": "Maruti", "model": "Swift", "age_years": 3},
        "incident_narrative": "Rear bumper scratched during parking.",
        "uploaded_images": [],
        "estimate_line_items": [{"part_name": "Rear Bumper", "claimed_cost": 5000.0}],
        "customer_history": [],
        "held_policy_endorsements": ["ZERO_DEP"],
    }
    assert state["claim_id"] == "CLM-1001"
    assert state["held_policy_endorsements"] == ["ZERO_DEP"]


def test_image_downsampling_and_compression():
    """Verify image resolution downsampling to 1024x1024 and JPEG encoding."""
    img = Image.new("RGB", (2048, 2048), color="red")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")

    compressed_bytes, mime_type = compress_and_downsample_image(img_bytes.getvalue())
    assert mime_type == "image/jpeg"
    assert len(compressed_bytes) > 0

    result_img = Image.open(io.BytesIO(compressed_bytes))
    assert result_img.width <= 1024
    assert result_img.height <= 1024


def test_image_base64_encoding():
    """Verify Base64 encoding utility output format."""
    img = Image.new("RGB", (500, 500), color="blue")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")

    result = encode_image_base64(img_bytes.getvalue())
    assert result["mime_type"] == "image/jpeg"
    assert isinstance(result["data_b64"], str)
    assert len(result["data_b64"]) > 0


def test_file_utils_json_roundtrip(tmp_path):
    """Verify reading and writing JSON files."""
    test_data = {"test_key": "test_value", "items": [1, 2, 3]}
    json_path = tmp_path / "test.json"

    write_json_file(json_path, test_data)
    loaded_data = read_json_file(json_path)

    assert loaded_data == test_data
