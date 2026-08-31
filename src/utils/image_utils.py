"""
Image Preprocessing Utility Module.

Provides functions for downsampling high-resolution vehicle damage photographs,
compressing image payloads to manage latency, and encoding images to Base64 format
for transmission to Gemini 1.5 Pro multimodal vision API.
"""

import io
import base64
from typing import Tuple, Dict, Any, Union
from PIL import Image


def compress_and_downsample_image(
    image_input: Union[str, bytes, Image.Image],
    max_dimensions: Tuple[int, int] = (1024, 1024),
    quality: int = 85,
) -> Tuple[bytes, str]:
    """Downsample image resolution and compress to JPEG format."""
    if isinstance(image_input, str):
        img = Image.open(image_input)
    elif isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise ValueError("Unsupported image_input type. Must be file path, bytes, or PIL Image.")

    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")

    img.thumbnail(max_dimensions, Image.Resampling.LANCZOS)

    output_buffer = io.BytesIO()
    img.save(output_buffer, format="JPEG", quality=quality, optimize=True)
    compressed_bytes = output_buffer.getvalue()

    return compressed_bytes, "image/jpeg"


def encode_image_base64(
    image_input: Union[str, bytes, Image.Image],
    max_dimensions: Tuple[int, int] = (1024, 1024),
    quality: int = 85,
) -> Dict[str, str]:
    """Preprocess image and convert to Base64 payload dictionary."""
    compressed_bytes, mime_type = compress_and_downsample_image(
        image_input=image_input,
        max_dimensions=max_dimensions,
        quality=quality,
    )
    b64_str = base64.b64encode(compressed_bytes).decode("utf-8")

    return {
        "mime_type": mime_type,
        "data_b64": b64_str,
    }
