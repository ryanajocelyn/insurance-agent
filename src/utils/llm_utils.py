"""
LLM Utility Module for Centralized Google Gemini 1.5 Pro Invocation.

This module provides reusable helper functions for calling Gemini 1.5 Pro with
structured JSON schema enforcement, exponential backoff retry handling, multimodal image payload support,
and graceful fallback handling.
"""

import time
import json
from typing import Type, TypeVar, List, Optional, Dict, Any
from pydantic import BaseModel
from google import genai
from google.genai import types
from src.config import config

T = TypeVar("T", bound=BaseModel)


def get_gemini_client() -> genai.Client:
    """Initialize and return the Google GenAI Client."""
    api_key = config.GOOGLE_API_KEY
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set in environment or config. Cannot initialize Gemini Client.")
    return genai.Client(api_key=api_key)


def _clean_schema_dict(schema: Any) -> Any:
    """Recursively remove 'additionalProperties' from Pydantic JSON schema for Gemini Developer API compatibility."""
    if isinstance(schema, dict):
        cleaned = {}
        for key, value in schema.items():
            if key == "additionalProperties":
                continue
            cleaned[key] = _clean_schema_dict(value)
        return cleaned
    elif isinstance(schema, list):
        return [_clean_schema_dict(item) for item in schema]
    return schema


def invoke_gemini_json(
    prompt: str,
    response_schema: Type[T],
    images: Optional[List[Dict[str, str]]] = None,
    temperature: Optional[float] = None,
    max_retries: int = 3,
) -> T:
    """Invoke Gemini 1.5 Pro with structured JSON schema enforcement."""
    temp = temperature if temperature is not None else config.GEMINI_TEMPERATURE
    client = get_gemini_client()

    contents: List[Any] = [prompt]
    if images:
        for img in images:
            contents.append(
                types.Part.from_bytes(
                    data=bytes(img["data_b64"], "utf-8") if isinstance(img["data_b64"], str) else img["data_b64"],
                    mime_type=img.get("mime_type", "image/jpeg"),
                )
            )

    cleaned_schema = _clean_schema_dict(response_schema.model_json_schema())

    gen_config = types.GenerateContentConfig(
        temperature=temp,
        response_mime_type="application/json",
        response_schema=cleaned_schema,
    )

    last_exception: Optional[Exception] = None
    backoff_seconds = 2.0

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=config.GEMINI_MODEL_NAME,
                contents=contents,
                config=gen_config,
            )

            if not response.text:
                raise ValueError("Empty response received from Gemini API.")

            parsed_data = response_schema.model_validate_json(response.text)
            return parsed_data

        except Exception as exc:
            last_exception = exc
            print(f"[GEMINI RETRY WARNING] Attempt {attempt}/{max_retries} failed: {exc}. Retrying in {backoff_seconds}s...")
            if attempt < max_retries:
                time.sleep(backoff_seconds)
                backoff_seconds *= 2.0

    raise RuntimeError(f"Failed to invoke Gemini 1.5 Pro after {max_retries} attempts. Last error: {last_exception}")
