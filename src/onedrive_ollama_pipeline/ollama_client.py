"""Client for interacting with the Ollama OpenAI-compatible API."""
from __future__ import annotations

import base64
import json
import logging
import os
from typing import Optional, Sequence, Mapping, Any

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except ImportError:  # pragma: no cover - fallback message at runtime
    OpenAI = None  # type: ignore[assignment]

from .config import OllamaSettings
from .prompt_loader import load_prompt

_LOGGER = logging.getLogger(__name__)

METADATA_USER_PROMPT = load_prompt("metadata_user_prompt.txt")
STRUCTURE_SYSTEM_PROMPT = load_prompt("structure_system_prompt.txt")
TEST_MODEL_SYSTEM_PROMPT = load_prompt("test_model_system_prompt.txt")
TEST_MODEL_USER_PROMPT = load_prompt("test_model_user_prompt.txt")
QUICK_TEST_PROMPT = load_prompt("quick_test_prompt.txt")


class OllamaClient:
    """Very small helper around the Ollama OpenAI-compatible REST API."""

    def __init__(self, settings: OllamaSettings, client: Optional["OpenAI"] = None):
        self._settings = settings
        if client is not None:
            self._client = client
        else:
            if OpenAI is None:  # pragma: no cover - import guard
                raise ImportError(
                    "The 'openai' package is required to use OllamaClient. "
                    "Install it with `pip install openai`."
                )
            self._client = OpenAI(
                base_url=settings.base_url.rstrip("/"),
                api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
            )

    def _chat_completion(
        self,
        messages: Sequence[dict],
        *,
        model: Optional[str],
        max_tokens: int,
        temperature: float,
        timeout: int,
        format_hint: Optional[str] = None,
        response_format: Optional[Mapping[str, Any]] = None,
    ) -> dict:
        request_format: Optional[Mapping[str, Any]] = None
        if response_format is not None:
            request_format = response_format
        elif format_hint == "json":
            request_format = {"type": "json_object"}
        try:
            completion = self._client.chat.completions.create(
                model=model or self._settings.model,
                messages=list(messages),
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=request_format,
                timeout=timeout,
            )
        except Exception as exc:  # pragma: no cover - network failures
            raise RuntimeError(f"Ollama chat completion failed: {exc}") from exc

        if hasattr(completion, "model_dump"):
            return completion.model_dump()
        if hasattr(completion, "to_dict"):
            return completion.to_dict()
        return completion  # Best effort fallback

    def _image_to_data_url(self, image_bytes: bytes) -> str:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def request_metadata(self, *, image_bytes: Optional[bytes] = None, text: Optional[str] = None) -> dict:
        """Send PDF-derived content to the model and return JSON metadata."""
        normalized_text = text.strip() if isinstance(text, str) else None
        if image_bytes is None and not normalized_text:
            raise ValueError("Provide 'image_bytes', 'text', or both.")

        content: list[dict[str, Any]] = [
            {"type": "text", "text": METADATA_USER_PROMPT},
        ]
        if normalized_text:
            content.append({"type": "text", "text": normalized_text})
        if image_bytes is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_to_data_url(image_bytes)},
                }
            )

        payload = {
            "role": "user",
            "content": content,
        }
        messages = [
            {"role": "system", "content": self._settings.metadata_prompt},
            payload,
        ]
        payload_preview = json.dumps({"messages": messages}, ensure_ascii=False)
        approx_tokens = max(1, (len(payload_preview) + 3) // 4)
        input_tags = []
        if normalized_text:
            input_tags.append("text")
        if image_bytes is not None:
            input_tags.append("image")
        descriptor = "+".join(input_tags) if input_tags else "unknown"
        _LOGGER.info(
            "Metadata prompt size (%s): %d chars (≈%d tokens)",
            descriptor,
            len(payload_preview),
            approx_tokens,
        )
        data = self._chat_completion(messages, model=None, max_tokens=512, temperature=0.2, timeout=120)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Unexpected Ollama response: {json.dumps(data)}") from exc
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```", 2)[1]
        try:
            metadata = json.loads(content)
        except json.JSONDecodeError as exc:
            _LOGGER.error("Failed to parse metadata JSON: %s", content)
            raise RuntimeError("Ollama returned non-JSON output") from exc
        return metadata

    def request_structure_plan(
        self,
        prompt: str,
        *,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        json_schema: Optional[dict] = None,
        schema_name: str = "StructurePlan",
        system_prompt: Optional[str] = None,
    ) -> str:
        """Request a JSON-only structure plan from the Ollama server."""
        return self.request_json_schema_completion(
            system_prompt=system_prompt or STRUCTURE_SYSTEM_PROMPT,
            prompt=prompt,
            model=model,
            json_schema=json_schema,
            schema_name=schema_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def request_json_schema_completion(
        self,
        *,
        system_prompt: str,
        prompt: str,
        model: str,
        json_schema: Optional[dict],
        schema_name: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        timeout: int = 180,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": prompt},
        ]
        data = self._chat_completion(
            messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            format_hint=None if json_schema else "json",
            response_format=(
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name or "StructureJson",
                        "schema": json_schema,
                    },
                }
                if json_schema
                else None
            ),
        )
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Unexpected Ollama response: {json.dumps(data)}") from exc
        if not isinstance(content, str):
            raise RuntimeError("Ollama returned non-string content for structured response.")
        content = content.strip()
        if content.startswith("```"):
            segments = content.split("```")
            if len(segments) >= 3:
                content = segments[1]
        return content

    def ensure_ready(self) -> None:
        """Perform a lightweight sanity check against the default model."""
        self.test_model(
            model=self._settings.model,
            prompt=QUICK_TEST_PROMPT,
            expected="test",
            timeout=30,
        )
        _LOGGER.debug("Ollama quick-test succeeded")

    def test_model(
        self,
        *,
        model: str,
        prompt: Optional[str] = None,
        expected: str = "success",
        timeout: int = 60,
    ) -> None:
        user_prompt = prompt or TEST_MODEL_USER_PROMPT
        messages = [
            {
                "role": "system",
                "content": TEST_MODEL_SYSTEM_PROMPT,
            },
            {"role": "user", "content": user_prompt},
        ]
        data = self._chat_completion(
            messages,
            model=model,
            max_tokens=32,
            temperature=0.0,
            timeout=timeout,
        )
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Unexpected Ollama response: {json.dumps(data)}") from exc
        if not isinstance(content, str):
            raise RuntimeError("Ollama returned non-string content for sanity check.")
        if expected and content.strip().lower() != expected.lower():
            raise RuntimeError(f"Unexpected Ollama reply: {content}")
