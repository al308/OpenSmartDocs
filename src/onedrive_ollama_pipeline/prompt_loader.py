"""Utility helpers for loading reusable prompt snippets."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


@lru_cache(maxsize=None)
def load_prompt(name: str, *, strip: bool = True) -> str:
    """Load a prompt file from the embedded prompts directory."""
    path = _PROMPT_DIR / name
    if not path.exists():  # pragma: no cover - defensive guard
        raise FileNotFoundError(f"Prompt file '{name}' not found in {_PROMPT_DIR}.")
    text = path.read_text(encoding="utf-8")
    return text.strip() if strip else text
