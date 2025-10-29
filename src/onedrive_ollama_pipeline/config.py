"""Configuration helpers for the OneDrive to Ollama pipeline."""
from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - fallback when package missing
    load_dotenv = None


if load_dotenv:
    load_dotenv()

from .prompt_loader import load_prompt

DEFAULT_METADATA_PROMPT = load_prompt("metadata_system_prompt.txt")


CONFIG_DEFAULTS: Dict[str, Any] = {
    "graph": {
        "inbox_folder": "_inbox",
        "sorted_folder": "_sorted",
    },
    "ollama": {
        "model": "llama3",
        "metadata_prompt": DEFAULT_METADATA_PROMPT,
    },
    "pipeline": {
        "poll_interval_seconds": None,
        "log_level": "INFO",
        "auto_process_inbox": False,
    },
    "structure": {
        "model": "",
        "language": "auto",
    },
}


@dataclass
class GraphSettings:
    tenant_id: str
    client_id: str
    client_secret: Optional[str]
    user_id: str
    inbox_folder: str = "_inbox"
    sorted_folder: str = "_sorted"
    authority: Optional[str] = None
    scopes: tuple[str, ...] = ("Files.ReadWrite.All",)
    token_cache_path: Path = Path("~/.onedrive_ollama_token_cache.json")


@dataclass
class OllamaSettings:
    base_url: str = "http://localhost:11434/v1"
    model: str = "llama3"
    metadata_prompt: str = DEFAULT_METADATA_PROMPT


@dataclass
class PipelineSettings:
    graph: GraphSettings
    ollama: OllamaSettings
    db_path: Path
    log_path: Path
    config_path: Path
    log_level: str = "INFO"
    poll_interval_seconds: Optional[int] = None
    auto_process_inbox: bool = False
    structure_model: str = "llama3"
    structure_language: str = "auto"
    env_file: Path = Path(".env")


def _get_env(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    value = os.getenv(name, default)
    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _deep_update(target: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def load_config_file(path: Path) -> Dict[str, Any]:
    """Load pipeline configuration from a JSON file, applying defaults."""
    path = Path(path).expanduser()
    if not path.exists():
        save_config_file(path, CONFIG_DEFAULTS)
        return deepcopy(CONFIG_DEFAULTS)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):  # pragma: no cover - corrupt file
        data = {}

    merged = deepcopy(CONFIG_DEFAULTS)
    _deep_update(merged, data)
    return merged


def save_config_file(path: Path, data: Dict[str, Any]) -> None:
    """Persist pipeline configuration to disk."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def load_settings() -> PipelineSettings:
    """Load settings from environment variables."""
    default_env_path = Path(_get_env("PIPELINE_ENV_PATH", ".env"))
    config_path = Path(_get_env("PIPELINE_CONFIG_PATH", "config.json"))
    config_data = load_config_file(config_path)

    db_path = Path(_get_env("PIPELINE_DB_PATH", os.path.expanduser("~/.onedrive_ollama_pipeline.sqlite3")))
    log_path = Path(_get_env("PIPELINE_LOG_PATH", os.path.expanduser("~/.onedrive_ollama_pipeline.log")))

    graph = GraphSettings(
        tenant_id=_get_env("GRAPH_TENANT_ID", required=True),
        client_id=_get_env("GRAPH_CLIENT_ID", required=True),
        client_secret=_get_env("GRAPH_CLIENT_SECRET"),
        user_id=_get_env("GRAPH_USER_ID", required=True),
        inbox_folder=config_data["graph"].get("inbox_folder", CONFIG_DEFAULTS["graph"]["inbox_folder"]),
        sorted_folder=config_data["graph"].get("sorted_folder", CONFIG_DEFAULTS["graph"]["sorted_folder"]),
        authority=_get_env("GRAPH_AUTHORITY_URL"),
        token_cache_path=Path(_get_env("GRAPH_TOKEN_CACHE_PATH", "~/.onedrive_ollama_token_cache.json")).expanduser(),
    )

    ollama = OllamaSettings(
        base_url=_get_env("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        model=config_data["ollama"].get("model", CONFIG_DEFAULTS["ollama"]["model"]),
        metadata_prompt=config_data["ollama"].get("metadata_prompt", CONFIG_DEFAULTS["ollama"]["metadata_prompt"]),
    )

    pipeline_section = config_data.get("pipeline", {})
    poll_interval = pipeline_section.get("poll_interval_seconds")
    log_level = pipeline_section.get("log_level") or "INFO"
    auto_process = bool(pipeline_section.get("auto_process_inbox", False))

    structure_section = config_data.get("structure", {})
    structure_model = structure_section.get("model")
    if not structure_model:
        structure_model = config_data["ollama"].get("model", CONFIG_DEFAULTS["ollama"]["model"])
    structure_language = structure_section.get("language", CONFIG_DEFAULTS["structure"].get("language", "auto"))

    return PipelineSettings(
        graph=graph,
        ollama=ollama,
        db_path=db_path,
        log_path=log_path,
        config_path=config_path,
        log_level=log_level.upper(),
        poll_interval_seconds=poll_interval,
        auto_process_inbox=auto_process,
        structure_model=structure_model,
        structure_language=structure_language,
        env_file=default_env_path,
    )
