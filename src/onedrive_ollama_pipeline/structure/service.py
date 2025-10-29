"""High-level orchestration for structure proposals."""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import PipelineSettings, load_settings
from ..database import Database, get_database
from ..ollama_client import OllamaClient
from ..onedrive_client import OneDriveClient, SortedTreeEntry
from . import constants
from .models import (
    StructureCache,
    StructureContext,
    StructureServiceError,
    StructureSource,
    STRUCTURE_PLAN_JSON_SCHEMA,
    utc_now_iso,
)
from .plan_engine import PlanEngine
from .prompt_builder import build_prompt

LOGGER = logging.getLogger(__name__)


class StructureService:
    """Orchestrate structure recommendations via the Ollama-backed LLM."""

    def __init__(
        self,
        *,
        settings: Optional[PipelineSettings] = None,
        cache: Optional[StructureCache] = None,
        onedrive_client: Optional[OneDriveClient] = None,
        ollama_client: Optional[OllamaClient] = None,
    ):
        self._settings = settings or load_settings()
        self._db: Database = get_database(self._settings.db_path)
        self._onedrive = onedrive_client or OneDriveClient(self._settings.graph)
        self._ollama = ollama_client or OllamaClient(self._settings.ollama)
        self._cache = cache or StructureCache.default()
        model = getattr(self._settings, "structure_model", "") or ""
        self._model = model if model else self._settings.ollama.model
        self._structure_locale = getattr(self._settings, "structure_language", "auto") or "auto"
        self._plan_engine = PlanEngine(
            ollama=self._ollama,
            model_name=self._model,
            cache=self._cache,
            endpoint=self._settings.ollama.base_url,
        )
        self._done_folder_ready = False

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def get_state(self) -> Dict[str, Any]:
        state = self._cache.read()
        return {
            "plan": state.get("plan"),
            "applied": state.get("applied"),
            "log": state.get("log", []),
        }

    def analyze(self) -> Dict[str, Any]:
        try:
            context = self._collect_context()
        except Exception as exc:
            message = f"Failed to read sorted folder: {exc}"
            LOGGER.exception("Structure context collection failed: %s", exc)
            self._cache.append_log(message, level="error")
            raise StructureServiceError(message) from exc

        if not context.sources:
            message = "No eligible files found in the sorted root; skipping analysis."
            LOGGER.info(message)
            self._cache.append_log(message)
            plan = {
                "plan_id": str(uuid.uuid4()),
                "generated_at": utc_now_iso(),
                "model": self._model,
                "summary": message,
                "operations": [],
                "context": context.snapshot,
            }
            self._cache.store_plan(plan)
            return plan

        prompt_payload = build_prompt(context)
        LOGGER.debug("Structure prompt (truncated): %s", prompt_payload[:2000])
        self._cache.append_log(f"Structure prompt preview (model {self._model}): {prompt_payload[:200]}…", level="info")

        try:
            raw_response = self._ollama.request_structure_plan(
                prompt_payload,
                model=self._model,
                json_schema=STRUCTURE_PLAN_JSON_SCHEMA,
                schema_name="StructurePlan",
            )
            plan = self._plan_engine.generate_plan(prompt_payload, raw_response, context)
        except StructureServiceError:
            raise
        except Exception as exc:
            message = f"Failed to interpret structure plan after repair attempts: {exc}"
            LOGGER.exception(message)
            self._cache.append_log(message, level="error")
            raise StructureServiceError(message) from exc

        validation_status = (plan.get("validation", {}) or {}).get("status")
        if validation_status in {"warn", "error"}:
            summary_note = plan.get("validation", {}).get("notes")
            if summary_note and "Sanity check:" not in plan["summary"]:
                plan["summary"] += f" (Sanity check: {summary_note})"

        self._cache.store_plan(plan)
        self._cache.append_log(f"Generated structure plan with {len(plan['operations'])} operation(s).")
        return plan

    def apply(self) -> Dict[str, Any]:
        state = self._cache.read()
        plan = state.get("plan")
        if not plan:
            raise ValueError("No structure plan has been generated yet.")
        operations = plan.get("operations") or []
        if not operations:
            message = "Structure plan contains no operations; nothing to apply."
            LOGGER.info(message)
            self._cache.append_log(message)
            return {"applied": [], "created_files": [], "created_folders": []}

        sources_index = {source["id"]: source for source in plan["context"]["sources"]}
        created_files: list[str] = []
        created_folders: list[str] = []
        applied_ops: list[Dict[str, Any]] = []
        existing_folders = set(plan["context"].get("existing_folders", []))
        executed_folder_paths: set[str] = set()
        created_folder_paths: set[str] = set()
        pending_folder_ops: list[Dict[str, Any]] = []
        processed_sources: set[str] = set()

        def _execute_pending_folders() -> None:
            nonlocal pending_folder_ops
            if not pending_folder_ops:
                return
            items = pending_folder_ops
            pending_folder_ops = []
            for folder_entry in items:
                detail = self._normalized_relative(folder_entry.get("path"))
                if not detail or detail in executed_folder_paths:
                    continue
                try:
                    self._onedrive.ensure_sorted_subfolder(detail)
                    executed_folder_paths.add(detail)
                    applied_ops.append(folder_entry)
                    if detail not in existing_folders and detail not in created_folder_paths:
                        created_folders.append(detail)
                        created_folder_paths.add(detail)
                    self._cache.append_log(f"Ensured folder '{detail}' exists.")
                except Exception as exc:  # pragma: no cover - network failure
                    LOGGER.exception("Failed to ensure folder '%s': %s", detail, exc)
                    self._cache.append_log(f"Failed to create folder '{detail}': {exc}", level="error")

        def _record_copy(entry: Dict[str, Any], target_folder: str, target_name: str, source_info: Dict[str, Any]) -> None:
            applied_ops.append(entry)
            destination_path = "/".join(part for part in [target_folder, target_name] if part)
            if destination_path not in created_files:
                created_files.append(destination_path)
            self._cache.append_log(f"Copied '{source_info['name']}' to '{destination_path}'.")

        for entry in operations:
            action = entry["action"]
            if action == "create_folder":
                pending_folder_ops.append(entry)
            elif action == "copy_file":
                source_id = entry["source_id"]
                source_info = sources_index.get(source_id)
                if not source_info:
                    raise ValueError(f"Unknown source id '{source_id}' in plan.")
                if source_id in processed_sources:
                    self._cache.append_log(
                        f"Skipping duplicate copy operation for '{source_info['name']}' (already processed).",
                        level="warn",
                    )
                    continue
                _execute_pending_folders()
                target_folder = self._normalized_relative(entry.get("target_folder", ""))
                target_name = entry["target_name"]
                try:
                    _, content = self._onedrive.download_sorted_file(source_info["relative_path"])
                    self._onedrive.upload_bytes_to_sorted(target_name, content, folder_path=target_folder)
                    _record_copy(entry, target_folder, target_name, source_info)
                    self._move_source_to_done(source_info)
                    processed_sources.add(source_id)
                except Exception as exc:  # pragma: no cover - network failure
                    LOGGER.exception("Failed to copy '%s' to '%s': %s", source_info["relative_path"], target_folder, exc)
                    self._cache.append_log(f"Failed to copy '{source_info['name']}': {exc}", level="error")
                    continue
            else:
                raise ValueError(f"Unsupported action '{action}' in plan.")

        _execute_pending_folders()

        applied_state = {
            "plan_id": plan["plan_id"],
            "applied_at": utc_now_iso(),
            "created_files": created_files,
            "created_folders": created_folders,
            "operations": applied_ops,
        }
        self._cache.store_applied(applied_state)
        LOGGER.info("Applied %d structure operations.", len(applied_ops))
        return applied_state

    def revert(self) -> Dict[str, Any]:
        state = self._cache.read()
        applied = state.get("applied")
        if not applied:
            message = "No applied structure changes recorded; revert skipped."
            LOGGER.info(message)
            self._cache.append_log(message)
            return {"removed_files": [], "removed_folders": []}

        removed_files: list[str] = []
        removed_folders: list[str] = []

        for path in applied.get("created_files", []):
            try:
                self._onedrive.delete_sorted_path(path)
                removed_files.append(path)
                self._cache.append_log(f"Deleted copied file '{path}'.")
            except Exception as exc:  # pragma: no cover - network failure
                LOGGER.exception("Failed to delete file '%s': %s", path, exc)
                self._cache.append_log(f"Failed to delete '{path}': {exc}", level="error")
                raise StructureServiceError(f"Failed to delete '{path}': {exc}") from exc

        for folder in sorted(applied.get("created_folders", []), key=lambda value: value.count("/"), reverse=True):
            try:
                self._onedrive.delete_sorted_path(folder)
                removed_folders.append(folder)
                self._cache.append_log(f"Deleted folder '{folder}'.")
            except Exception as exc:  # pragma: no cover - network failure
                LOGGER.exception("Failed to delete folder '%s': %s", folder, exc)
                self._cache.append_log(f"Failed to delete folder '{folder}': {exc}", level="error")
                raise StructureServiceError(f"Failed to delete folder '{folder}': {exc}") from exc

        cache_state = self._cache.read()
        cache_state.pop("applied", None)
        self._cache.write(cache_state)
        self._cache.append_log("Reverted applied structure changes.")

        return {"removed_files": removed_files, "removed_folders": removed_folders}

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _collect_context(self) -> StructureContext:
        metadata_lookup = self._load_metadata()
        entries = list(self._onedrive.walk_sorted_tree())
        sources: list[StructureSource] = []
        existing_folders: set[str] = set()
        folder_examples: Dict[str, list[str]] = {}

        for entry in entries:
            if entry.is_folder:
                existing_folders.add(entry.path)
                folder_examples.setdefault(entry.path or "/", [])
                continue

            folder_path = self._folder_from_path(entry.path)
            folder_examples.setdefault(folder_path or "/", [])
            if len(folder_examples[folder_path or "/"]) < 3:
                folder_examples[folder_path or "/"].append(entry.name)

            if "/" in entry.path:
                continue  # Only root-level files considered for now

            source_id = f"SRC{len(sources) + 1:03d}"
            metadata = metadata_lookup.get(entry.name) or metadata_lookup.get(entry.name.lower()) or {}
            sources.append(
                StructureSource(
                    source_id=source_id,
                    relative_path=entry.path,
                    name=entry.name,
                    metadata=metadata if isinstance(metadata, dict) else {},
                    locale=self._structure_locale,
                )
            )

        return StructureContext(
            sources=sources[: constants.MAX_SOURCES_IN_PROMPT],
            existing_folders=existing_folders,
            folder_examples=folder_examples,
        )

    def _load_metadata(self) -> Dict[str, Any]:
        rows = self._db.query(
            "SELECT filename, metadata_json FROM processed_items WHERE status = 'success'"
        )
        data: Dict[str, Any] = {}
        for row in rows:
            metadata = Database.decode_json(row["metadata_json"])
            if metadata is None:
                continue
            data[row["filename"]] = metadata
            data[row["filename"].lower()] = metadata
        return data

    @staticmethod
    def _folder_from_path(path: str) -> str:
        normalized = Path(path)
        parent = normalized.parent.as_posix()
        return "" if parent == "." else parent

    @staticmethod
    def _normalized_relative(value: Optional[str]) -> str:
        raw = (value or "").strip()
        return raw.strip("/")

    def _ensure_done_folder(self) -> None:
        if not self._done_folder_ready:
            self._onedrive.ensure_sorted_subfolder("_done")
            self._done_folder_ready = True

    def _move_source_to_done(self, info: Dict[str, Any]) -> None:
        self._ensure_done_folder()
        try:
            self._onedrive.move_sorted_item(info["relative_path"], "_done", info["name"])
            self._cache.append_log(f"Moved '{info['name']}' to '_done/'.")
        except Exception as exc:
            LOGGER.exception("Failed to move '%s' to _done: %s", info["relative_path"], exc)
            self._cache.append_log(f"Failed to move '{info['name']}' to '_done': {exc}", level="error")


__all__ = ["StructureService", "StructureServiceError"]
