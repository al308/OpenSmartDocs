"""Plan parsing, repair, and validation utilities."""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

from ..ollama_client import OllamaClient
from .models import (
    StructureContext,
    StructureValidationModel,
    STRUCTURE_FIX_PROMPT_TEMPLATE,
    STRUCTURE_FIX_SYSTEM_PROMPT,
    STRUCTURE_PLAN_JSON_SCHEMA,
    STRUCTURE_VALIDATION_JSON_SCHEMA,
    STRUCTURE_VALIDATION_PROMPT_TEMPLATE,
    STRUCTURE_VALIDATION_SYSTEM_PROMPT,
    StructureCache,
    StructureServiceError,
    utc_now_iso,
)
from .constants import MAX_SOURCES_IN_PROMPT

LOGGER = logging.getLogger(__name__)


class PlanEngine:
    def __init__(
        self,
        *,
        ollama: OllamaClient,
        model_name: str,
        cache: StructureCache,
        endpoint: str,
    ):
        self._ollama = ollama
        self._model_name = model_name
        self._cache = cache
        self._endpoint = endpoint

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def generate_plan(self, prompt_payload: str, raw_response: str, context: StructureContext) -> Dict[str, Any]:
        repaired = False
        response = raw_response
        while True:
            try:
                plan = self._parse_plan(response, context)
                break
            except Exception as exc:
                if repaired:
                    self._handle_plan_failure(response, exc)
                repaired = True
                try:
                    response = self._attempt_plan_repair(prompt_payload, response, str(exc))
                    self._cache.append_log("Attempted to repair structure plan output.", level="warning")
                except Exception as repair_exc:
                    self._handle_plan_failure(response, repair_exc)

        if plan["operations"]:
            validation = self._run_plan_validation(plan, context)
            plan["validation"] = validation
            status = (validation.get("status") or "").lower()
            if status == "warn":
                LOGGER.warning("Structure plan validation warning: %s", validation.get("notes"))
                self._cache.append_log(f"Sanity check warning: {validation.get('notes')}", level="warning")
            elif status == "error":
                LOGGER.error("Structure plan validation error: %s", validation.get("notes"))
                self._cache.append_log(f"Sanity check error: {validation.get('notes')}", level="error")
        return plan

    # ------------------------------------------------------------------ #
    # Core parsing & validation                                          #
    # ------------------------------------------------------------------ #

    def _parse_plan(self, raw_json: str, context: StructureContext) -> Dict[str, Any]:
        data = self._decode_plan_payload(raw_json)
        summary = self._extract_summary(data, raw_json)
        operations = self._extract_operations(data)

        allowed_sources = {src.source_id for src in context.sources[:MAX_SOURCES_IN_PROMPT]}
        validated_ops: list[Dict[str, Any]] = []

        for op in operations:
            if not isinstance(op, dict):
                continue
            action = op.get("action")
            if action not in {"create_folder", "copy_file"}:
                continue
            if action == "create_folder":
                path = self._normalized_relative(op.get("path"))
                if not path:
                    continue
                validated_ops.append(
                    {
                        "action": "create_folder",
                        "path": path,
                        "justification": (op.get("justification") or "").strip(),
                    }
                )
            elif action == "copy_file":
                source_id = op.get("source_id")
                target_name = op.get("target_name")
                if source_id not in allowed_sources or not isinstance(target_name, str) or not target_name.strip():
                    continue
                target_folder = self._normalized_relative(op.get("target_folder", ""))
                validated_ops.append(
                    {
                        "action": "copy_file",
                        "source_id": source_id,
                        "target_folder": target_folder,
                        "target_name": target_name.strip(),
                        "justification": (op.get("justification") or "").strip(),
                    }
                )

        plan_id = str(uuid.uuid4())
        plan = {
            "plan_id": plan_id,
            "generated_at": utc_now_iso(),
            "model": self._model_name,
            "summary": summary.strip(),
            "operations": self._ensure_coverage(validated_ops, context),
            "context": context.snapshot,
        }

        if context.sources and not plan["operations"]:
            raise ValueError("Model returned no operations despite available sources.")
        return plan

    def _ensure_coverage(self, operations: List[Dict[str, Any]], context: StructureContext) -> List[Dict[str, Any]]:
        sources = context.sources[:MAX_SOURCES_IN_PROMPT]
        covered_sources = {
            op["source_id"]
            for op in operations
            if op["action"] == "copy_file" and isinstance(op.get("source_id"), str)
        }
        missing_sources = [src for src in sources if src.source_id not in covered_sources]
        if not missing_sources:
            return operations

        fallback_ops: list[Dict[str, Any]] = []
        planned_folders = {
            op["path"]
            for op in operations
            if op["action"] == "create_folder" and isinstance(op.get("path"), str)
        }
        existing_folders = set(context.existing_folders)

        for src in missing_sources:
            suggested_folder = self._normalized_relative(src.suggested_folder() or "")
            if suggested_folder and suggested_folder not in existing_folders and suggested_folder not in planned_folders:
                fallback_ops.append(
                    {
                        "action": "create_folder",
                        "path": suggested_folder,
                        "justification": f"Create folder for {src.metadata.get('document_type', 'documents')} based on heuristics.",
                    }
                )
                planned_folders.add(suggested_folder)
            target_name = src.suggested_target_name()
            fallback_ops.append(
                {
                    "action": "copy_file",
                    "source_id": src.source_id,
                    "target_folder": suggested_folder,
                    "target_name": target_name,
                    "justification": src.default_justification(suggested_folder),
                }
            )

        if fallback_ops:
            LOGGER.warning("Structure model omitted %d source(s); generated fallback operations.", len(missing_sources))
            try:
                self._cache.append_log(
                    f"Generated fallback operations for {len(missing_sources)} source(s).",
                    level="warning",
                )
            except Exception:  # pragma: no cover - cache write failure
                LOGGER.debug("Failed to append fallback log entry.", exc_info=True)
            operations = operations + fallback_ops
        return operations

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _attempt_plan_repair(self, prompt_payload: str, raw_response: str, error_message: str) -> str:
        repair_prompt = STRUCTURE_FIX_PROMPT_TEMPLATE.format(
            schema_json=json.dumps(STRUCTURE_PLAN_JSON_SCHEMA, indent=2, ensure_ascii=False),
            instruction_prompt=prompt_payload,
            model_response=raw_response,
            error_message=error_message,
        )
        LOGGER.warning("Attempting to repair structure plan output due to parse error: %s", error_message)
        return self._ollama.request_json_schema_completion(
            system_prompt=STRUCTURE_FIX_SYSTEM_PROMPT,
            prompt=repair_prompt,
            model=self._model_name,
            json_schema=STRUCTURE_PLAN_JSON_SCHEMA,
            schema_name="StructurePlan",
            max_tokens=2048,
            temperature=0.2,
        )

    def _run_plan_validation(self, plan: Dict[str, Any], context: StructureContext) -> Dict[str, Any]:
        validation_prompt = STRUCTURE_VALIDATION_PROMPT_TEMPLATE.format(
            schema_json=json.dumps(STRUCTURE_VALIDATION_JSON_SCHEMA, indent=2, ensure_ascii=False),
            plan_json=json.dumps({"summary": plan.get("summary"), "operations": plan.get("operations", [])}, indent=2, ensure_ascii=False),
            existing_folders=json.dumps(sorted(context.existing_folders), indent=2, ensure_ascii=False),
        )
        try:
            raw = self._ollama.request_json_schema_completion(
                system_prompt=STRUCTURE_VALIDATION_SYSTEM_PROMPT,
                prompt=validation_prompt,
                model=self._model_name,
                json_schema=STRUCTURE_VALIDATION_JSON_SCHEMA,
                schema_name="StructureValidation",
                max_tokens=1024,
                temperature=0.0,
            )
        except Exception as exc:  # pragma: no cover - network failure
            LOGGER.exception("Validation request failed: %s", exc)
            self._cache.append_log(f"Sanity check failed: {exc}", level="error")
            return {"status": "error", "notes": f"Validation failed: {exc}", "issues": []}
        try:
            data = json.loads(raw)
            feedback = StructureValidationModel.model_validate(data)
            return feedback.model_dump()
        except Exception as exc:
            LOGGER.exception("Failed to parse validation feedback: %s", exc)
            self._cache.append_log(f"Sanity check parsing failed: {exc}", level="error")
            return {"status": "error", "notes": f"Validation parsing failed: {exc}", "issues": []}

    def _handle_plan_failure(self, raw_response: str, exc: Exception) -> None:
        snippet_source = raw_response if isinstance(raw_response, str) else json.dumps(raw_response, ensure_ascii=False)
        trimmed = (snippet_source or "").strip()
        debug_info = f"model={self._model_name} endpoint={self._endpoint}"
        if not trimmed:
            LOGGER.error("Structure model returned an empty response (%s).", debug_info)
            self._cache.append_log(f"Model returned empty response ({debug_info}).", level="error")
        else:
            LOGGER.error("Structure plan raw response (truncated): %r (%s)", trimmed[:1000], debug_info)
            preview = trimmed[:200].replace("\n", " ")
            self._cache.append_log(f"Model returned non-JSON output: {preview} ({debug_info})", level="error")
        message = f"Failed to interpret structure plan: {exc}"
        LOGGER.exception("Structure plan parsing failed: %s", exc)
        self._cache.append_log(message, level="error")
        raise StructureServiceError(message) from exc

    @staticmethod
    def _normalized_relative(value: Optional[str]) -> str:
        raw = (value or "").strip()
        return raw.strip("/")

    @staticmethod
    def _decode_plan_payload(raw_json: str) -> dict:
        text = (raw_json or "").strip()
        decoder = json.JSONDecoder()

        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
            raise ValueError("Structure plan must be a JSON object.")
        except json.JSONDecodeError:
            pass

        try:
            import ast

            data = ast.literal_eval(text)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

        try:
            obj, _ = decoder.raw_decode(text)
            if isinstance(obj, dict):
                LOGGER.debug("Recovered structure JSON by trimming trailing text.")
                return obj
        except json.JSONDecodeError:
            pass

        for index, char in enumerate(text):
            if char in "[{":
                try:
                    obj, _ = decoder.raw_decode(text[index:])
                    if isinstance(obj, dict):
                        LOGGER.debug("Recovered structure JSON by skipping leading text.")
                        return obj
                except json.JSONDecodeError:
                    continue

        brace_index = text.rfind("}")
        while brace_index != -1:
            candidate = text[: brace_index + 1]
            try:
                candidate = candidate.strip()
                if candidate and candidate[0] not in "{[":
                    start = candidate.find("{")
                    if start != -1:
                        candidate = candidate[start:]
                data = json.loads(candidate)
                if isinstance(data, dict):
                    LOGGER.debug("Recovered structure JSON by truncating trailing content.")
                    return data
            except json.JSONDecodeError:
                pass
            brace_index = text.rfind("}", 0, brace_index)

        raise ValueError("Structure model returned invalid JSON")

    @staticmethod
    def _extract_summary(data: dict, raw_json: str) -> str:
        summary = data.get("summary")
        if isinstance(summary, str):
            value = summary
        else:
            value = None
            if summary is None:
                for alt_key in ("Summary", "summary_text", "summary_texts", "plan_summary", "description", "title"):
                    candidate = data.get(alt_key)
                    if candidate is not None:
                        value = candidate
                        break
                if value is None:
                    for regex in (
                        r'"summary"\s*:\s*"([^"\]*(?:\.[^"\]*)*)',
                        r"'summary'\s*:\s*'([^'\]*(?:\.[^'\]*)*)",
                    ):
                        match = re.search(regex, raw_json)
                        if match:
                            value = match.group(1)
                            break
            else:
                value = summary

        if value is None:
            raise ValueError("Plan summary missing from model response.")
        if isinstance(value, list):
            value = " ".join(str(item) for item in value if item)
        elif isinstance(value, (dict, tuple, set)):
            value = json.dumps(value, ensure_ascii=False)
        else:
            value = str(value)

        value = value.replace("\\n", " ").replace("\\t", " ")
        return value.strip().strip('"').strip("'")

    @staticmethod
    def _extract_operations(data: dict) -> List[Dict[str, Any]]:
        operations = data.get("operations") or []
        if isinstance(operations, str):
            try:
                operations = json.loads(operations)
            except json.JSONDecodeError:
                raise ValueError("Plan operations must be a list.") from None
        if not isinstance(operations, list):
            raise ValueError("Plan operations must be a list.")
        return operations


__all__ = ["PlanEngine"]
