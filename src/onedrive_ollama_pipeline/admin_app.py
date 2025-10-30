"""FastAPI admin interface for the OneDrive → Ollama pipeline."""
from __future__ import annotations

import io
import json
import re
import sqlite3
from hashlib import md5
from pathlib import Path
from typing import Any, Literal, Optional

import pikepdf
from fastapi import FastAPI, HTTPException, File, Form, UploadFile, Body
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from .config import CONFIG_DEFAULTS, load_config_file, load_settings, save_config_file
from .database import Database, get_database
from .logging_utils import get_runtime_log_level, set_runtime_log_level
from .onedrive_client import DriveItem, OneDriveClient
from .pipeline import Pipeline
from .pdf_processor import inspect_pdf_content
from .state_store import StateStore
from .structure_service import StructureService, StructureServiceError
from .ollama_client import OllamaClient

app = FastAPI(title="OneDrive Ollama Pipeline Admin")

_TEMPLATE_PATH = Path(__file__).resolve().parent / "static" / "admin.html"
_ADMIN_TEMPLATE: Optional[str] = None
_TEXT_INFO_CACHE: dict[str, tuple[str, bool]] = {}


class ConfigResponse(BaseModel):
    ollama_model: str
    poll_interval_seconds: Optional[int]
    inbox_folder: str
    sorted_folder: str
    log_level: str
    auto_process_inbox: bool
    structure_model: str
    structure_language: str


class ConfigUpdateRequest(BaseModel):
    ollama_model: Optional[str] = Field(default=None, alias="ollamaModel")
    poll_interval_seconds: Optional[int] = Field(default=None, alias="pollIntervalSeconds")
    inbox_folder: Optional[str] = Field(default=None, alias="inboxFolder")
    sorted_folder: Optional[str] = Field(default=None, alias="sortedFolder")
    log_level: Optional[str] = Field(default=None, alias="logLevel")
    auto_process_inbox: Optional[bool] = Field(default=None, alias="autoProcessInbox")
    structure_model: Optional[str] = Field(default=None, alias="structureModel")
    structure_language: Optional[str] = Field(default=None, alias="structureLanguage")

    @field_validator("poll_interval_seconds")
    def _non_negative(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value < 0:
            raise ValueError("poll_interval_seconds must be >= 0")
        return value


class SQLRequest(BaseModel):
    sql: str


class InboxProcessRequest(BaseModel):
    item_ids: list[str] = Field(default_factory=list, alias="itemIds")

    @field_validator("item_ids")
    def _ensure_non_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("itemIds must not be empty")
        return value


class IngestProcessRequest(BaseModel):
    item_id: str = Field(alias="itemId")
    drive_id: Optional[str] = Field(default=None, alias="driveId")
    name: str
    mode: Literal["auto", "text", "image", "both"] = "auto"
    download_url: Optional[str] = Field(default=None, alias="downloadUrl")
    web_url: Optional[str] = Field(default=None, alias="webUrl")

    @field_validator("name")
    def _name_not_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("name must not be empty")
        return cleaned


class StructureAnalyzeRequest(BaseModel):
    relative_paths: list[str] = Field(default_factory=list, alias="relativePaths")


def _build_metadata_preview(metadata: Any) -> str:
    if isinstance(metadata, dict):
        preview = metadata.get("summary") or metadata.get("title") or ""
        if not preview:
            preview = ", ".join(f"{key}: {value}" for key, value in list(metadata.items())[:3])
        if not preview:
            preview = json.dumps(metadata, ensure_ascii=False)
        return preview
    if metadata is not None:
        return str(metadata)
    return ""


def _get_onedrive_client() -> OneDriveClient:
    settings = load_settings()
    return OneDriveClient(settings.graph)


def _get_structure_service() -> StructureService:
    return StructureService()


def _sanitize_pdf_filename(candidate: str, *, allow_extensionless: bool = False) -> str:
    name = Path(candidate or "").name
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Filename must not be empty")
    if allow_extensionless and "." not in name:
        name = f"{name}.pdf"
    if not name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    base = Path(name).stem
    safe_base = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    if not safe_base:
        safe_base = "document"
    return f"{safe_base}.pdf"


def _combine_duplex_pdfs(odd_pdf: bytes, even_pdf: bytes) -> tuple[bytes, int]:
    """Merge simplex scans into a duplex PDF, assuming back sides are reversed."""
    try:
        with pikepdf.Pdf.open(io.BytesIO(odd_pdf)) as odd_doc, pikepdf.Pdf.open(io.BytesIO(even_pdf)) as even_doc:
            odd_pages = list(odd_doc.pages)
            even_pages = list(even_doc.pages)
            if len(odd_pages) != len(even_pages):
                raise HTTPException(
                    status_code=400,
                    detail=f"Odd ({len(odd_pages)}) and even ({len(even_pages)}) scans must contain the same number of pages.",
                )
            combined = pikepdf.Pdf.new()
            reversed_even = list(reversed(even_pages))
            for index, odd_page in enumerate(odd_pages):
                combined.pages.append(odd_page)
                combined.pages.append(reversed_even[index])
    except pikepdf.PdfError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid PDF upload: {exc}") from exc

    buffer = io.BytesIO()
    combined.save(buffer)
    combined.close()
    return buffer.getvalue(), len(odd_pages) + len(even_pages)


def _has_meaningful_text(pdf_bytes: bytes, *, min_chars: int = 200) -> bool:
    try:
        inspection = inspect_pdf_content(pdf_bytes, text_max_pages=3, text_max_chars=3000)
    except Exception:
        return False
    text_section = inspection.get("text") or {}
    if not isinstance(text_section, dict):
        return False
    available = bool(text_section.get("available"))
    chars = int(text_section.get("chars") or 0)
    return available and chars >= min_chars


def _cache_text_info(cache_key: str, digest: str, has_text: bool) -> None:
    _TEXT_INFO_CACHE[cache_key] = (digest, has_text)


def _lookup_text_info(cache_key: str, digest: str) -> Optional[bool]:
    cached = _TEXT_INFO_CACHE.get(cache_key)
    if not cached:
        return None
    cached_digest, cached_value = cached
    if cached_digest == digest:
        return cached_value
    return None


@app.get("/", response_class=HTMLResponse)
def admin_home() -> str:
    return _render_admin_page()


@app.get("/api/status")
def api_status() -> dict[str, Any]:
    settings = load_settings()
    db = get_database(settings.db_path)
    totals = db.fetch_one(
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS success,
            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed
        FROM processed_items
        """
    )
    raw_recent = StateStore(settings.db_path).recent_items(limit=25)
    recent: list[dict[str, Any]] = []
    for entry in raw_recent:
        metadata = entry.get("metadata")
        preview = ""
        if isinstance(metadata, dict):
            preview = metadata.get("summary") or metadata.get("title") or ""
            if not preview:
                preview = ", ".join(f"{key}: {value}" for key, value in list(metadata.items())[:3])
            if not preview:
                preview = json.dumps(metadata, ensure_ascii=False)
        elif metadata is not None:
            preview = str(metadata)
        if preview and len(preview) > 400:
            preview = preview[:400] + "…"
        trimmed = {key: value for key, value in entry.items() if key != "metadata"}
        trimmed["metadata_preview"] = preview
        recent.append(trimmed)
    return {
        "counts": {
            "total": totals["total"] if totals else 0,
            "success": totals["success"] if totals else 0,
            "failed": totals["failed"] if totals else 0,
        },
        "recent": recent,
    }


@app.get("/api/config", response_model=ConfigResponse)
def api_get_config() -> ConfigResponse:
    settings = load_settings()
    config_data = load_config_file(settings.config_path)
    return ConfigResponse(
        ollama_model=config_data["ollama"].get("model", CONFIG_DEFAULTS["ollama"]["model"]),
        poll_interval_seconds=config_data["pipeline"].get("poll_interval_seconds"),
        inbox_folder=config_data["graph"].get("inbox_folder", CONFIG_DEFAULTS["graph"]["inbox_folder"]),
        sorted_folder=config_data["graph"].get("sorted_folder", CONFIG_DEFAULTS["graph"]["sorted_folder"]),
        log_level=config_data["pipeline"].get("log_level", get_runtime_log_level()),
        auto_process_inbox=config_data["pipeline"].get("auto_process_inbox", CONFIG_DEFAULTS["pipeline"]["auto_process_inbox"]),
        structure_model=config_data.get("structure", {}).get("model")
        or config_data["ollama"].get("model", CONFIG_DEFAULTS["ollama"]["model"]),
        structure_language=config_data.get("structure", {}).get("language", CONFIG_DEFAULTS["structure"].get("language", "auto")),
    )


@app.put("/api/config", response_model=ConfigResponse)
def api_update_config(request: ConfigUpdateRequest) -> ConfigResponse:
    settings = load_settings()
    config_data = load_config_file(settings.config_path)

    if request.log_level is not None:
        set_runtime_log_level(request.log_level)
        config_data.setdefault("pipeline", {})["log_level"] = request.log_level.upper()
    if request.ollama_model is not None:
        config_data.setdefault("ollama", {})["model"] = request.ollama_model
    if request.poll_interval_seconds is not None:
        config_data.setdefault("pipeline", {})["poll_interval_seconds"] = request.poll_interval_seconds
    if request.inbox_folder is not None:
        config_data.setdefault("graph", {})["inbox_folder"] = request.inbox_folder
    if request.sorted_folder is not None:
        config_data.setdefault("graph", {})["sorted_folder"] = request.sorted_folder
    if request.auto_process_inbox is not None:
        config_data.setdefault("pipeline", {})["auto_process_inbox"] = bool(request.auto_process_inbox)
    if request.structure_model is not None:
        value = (request.structure_model or "").strip()
        if value:
            config_data.setdefault("structure", {})["model"] = value
        else:
            config_data.setdefault("structure", {})
            config_data["structure"].pop("model", None)
    if request.structure_language is not None:
        lang = (request.structure_language or "").strip() or "auto"
        config_data.setdefault("structure", {})["language"] = lang

    save_config_file(settings.config_path, config_data)

    return api_get_config()


@app.post("/api/query")
def api_sql_query(request: SQLRequest) -> dict[str, Any]:
    sql = request.sql.strip()
    if not sql:
        raise HTTPException(status_code=400, detail="SQL query must not be empty")
    if not sql.lower().startswith("select"):
        raise HTTPException(status_code=400, detail="Only SELECT queries are permitted")
    settings = load_settings()
    db = get_database(settings.db_path)
    try:
        rows = db.query(sql)
    except sqlite3.Error as exc:  # pragma: no cover - depends on query
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "columns": rows[0].keys() if rows else [],
        "rows": [dict(row) for row in rows],
    }


@app.get("/api/logs")
def api_logs(limit: int = 200) -> dict[str, Any]:
    settings = load_settings()
    lines = _tail_file(settings.log_path, limit)
    return {"lines": lines, "path": settings.log_path.as_posix()}


@app.get("/api/processed")
def api_processed(limit: int = 50) -> dict[str, Any]:
    settings = load_settings()
    store = StateStore(settings.db_path)
    return {"items": store.recent_items(limit=limit)}


@app.get("/api/inbox")
def api_inbox() -> dict[str, Any]:
    settings = load_settings()
    client = _get_onedrive_client()
    try:
        inbox_items = list(client.list_pdfs_in_inbox())
    except Exception as exc:  # pragma: no cover - network failure
        raise HTTPException(status_code=500, detail=f"Failed to list inbox items: {exc}") from exc

    ids = [item.item_id for item in inbox_items if item.item_id]
    status_lookup: dict[str, dict[str, Any]] = {}
    db = get_database(settings.db_path)
    if ids:
        placeholders = ",".join("?" for _ in ids)
        rows = db.query(
            f"""
            SELECT onedrive_id, filename, processed_at, status, model, metadata_json, error_message
            FROM processed_items
            WHERE onedrive_id IN ({placeholders})
            """,
            ids,
        )
        for row in rows:
            metadata = Database.decode_json(row["metadata_json"])
            preview = _build_metadata_preview(metadata)
            if preview and len(preview) > 400:
                preview = preview[:400] + "…"
            status_lookup[row["onedrive_id"]] = {
                "status": row["status"],
                "processed_at": row["processed_at"],
                "model": row["model"],
                "metadata_preview": preview,
                "error_message": row["error_message"],
            }

    response_items: list[dict[str, Any]] = []
    processed = failed = 0
    for item in inbox_items:
        info = status_lookup.get(item.item_id, {})
        status = info.get("status", "pending")
        if status == "success":
            processed += 1
        elif status == "failed":
            failed += 1
        response_items.append(
            {
                "id": item.item_id,
                "name": item.name,
                "web_url": item.web_url,
                "status": status,
                "processed_at": info.get("processed_at"),
                "model": info.get("model"),
                "metadata_preview": info.get("metadata_preview", ""),
                "error_message": info.get("error_message"),
            }
        )

    pending = len(response_items) - processed - failed
    config_data = load_config_file(settings.config_path)
    config = {
        "inboxFolder": config_data["graph"].get("inbox_folder", CONFIG_DEFAULTS["graph"]["inbox_folder"]),
        "autoProcessInbox": config_data["pipeline"].get("auto_process_inbox", CONFIG_DEFAULTS["pipeline"]["auto_process_inbox"]),
        "pollIntervalSeconds": config_data["pipeline"].get("poll_interval_seconds"),
        "structureLanguage": config_data.get("structure", {}).get("language", CONFIG_DEFAULTS["structure"].get("language", "auto")),
        "structureModel": config_data.get("structure", {}).get("model")
        or config_data["ollama"].get("model", CONFIG_DEFAULTS["ollama"]["model"]),
    }
    return {
        "config": config,
        "items": response_items,
        "counts": {
            "total": len(response_items),
            "processed": processed,
            "failed": failed,
            "pending": pending,
        },
    }


@app.get("/api/structure")
def api_structure_state() -> dict[str, Any]:
    service = _get_structure_service()
    return service.get_state()


@app.get("/api/structure/sources")
def api_structure_sources(limit: int = 200) -> dict[str, Any]:
    service = _get_structure_service()
    safe_limit = max(1, min(limit, 2000))
    return service.list_sources(max_items=safe_limit)


@app.get("/api/inbox/preview/{item_id}")
def api_inbox_preview(item_id: str) -> StreamingResponse:
    client = _get_onedrive_client()
    try:
        for item in client.list_pdfs_in_inbox():
            if item.item_id == item_id:
                try:
                    content = client.download_item(item)
                except Exception as exc:  # pragma: no cover - network failure
                    raise HTTPException(status_code=500, detail=f"Failed to download inbox file: {exc}") from exc
                digest = md5(content).hexdigest()
                cache_key = f"inbox:{item.item_id}"
                cached = _lookup_text_info(cache_key, digest)
                if cached is None:
                    result = _has_meaningful_text(content)
                    _cache_text_info(cache_key, digest, result)
                headers = {"Content-Disposition": f'inline; filename="{item.name}"'}
                return StreamingResponse(io.BytesIO(content), media_type="application/pdf", headers=headers)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - network failure
        raise HTTPException(status_code=500, detail=f"Failed to enumerate inbox files: {exc}") from exc
    raise HTTPException(status_code=404, detail="Inbox file not found")


@app.get("/api/inbox/text-info/{item_id}")
def api_inbox_text_info(item_id: str) -> dict[str, bool]:
    client = _get_onedrive_client()
    try:
        for item in client.list_pdfs_in_inbox():
            if item.item_id != item_id:
                continue
            try:
                content = client.download_item(item)
            except Exception as exc:  # pragma: no cover - network failure
                raise HTTPException(status_code=500, detail=f"Failed to download inbox file: {exc}") from exc
            digest = md5(content).hexdigest()
            cache_key = f"inbox:{item.item_id}"
            cached = _lookup_text_info(cache_key, digest)
            if cached is None:
                cached = _has_meaningful_text(content)
                _cache_text_info(cache_key, digest, cached)
            return {"hasText": cached}
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - network failure
        raise HTTPException(status_code=500, detail=f"Failed to enumerate inbox files: {exc}") from exc
    raise HTTPException(status_code=404, detail="Inbox file not found")


@app.get("/api/structure/preview")
def api_structure_preview(relative_path: str) -> StreamingResponse:
    normalized = (relative_path or "").strip().strip("/")
    if not normalized:
        raise HTTPException(status_code=400, detail="relative_path is required")
    if ".." in normalized.split("/"):
        raise HTTPException(status_code=400, detail="Invalid relative_path")
    client = _get_onedrive_client()
    try:
        entry, content = client.download_sorted_file(normalized)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - network failure
        raise HTTPException(status_code=500, detail=f"Failed to download sorted file: {exc}") from exc
    cache_key = f"structure:{entry.item_id or normalized}"
    digest = md5(content).hexdigest()
    cached = _lookup_text_info(cache_key, digest)
    if cached is None:
        cached = _has_meaningful_text(content)
        _cache_text_info(cache_key, digest, cached)
    filename = entry.name if getattr(entry, "name", None) else Path(normalized).name
    headers = {"Content-Disposition": f'inline; filename="{filename}"'}
    return StreamingResponse(io.BytesIO(content), media_type="application/pdf", headers=headers)


@app.get("/api/structure/text-info")
def api_structure_text_info(relative_path: str) -> dict[str, bool]:
    normalized = (relative_path or "").strip().strip("/")
    if not normalized:
        raise HTTPException(status_code=400, detail="relative_path is required")
    if ".." in normalized.split("/"):
        raise HTTPException(status_code=400, detail="Invalid relative_path")
    client = _get_onedrive_client()
    try:
        entry, content = client.download_sorted_file(normalized)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - network failure
        raise HTTPException(status_code=500, detail=f"Failed to download sorted file: {exc}") from exc
    cache_key = f"structure:{entry.item_id or normalized}"
    digest = md5(content).hexdigest()
    cached = _lookup_text_info(cache_key, digest)
    if cached is None:
        cached = _has_meaningful_text(content)
        _cache_text_info(cache_key, digest, cached)
    return {"hasText": cached}


@app.post("/api/structure/analyze")
def api_structure_analyze(request: StructureAnalyzeRequest | None = Body(default=None)) -> dict[str, Any]:
    service = _get_structure_service()
    include_paths: Optional[set[str]] = None
    if request and request.relative_paths:
        include_paths = {path for path in request.relative_paths if path}
    try:
        return service.analyze(include_paths)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except StructureServiceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/structure/apply")
def api_structure_apply() -> dict[str, Any]:
    service = _get_structure_service()
    try:
        return service.apply()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except StructureServiceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/structure/revert")
def api_structure_revert() -> dict[str, Any]:
    service = _get_structure_service()
    try:
        return service.revert()
    except StructureServiceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/inbox/process")
def api_inbox_process(request: InboxProcessRequest) -> dict[str, Any]:
    settings = load_settings()
    pipeline = Pipeline(settings)
    client = pipeline._onedrive
    state = pipeline._state
    try:
        inbox_items = {item.item_id: item for item in client.list_pdfs_in_inbox()}
    except Exception as exc:  # pragma: no cover - network failure
        raise HTTPException(status_code=500, detail=f"Failed to list inbox items: {exc}") from exc

    results: list[dict[str, Any]] = []
    for item_id in request.item_ids:
        item = inbox_items.get(item_id)
        if item is None:
            results.append({"id": item_id, "status": "missing"})
            continue
        try:
            progress: list[str] = []

            def _progress(stage: str) -> None:
                progress.append(stage)

            metadata = pipeline._process_item(item, progress_callback=_progress)  # type: ignore[attr-defined]
            state.record_success(
                item_id=item.item_id,
                filename=item.name,
                model=settings.ollama.model,
                metadata=metadata,
            )
            results.append({"id": item_id, "status": "success", "name": item.name, "progress": progress})
        except Exception as exc:  # pragma: no cover - runtime failure
            state.record_failure(
                item_id=item.item_id,
                filename=item.name,
                model=settings.ollama.model,
                error=str(exc),
            )
            results.append({"id": item_id, "status": "failed", "name": item.name, "error": str(exc), "progress": progress})

    processed = sum(1 for result in results if result["status"] == "success")
    failed = sum(1 for result in results if result["status"] == "failed")
    return {"processed": processed, "failed": failed, "results": results}


@app.post("/api/ollama/test")
def api_ollama_test() -> dict[str, Any]:
    settings = load_settings()
    client = OllamaClient(settings.ollama)
    results: list[dict[str, str]] = []

    def _run(model: str, label: str) -> None:
        try:
            client.test_model(model=model, prompt='Sanity check, respond with "success"', expected="success")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"{label.capitalize()} model '{model}' failed: {exc}") from exc
        results.append({"label": label, "model": model, "status": "success"})

    metadata_model = settings.ollama.model
    _run(metadata_model, "metadata")
    structure_model = settings.structure_model or metadata_model
    if structure_model != metadata_model:
        _run(structure_model, "structure")

    return {"status": "ok", "results": results}


@app.post("/api/ingest/upload")
async def api_ingest_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    filename = _sanitize_pdf_filename(file.filename)
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    inspection = inspect_pdf_content(content)

    client = _get_onedrive_client()
    try:
        upload_info = client.upload_pdf_to_inbox(filename, content)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - network errors
        raise HTTPException(status_code=500, detail=f"Failed to upload PDF: {exc}") from exc

    drive_item: Optional[DriveItem] = None
    if isinstance(upload_info, dict):
        try:
            drive_item = OneDriveClient.drive_item_from_payload(upload_info)
        except Exception:
            drive_item = None

    recommended_strategy = "text" if inspection.get("text", {}).get("available") else "image"

    return {
        "status": "ok",
        "filename": filename,
        "analysis": inspection,
        "recommendedStrategy": recommended_strategy,
        "item": {
            "itemId": (drive_item.item_id if drive_item else None),
            "driveId": (drive_item.drive_id if drive_item else None),
            "name": (drive_item.name if drive_item else filename),
            "downloadUrl": (drive_item.download_url if drive_item else None),
            "webUrl": (drive_item.web_url if drive_item else None),
        },
    }


@app.post("/api/ingest/process")
def api_ingest_process(request: IngestProcessRequest) -> dict[str, Any]:
    settings = load_settings()
    pipeline = Pipeline(settings)
    state = pipeline._state
    drive_item = DriveItem(
        item_id=request.item_id,
        drive_id=request.drive_id,
        name=request.name,
        download_url=request.download_url,
        web_url=request.web_url,
    )

    progress: list[str] = []

    def _progress(stage: str) -> None:
        progress.append(stage)

    try:
        metadata = pipeline._process_item(
            drive_item,
            progress_callback=_progress,
            metadata_strategy=request.mode,
        )
        state.record_success(
            item_id=drive_item.item_id,
            filename=drive_item.name,
            model=settings.ollama.model,
            metadata=metadata,
        )
        return {
            "status": "ok",
            "itemId": drive_item.item_id,
            "mode": request.mode,
            "metadata": metadata,
            "progress": progress,
        }
    except Exception as exc:
        state.record_failure(
            item_id=drive_item.item_id,
            filename=drive_item.name,
            model=settings.ollama.model,
            error=str(exc),
        )
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc


@app.post("/api/ingest/mass-scan")
async def api_ingest_mass_scan(
    odd_file: UploadFile = File(..., alias="oddFile"),
    even_file: UploadFile = File(..., alias="evenFile"),
    output_name: str = Form("", alias="outputName"),
) -> dict[str, Any]:
    if not odd_file.filename or not even_file.filename:
        raise HTTPException(status_code=400, detail="Both odd and even PDFs are required")

    odd_filename = _sanitize_pdf_filename(odd_file.filename)
    even_filename = _sanitize_pdf_filename(even_file.filename)

    odd_bytes = await odd_file.read()
    even_bytes = await even_file.read()
    if not odd_bytes or not even_bytes:
        raise HTTPException(status_code=400, detail="Uploaded PDFs must not be empty")

    try:
        combined_pdf, total_pages = _combine_duplex_pdfs(odd_bytes, even_bytes)
    except HTTPException:
        raise

    output_candidate = output_name.strip()
    if output_candidate:
        output_filename = _sanitize_pdf_filename(output_candidate, allow_extensionless=True)
    else:
        fallback_stem = Path(odd_filename).stem or "document"
        output_filename = _sanitize_pdf_filename(f"{fallback_stem}_duplex.pdf")

    client = _get_onedrive_client()
    try:
        client.upload_pdf_to_inbox(output_filename, combined_pdf)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - network errors
        raise HTTPException(status_code=500, detail=f"Failed to upload merged PDF: {exc}") from exc

    return {
        "status": "ok",
        "filename": output_filename,
        "pages": total_pages,
        "sources": {"odd": odd_filename, "even": even_filename},
    }


def _tail_file(path: Path, limit: int) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            lines = handle.readlines()
    except OSError:  # pragma: no cover - rare filesystem errors
        return []
    return [line.rstrip("\n") for line in lines[-limit:]]


def _render_admin_page() -> str:
    global _ADMIN_TEMPLATE
    if _ADMIN_TEMPLATE is None:
        try:
            _ADMIN_TEMPLATE = _TEMPLATE_PATH.read_text(encoding="utf-8")
        except FileNotFoundError as exc:  # pragma: no cover - template missing is critical
            raise RuntimeError(f"Admin template not found at {_TEMPLATE_PATH}") from exc
    return _ADMIN_TEMPLATE
