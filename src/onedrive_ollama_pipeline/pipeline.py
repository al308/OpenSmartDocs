"""End-to-end processing pipeline to enrich OneDrive PDFs with Ollama metadata."""
from __future__ import annotations

import io
import logging
import time
from typing import Callable, Optional

from PIL import Image

from .config import PipelineSettings, load_settings
from .ollama_client import OllamaClient
from .onedrive_client import OneDriveClient
from .pdf_processor import embed_metadata, pdf_to_png_pages
from .state_store import StateStore

_LOGGER = logging.getLogger(__name__)


class Pipeline:
    """Orchestrate the OneDrive to Ollama metadata enrichment flow."""

    def __init__(self, settings: PipelineSettings):
        self._settings = settings
        self._state = StateStore(settings.db_path)
        self._onedrive = OneDriveClient(settings.graph)
        self._ollama = OllamaClient(settings.ollama)

    def run_startup_checks(self) -> None:
        """Run lightweight diagnostics to fail fast on misconfiguration."""
        if not self._settings.auto_process_inbox:
            _LOGGER.info("Auto-processing disabled; skipping startup checks.")
            return
        _LOGGER.info("Running startup configuration checks")
        self._ollama.ensure_ready()

    def run_once(self) -> int:
        """Process each unprocessed PDF from the inbox exactly one time."""
        if not self._settings.auto_process_inbox:
            _LOGGER.info("Auto-processing disabled; skipping inbox scan.")
            return 0
        processed = 0
        items = list(self._onedrive.list_pdfs_in_inbox())
        total = len(items)
        folder_label = self._inbox_folder_label()
        _LOGGER.info("Discovered %d PDF(s) in OneDrive folder '%s'", total, folder_label)
        for index, item in enumerate(items, start=1):
            if self._state.is_processed(item.item_id):
                continue
            try:
                metadata = self._process_item(item, index=index, total=total)
            except Exception as exc:  # pragma: no cover - fail fast during runtime
                _LOGGER.exception("Failed to process %s: %s", item.name, exc)
                self._state.record_failure(
                    item_id=item.item_id,
                    filename=item.name,
                    model=self._settings.ollama.model,
                    error=str(exc),
                )
                continue
            self._state.record_success(
                item_id=item.item_id,
                filename=item.name,
                model=self._settings.ollama.model,
                metadata=metadata,
            )
            processed += 1
        return processed

    def _process_item(
        self,
        item,
        *,
        index: Optional[int] = None,
        total: Optional[int] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> dict:
        position = self._format_position(index, total)
        _LOGGER.info("Processing %s%s", position, item.name)
        self._notify_progress(progress_callback, "download")
        self._log_progress(position, item.name, "download")
        pdf_bytes = self._onedrive.download_item(item)
        self._notify_progress(progress_callback, "convert")
        self._log_progress(position, item.name, "convert")
        png_pages = pdf_to_png_pages(pdf_bytes, max_pages=1)
        if not png_pages:
            raise RuntimeError("PDF conversion returned no pages")
        image_bytes = self._compress_image_if_needed(png_pages[0], item.name)
        size_mb = len(image_bytes) / (1024 * 1024)
        _LOGGER.info("%s%s: preparing metadata request (image %.2f MB)", position, item.name, size_mb)
        self._notify_progress(progress_callback, "metadata")
        self._log_progress(position, item.name, "metadata")
        metadata = self._ollama.request_metadata(image_bytes)
        self._notify_progress(progress_callback, "embed")
        self._log_progress(position, item.name, "embed")
        enriched_pdf = embed_metadata(pdf_bytes, metadata)
        self._notify_progress(progress_callback, "upload")
        self._log_progress(position, item.name, "upload")
        self._onedrive.upload_pdf_to_sorted(item.name, enriched_pdf)
        self._notify_progress(progress_callback, "done")
        self._log_progress(position, item.name, "done")
        return metadata

    def _inbox_folder_label(self) -> str:
        raw = (self._settings.graph.inbox_folder or "").strip()
        cleaned = raw.strip("/")
        return cleaned or "/"

    @staticmethod
    def _format_position(index: Optional[int], total: Optional[int]) -> str:
        if index is None or total is None or total == 0:
            return ""
        return f"[{index}/{total}] "

    def _log_progress(self, position: str, item_name: str, stage: str) -> None:
        messages = {
            "download": "downloading from OneDrive",
            "convert": "converting to image",
            "metadata": "requesting metadata",
            "embed": "embedding metadata",
            "upload": "uploading to sorted folder",
            "done": "completed",
        }
        detail = messages.get(stage, stage)
        _LOGGER.info("%s%s: %s", position, item_name, detail)

    @staticmethod
    def _notify_progress(callback: Optional[Callable[[str], None]], stage: str) -> None:
        if callback is not None:
            try:
                callback(stage)
            except Exception:  # pragma: no cover - defensive
                pass

    def _compress_image_if_needed(self, image_bytes: bytes, item_name: str) -> bytes:
        max_bytes = 4 * 1024 * 1024  # 4 MB limit
        if len(image_bytes) <= max_bytes:
            return image_bytes
        try:
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as exc:  # pragma: no cover - defensive
            _LOGGER.warning("Could not open image for %s to downscale: %s", item_name, exc)
            return image_bytes

        current_width, current_height = image.size
        scale = 0.85
        while len(image_bytes) > max_bytes and current_width > 512 and current_height > 512:
            new_width = max(int(current_width * scale), 512)
            new_height = max(int(current_height * scale), 512)
            image = image.resize((new_width, new_height), Image.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", optimize=True)
            image_bytes = buffer.getvalue()
            current_width, current_height = image.size

        if len(image_bytes) > max_bytes:
            _LOGGER.warning(
                "Image for %s remains large after compression (%.2f MB). Proceeding anyway.",
                item_name,
                len(image_bytes) / (1024 * 1024),
            )
        else:
            _LOGGER.info(
                "Downscaled image for %s to %.2f MB (%dx%d).",
                item_name,
                len(image_bytes) / (1024 * 1024),
                current_width,
                current_height,
            )
        return image_bytes


def run(settings: Optional[PipelineSettings] = None) -> int:
    """Helper to load settings and run the pipeline once."""
    if settings is None:
        settings = load_settings()
    if not settings.auto_process_inbox:
        _LOGGER.info("Auto-processing is disabled; exiting without processing inbox.")
        return 0
    pipe = Pipeline(settings)
    pipe.run_startup_checks()
    interval = settings.poll_interval_seconds or 0
    if interval <= 0:
        return pipe.run_once()

    total_processed = 0
    try:
        while True:
            processed = pipe.run_once()
            total_processed += processed
            _LOGGER.info("Sleeping %d second(s) before next poll", interval)
            time.sleep(interval)
    except KeyboardInterrupt:  # pragma: no cover - manual stop
        _LOGGER.info("Polling interrupted by user")
        return total_processed
