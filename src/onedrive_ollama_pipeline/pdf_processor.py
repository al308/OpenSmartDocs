"""PDF helper utilities for splitting pages and embedding metadata."""
from __future__ import annotations

import io
import json
import os
from typing import Iterable, Mapping, Sequence

from pypdf import PdfReader

from pdf2image import convert_from_bytes
import pikepdf


def pdf_to_png_pages(pdf_bytes: bytes, dpi: int = 200, max_pages: int | None = None) -> list[bytes]:
    """Convert a PDF into PNG bytes per page."""
    poppler_path = os.getenv("POPPLER_PATH")
    convert_kwargs = {"dpi": dpi, "fmt": "png"}
    if poppler_path:
        # Allow running with a project-local Poppler build instead of a system install.
        convert_kwargs["poppler_path"] = poppler_path
    images = convert_from_bytes(pdf_bytes, **convert_kwargs)
    png_pages: list[bytes] = []
    for index, image in enumerate(images):
        if max_pages is not None and index >= max_pages:
            break
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        png_pages.append(buffer.getvalue())
    return png_pages


def extract_pdf_text(
    pdf_bytes: bytes,
    *,
    max_pages: int | None = 5,
    max_chars: int = 4000,
) -> str:
    """Extract plain text from the first pages of a PDF, if available."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception:
        return ""

    text_parts: list[str] = []
    collected = 0
    for index, page in enumerate(reader.pages):
        if max_pages is not None and index >= max_pages:
            break
        try:
            page_text = page.extract_text() or ""
        except Exception:
            continue
        page_text = page_text.strip()
        if not page_text:
            continue
        remaining = max_chars - collected
        if remaining <= 0:
            break
        if len(page_text) > remaining:
            page_text = page_text[:remaining]
        text_parts.append(page_text)
        collected += len(page_text)
        if collected >= max_chars:
            break
    return "\n\n".join(text_parts).strip()


def extract_pdf_metadata_fields(pdf_bytes: bytes) -> dict[str, str]:
    """Return PDF document info entries as a flat mapping."""
    try:
        with pikepdf.open(io.BytesIO(pdf_bytes)) as pdf:
            docinfo = pdf.docinfo
            if not docinfo:
                return {}
            results: dict[str, str] = {}
            for key, value in docinfo.items():
                key_name = str(key)
                if key_name.startswith("/"):
                    key_name = key_name[1:]
                if value is None:
                    continue
                text = str(value).strip()
                if text:
                    results[key_name] = text
            return results
    except Exception:
        return {}


def inspect_pdf_content(
    pdf_bytes: bytes,
    *,
    text_max_pages: int | None = 5,
    text_max_chars: int = 4000,
    text_preview_chars: int = 500,
) -> dict[str, object]:
    """Analyze a PDF for embedded metadata and extractable text."""
    text = extract_pdf_text(pdf_bytes, max_pages=text_max_pages, max_chars=text_max_chars)
    metadata_fields = extract_pdf_metadata_fields(pdf_bytes)
    preview = text[:text_preview_chars]
    return {
        "text": {
            "available": bool(text),
            "chars": len(text),
            "preview": preview,
        },
        "metadata": {
            "available": bool(metadata_fields),
            "fields": metadata_fields,
        },
    }


def embed_metadata(pdf_bytes: bytes, metadata: Mapping[str, object]) -> bytes:
    """Embed structured metadata into a PDF's document info dictionary."""
    json_blob = json.dumps(metadata, ensure_ascii=False)
    with pikepdf.open(io.BytesIO(pdf_bytes)) as pdf:
        docinfo = pdf.docinfo
        title = metadata.get("title")
        if isinstance(title, str) and title:
            docinfo["/Title"] = title
        author = metadata.get("author")
        if isinstance(author, str) and author:
            docinfo["/Author"] = author
        summary = metadata.get("summary")
        if isinstance(summary, str) and summary:
            docinfo["/Subject"] = summary
        tags = metadata.get("tags")
        if isinstance(tags, Sequence):
            docinfo["/Keywords"] = ", ".join(str(tag) for tag in tags if tag)
        docinfo["/Producer"] = "onedrive-ollama-pipeline"
        docinfo["/OllamaMetadata"] = json_blob
        output = io.BytesIO()
        pdf.save(output)
    return output.getvalue()
