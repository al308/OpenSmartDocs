"""PDF helper utilities for splitting pages and embedding metadata."""
from __future__ import annotations

import io
import json
import os
from typing import Iterable, Mapping, Sequence

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
