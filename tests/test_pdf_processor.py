import io

import pikepdf
import pytest
from PIL import Image

from onedrive_ollama_pipeline import pdf_processor


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(595, 842))
    buffer = io.BytesIO()
    pdf.save(buffer)
    return buffer.getvalue()


def test_pdf_to_png_pages(monkeypatch, sample_pdf_bytes):
    image = Image.new("RGB", (10, 10), color="white")

    def fake_convert_from_bytes(_: bytes, dpi: int, fmt: str):  # noqa: D401 - simple stub
        assert dpi == 200
        assert fmt == "png"
        return [image]

    monkeypatch.setattr(pdf_processor, "convert_from_bytes", fake_convert_from_bytes)
    pages = pdf_processor.pdf_to_png_pages(sample_pdf_bytes)
    assert len(pages) == 1
    assert pages[0].startswith(b"\x89PNG")


def test_embed_metadata(sample_pdf_bytes):
    updated = pdf_processor.embed_metadata(
        sample_pdf_bytes,
        {
            "title": "Invoice 123",
            "author": "ACME",
            "summary": "Q1 invoice",
            "tags": ["invoice", "q1"],
        },
    )
    with pikepdf.open(io.BytesIO(updated)) as pdf:
        docinfo = pdf.docinfo
        assert docinfo["/Title"] == "Invoice 123"
        assert docinfo["/Author"] == "ACME"
        assert docinfo["/Subject"] == "Q1 invoice"
        keywords = docinfo.get("/Keywords")
        assert keywords is not None and "invoice" in str(keywords)
        assert docinfo["/OllamaMetadata"]
