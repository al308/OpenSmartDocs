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


def test_extract_pdf_text(monkeypatch):
    class DummyPage:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class DummyReader:
        def __init__(self, _stream):
            self.pages = [DummyPage("page one text"), DummyPage("page two text")]

    monkeypatch.setattr(pdf_processor, "PdfReader", DummyReader)

    text = pdf_processor.extract_pdf_text(b"pdf-bytes", max_pages=1, max_chars=50)
    assert "page one text" in text


def test_extract_pdf_text_truncates(monkeypatch):
    class DummyPage:
        def extract_text(self):
            return "x" * 100

    class DummyReader:
        def __init__(self, _stream):
            self.pages = [DummyPage(), DummyPage()]

    monkeypatch.setattr(pdf_processor, "PdfReader", DummyReader)

    text = pdf_processor.extract_pdf_text(b"pdf-bytes", max_pages=5, max_chars=150)
    segments = text.split("\n\n")
    assert len(segments[0]) == 100
    assert len(segments[1]) == 50


def test_extract_pdf_metadata_fields(sample_pdf_bytes):
    with pikepdf.open(io.BytesIO(sample_pdf_bytes)) as pdf:
        pdf.docinfo["/Title"] = "Sample Title"
        pdf.docinfo["/Author"] = "Jane"
        buffer = io.BytesIO()
        pdf.save(buffer)
        updated_bytes = buffer.getvalue()

    fields = pdf_processor.extract_pdf_metadata_fields(updated_bytes)
    assert fields["Title"] == "Sample Title"
    assert fields["Author"] == "Jane"


def test_inspect_pdf_content(monkeypatch, sample_pdf_bytes):
    monkeypatch.setattr(pdf_processor, "extract_pdf_text", lambda data, max_pages=5, max_chars=4000: "hello world")
    monkeypatch.setattr(pdf_processor, "extract_pdf_metadata_fields", lambda data: {"Title": "Doc"})

    report = pdf_processor.inspect_pdf_content(sample_pdf_bytes, text_preview_chars=4)
    assert report["text"]["available"] is True
    assert report["text"]["preview"] == "hell"
    assert report["metadata"]["fields"]["Title"] == "Doc"
