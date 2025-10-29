from pathlib import Path

from onedrive_ollama_pipeline import pipeline
from onedrive_ollama_pipeline.config import GraphSettings, OllamaSettings, PipelineSettings


class DummyItem:
    def __init__(self, item_id: str, name: str):
        self.item_id = item_id
        self.name = name
        self.download_url = "https://example.com/file"
        self.web_url = "https://example.com/web"


class FakeOneDrive:
    def __init__(self, settings):
        self.uploaded = []
        self._items = [DummyItem("1", "file.pdf")]

    def list_pdfs_in_inbox(self):
        yield from self._items

    def download_item(self, item):
        return b"%PDF-1.5 fake pdf"

    def upload_pdf_to_sorted(self, filename, content):
        self.uploaded.append((filename, content))
        return {"id": "uploaded"}


class FakeOllama:
    def __init__(self, settings):
        pass

    def request_metadata(self, image_bytes):
        assert image_bytes == b"png"
        return {"title": "Test", "author": "Unit"}


def test_pipeline_run_once(monkeypatch, tmp_path: Path):
    settings = PipelineSettings(
        graph=GraphSettings(
            tenant_id="tenant",
            client_id="client",
            client_secret="secret",
            user_id="user",
            token_cache_path=tmp_path / "token_cache.json",
        ),
        ollama=OllamaSettings(model="test"),
        db_path=tmp_path / "state.db",
        log_path=tmp_path / "pipeline.log",
        config_path=tmp_path / "config.json",
        auto_process_inbox=True,
    )

    fake_onedrive = FakeOneDrive(settings.graph)

    def fake_onedrive_factory(_):
        return fake_onedrive

    def fake_ollama_factory(_):
        return FakeOllama(settings.ollama)

    monkeypatch.setattr(pipeline, "OneDriveClient", fake_onedrive_factory)
    monkeypatch.setattr(pipeline, "OllamaClient", fake_ollama_factory)
    monkeypatch.setattr(pipeline, "pdf_to_png_pages", lambda pdf, max_pages=1: [b"png"])
    monkeypatch.setattr(pipeline, "embed_metadata", lambda pdf, metadata: b"updated-pdf")

    pipe = pipeline.Pipeline(settings)
    processed = pipe.run_once()
    assert processed == 1
    assert fake_onedrive.uploaded[0][0] == "file.pdf"


def test_run_executes_startup_checks(monkeypatch, tmp_path: Path):
    settings = PipelineSettings(
        graph=GraphSettings(
            tenant_id="tenant",
            client_id="client",
            client_secret="secret",
            user_id="user",
            token_cache_path=tmp_path / "token_cache.json",
        ),
        ollama=OllamaSettings(model="test"),
        db_path=tmp_path / "state.db",
        log_path=tmp_path / "pipeline.log",
        config_path=tmp_path / "config.json",
        auto_process_inbox=True,
    )

    startup_ran = {"value": False}

    def fake_init(self, _settings):
        self._settings = _settings

    def fake_startup(self):
        startup_ran["value"] = True

    def fake_run_once(self):
        return 0

    monkeypatch.setattr(pipeline.Pipeline, "__init__", fake_init, raising=False)
    monkeypatch.setattr(pipeline.Pipeline, "run_startup_checks", fake_startup, raising=False)
    monkeypatch.setattr(pipeline.Pipeline, "run_once", fake_run_once, raising=False)

    processed = pipeline.run(settings)
    assert processed == 0
    assert startup_ran["value"] is True


def test_run_skips_when_auto_processing_disabled(monkeypatch, tmp_path: Path):
    settings = PipelineSettings(
        graph=GraphSettings(
            tenant_id="tenant",
            client_id="client",
            client_secret="secret",
            user_id="user",
            token_cache_path=tmp_path / "token_cache.json",
        ),
        ollama=OllamaSettings(model="test"),
        db_path=tmp_path / "state.db",
        log_path=tmp_path / "pipeline.log",
        config_path=tmp_path / "config.json",
        auto_process_inbox=False,
    )

    processed = pipeline.run(settings)
    assert processed == 0
