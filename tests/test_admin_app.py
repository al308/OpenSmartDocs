import json
from pathlib import Path

from fastapi import HTTPException
from fastapi.testclient import TestClient

from onedrive_ollama_pipeline.admin_app import app
from onedrive_ollama_pipeline.state_store import StateStore


def setup_env(monkeypatch, tmp_path: Path) -> tuple[Path, Path, Path]:
    db_path = tmp_path / "state.db"
    log_path = tmp_path / "pipeline.log"
    config_path = tmp_path / "config.json"

    config_path.write_text(
        json.dumps(
            {
                "graph": {
                    "inbox_folder": "_inbox",
                    "sorted_folder": "_sorted",
                },
                "ollama": {
                    "model": "model",
                    "metadata_prompt": "prompt",
                },
                "pipeline": {
                    "poll_interval_seconds": 0,
                    "log_level": "INFO",
                    "auto_process_inbox": False,
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("PIPELINE_DB_PATH", str(db_path))
    monkeypatch.setenv("PIPELINE_LOG_PATH", str(log_path))
    monkeypatch.setenv("PIPELINE_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("GRAPH_TENANT_ID", "tenant")
    monkeypatch.setenv("GRAPH_CLIENT_ID", "client")
    monkeypatch.setenv("GRAPH_USER_ID", "user")
    return db_path, log_path, config_path


def test_status_and_config_endpoints(monkeypatch, tmp_path):
    db_path, log_path, config_path = setup_env(monkeypatch, tmp_path)
    store = StateStore(db_path)
    store.record_success(item_id="123", filename="file.pdf", model="demo", metadata={"key": "value"})

    log_path.write_text("info: pipeline started\n", encoding="utf-8")

    client = TestClient(app)

    status = client.get("/api/status").json()
    assert status["counts"]["total"] == 1
    assert status["recent"][0]["filename"] == "file.pdf"

    config = client.get("/api/config").json()
    assert config["ollama_model"] == "model"
    assert config["auto_process_inbox"] is False
    assert config["structure_model"] == "model"

    update_payload = {
        "ollamaModel": "new-model",
        "pollIntervalSeconds": 5,
        "inboxFolder": "Inbox",
        "sortedFolder": "Sorted",
        "logLevel": "WARNING",
        "autoProcessInbox": True,
        "structureModel": "custom-structure",
    }
    updated = client.put("/api/config", json=update_payload)
    assert updated.status_code == 200

    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    assert config_data["ollama"]["model"] == "new-model"
    assert config_data["graph"]["inbox_folder"] == "Inbox"
    assert config_data["pipeline"]["poll_interval_seconds"] == 5
    assert config_data["pipeline"]["log_level"] == "WARNING"
    assert config_data["pipeline"]["auto_process_inbox"] is True
    assert config_data["structure"]["model"] == "custom-structure"

    sql_response = client.post("/api/query", json={"sql": "SELECT filename FROM processed_items"}).json()
    assert sql_response["rows"][0]["filename"] == "file.pdf"

    logs = client.get("/api/logs").json()
    assert logs["lines"][-1].startswith("info")


def test_regular_ingest_upload(monkeypatch, tmp_path):
    setup_env(monkeypatch, tmp_path)
    uploaded = {}

    class DummyClient:
        def upload_pdf_to_inbox(self, filename, content):
            uploaded["filename"] = filename
            uploaded["content"] = content
            return {"id": "item123", "name": filename, "@microsoft.graph.downloadUrl": "https://download"}

    monkeypatch.setattr("onedrive_ollama_pipeline.admin_app._get_onedrive_client", DummyClient)
    monkeypatch.setattr(
        "onedrive_ollama_pipeline.admin_app.inspect_pdf_content",
        lambda content: {
            "text": {"available": True, "chars": 120, "preview": "Sample text"},
            "metadata": {"available": True, "fields": {"Title": "Doc"}},
        },
    )

    client = TestClient(app)
    response = client.post(
        "/api/ingest/upload",
        files={"file": ("My Doc.PDF", b"%PDF-1.4", "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "My_Doc.pdf"
    assert data["analysis"]["text"]["available"] is True
    assert data["analysis"]["metadata"]["available"] is True
    assert data["recommendedStrategy"] == "text"
    assert data["item"]["itemId"] == "item123"
    assert uploaded["filename"] == "My_Doc.pdf"
    assert uploaded["content"] == b"%PDF-1.4"


def test_mass_scan_ingest_success(monkeypatch, tmp_path):
    setup_env(monkeypatch, tmp_path)
    uploaded = {}

    class DummyClient:
        def upload_pdf_to_inbox(self, filename, content):
            uploaded["filename"] = filename
            uploaded["content"] = content

    def fake_client():
        return DummyClient()

    monkeypatch.setattr("onedrive_ollama_pipeline.admin_app._get_onedrive_client", fake_client)
    monkeypatch.setattr(
        "onedrive_ollama_pipeline.admin_app._combine_duplex_pdfs",
        lambda odd, even: (b"merged-pdf", 6),
    )

    client = TestClient(app)
    response = client.post(
        "/api/ingest/mass-scan",
        files={
            "oddFile": ("odd.pdf", b"%PDF odd", "application/pdf"),
            "evenFile": ("even.pdf", b"%PDF even", "application/pdf"),
        },
        data={"outputName": "final"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "final.pdf"
    assert data["pages"] == 6
    assert uploaded["filename"] == "final.pdf"
    assert uploaded["content"] == b"merged-pdf"


def test_mass_scan_page_mismatch(monkeypatch, tmp_path):
    setup_env(monkeypatch, tmp_path)

    def raise_mismatch(odd, even):
        raise HTTPException(status_code=400, detail="Mismatch")

    def unexpected_client():
        raise AssertionError("Client should not be requested when merge fails")

    monkeypatch.setattr("onedrive_ollama_pipeline.admin_app._combine_duplex_pdfs", raise_mismatch)
    monkeypatch.setattr("onedrive_ollama_pipeline.admin_app._get_onedrive_client", unexpected_client)

    client = TestClient(app)
    response = client.post(
        "/api/ingest/mass-scan",
        files={
            "oddFile": ("odd.pdf", b"%PDF odd", "application/pdf"),
            "evenFile": ("even.pdf", b"%PDF even", "application/pdf"),
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Mismatch"


def test_ingest_process(monkeypatch, tmp_path):
    setup_env(monkeypatch, tmp_path)
    created = {}

    class DummyState:
        def __init__(self):
            self.success = []
            self.failure = []

        def record_success(self, **kwargs):
            self.success.append(kwargs)

        def record_failure(self, **kwargs):
            self.failure.append(kwargs)

    class DummyPipeline:
        def __init__(self, settings):
            self._settings = settings
            self._state = DummyState()
            created["pipeline"] = self

        def _process_item(self, item, *, progress_callback=None, metadata_strategy="auto"):
            created["mode"] = metadata_strategy
            if progress_callback:
                progress_callback("metadata")
            return {"title": "ok"}

    monkeypatch.setattr("onedrive_ollama_pipeline.admin_app.Pipeline", DummyPipeline)

    client = TestClient(app)
    response = client.post(
        "/api/ingest/process",
        json={
            "itemId": "item123",
            "driveId": "drive",
            "name": "file.pdf",
            "mode": "text",
            "downloadUrl": "https://download",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "text"
    assert data["metadata"]["title"] == "ok"
    assert created["mode"] == "text"
    assert created["pipeline"]._state.success[0]["item_id"] == "item123"


def test_structure_sources_endpoint(monkeypatch, tmp_path):
    setup_env(monkeypatch, tmp_path)

    class DummyStructureService:
        def list_sources(self, max_items=None):
            return {
                "sources": [
                    {
                        "id": "SRC001",
                        "relative_path": "Doc.pdf",
                        "name": "Doc.pdf",
                        "folder": "",
                    }
                ]
            }

        def get_state(self):
            return {"plan": None, "applied": None, "log": []}

    monkeypatch.setattr("onedrive_ollama_pipeline.admin_app._get_structure_service", lambda: DummyStructureService())

    client = TestClient(app)
    response = client.get("/api/structure/sources")
    assert response.status_code == 200
    data = response.json()
    assert data["sources"][0]["name"] == "Doc.pdf"


def test_structure_analyze_accepts_selection(monkeypatch, tmp_path):
    setup_env(monkeypatch, tmp_path)
    captured = {}

    class DummyStructureService:
        def analyze(self, include_relative_paths=None):
            captured["include"] = include_relative_paths
            return {"plan": {"operations": []}}

        def get_state(self):
            return {"plan": None, "applied": None, "log": []}

    monkeypatch.setattr("onedrive_ollama_pipeline.admin_app._get_structure_service", lambda: DummyStructureService())

    client = TestClient(app)
    response = client.post("/api/structure/analyze", json={"relativePaths": ["Doc.pdf", "Other.pdf"]})
    assert response.status_code == 200
    assert captured["include"] == {"Doc.pdf", "Other.pdf"}


def test_inbox_preview_endpoint(monkeypatch, tmp_path):
    setup_env(monkeypatch, tmp_path)

    class DummyItem:
        def __init__(self, item_id: str, name: str):
            self.item_id = item_id
            self.name = name

    class DummyClient:
        def list_pdfs_in_inbox(self):
            yield DummyItem("item-1", "sample.pdf")

        def download_item(self, item):
            assert item.item_id == "item-1"
            return b"%PDF-1.4"

    monkeypatch.setattr("onedrive_ollama_pipeline.admin_app._get_onedrive_client", lambda: DummyClient())
    monkeypatch.setattr(
        "onedrive_ollama_pipeline.admin_app.inspect_pdf_content",
        lambda content, **_kwargs: {"text": {"available": True, "chars": 600}},
    )
    monkeypatch.setattr("onedrive_ollama_pipeline.admin_app", "_TEXT_INFO_CACHE", {})

    client = TestClient(app)
    response = client.get("/api/inbox/preview/item-1")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")


def test_inbox_text_info_endpoint(monkeypatch, tmp_path):
    setup_env(monkeypatch, tmp_path)

    class DummyItem:
        def __init__(self, item_id: str, name: str):
            self.item_id = item_id
            self.name = name

    class DummyClient:
        def list_pdfs_in_inbox(self):
            yield DummyItem("item-1", "sample.pdf")

        def download_item(self, item):
            assert item.item_id == "item-1"
            return b"%PDF-1.4"

    monkeypatch.setattr("onedrive_ollama_pipeline.admin_app._get_onedrive_client", lambda: DummyClient())
    monkeypatch.setattr(
        "onedrive_ollama_pipeline.admin_app.inspect_pdf_content",
        lambda content, **_kwargs: {"text": {"available": True, "chars": 500}},
    )
    monkeypatch.setattr("onedrive_ollama_pipeline.admin_app", "_TEXT_INFO_CACHE", {})

    client = TestClient(app)
    response = client.get("/api/inbox/text-info/item-1")
    assert response.status_code == 200
    assert response.json()["hasText"] is True


def test_structure_preview_endpoint(monkeypatch, tmp_path):
    setup_env(monkeypatch, tmp_path)

    class DummyClient:
        def download_sorted_file(self, relative_path: str):
            class Entry:
                name = "sorted.pdf"

            assert relative_path == "sorted.pdf"
            return Entry(), b"%PDF-1.6"

        def list_pdfs_in_inbox(self):  # not used but keep interface consistent
            return iter([])

    monkeypatch.setattr("onedrive_ollama_pipeline.admin_app._get_onedrive_client", lambda: DummyClient())
    monkeypatch.setattr(
        "onedrive_ollama_pipeline.admin_app.inspect_pdf_content",
        lambda content, **_kwargs: {"text": {"available": False, "chars": 80}},
    )
    monkeypatch.setattr("onedrive_ollama_pipeline.admin_app", "_TEXT_INFO_CACHE", {})

    client = TestClient(app)
    response = client.get("/api/structure/preview", params={"relative_path": "sorted.pdf"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")


def test_structure_text_info_endpoint(monkeypatch, tmp_path):
    setup_env(monkeypatch, tmp_path)

    class DummyClient:
        def download_sorted_file(self, relative_path: str):
            class Entry:
                name = "sorted.pdf"

            assert relative_path == "sorted.pdf"
            return Entry(), b"%PDF-1.6"

        def list_pdfs_in_inbox(self):  # unused
            return iter([])

    monkeypatch.setattr("onedrive_ollama_pipeline.admin_app._get_onedrive_client", lambda: DummyClient())
    monkeypatch.setattr(
        "onedrive_ollama_pipeline.admin_app.inspect_pdf_content",
        lambda content, **_kwargs: {"text": {"available": False, "chars": 50}},
    )
    monkeypatch.setattr("onedrive_ollama_pipeline.admin_app", "_TEXT_INFO_CACHE", {})

    client = TestClient(app)
    response = client.get("/api/structure/text-info", params={"relative_path": "sorted.pdf"})
    assert response.status_code == 200
    assert response.json()["hasText"] is False


def test_ollama_connection_test(monkeypatch, tmp_path):
    _, _, config_path = setup_env(monkeypatch, tmp_path)
    config_data = json.loads(config_path.read_text(encoding='utf-8'))
    config_data['structure'] = {'model': 'structure-model'}
    config_path.write_text(json.dumps(config_data), encoding='utf-8')

    calls = []

    class DummyClient:
        def __init__(self, settings):
            pass

        def test_model(self, *, model, prompt, expected, timeout=60):
            calls.append(model)

    monkeypatch.setattr('onedrive_ollama_pipeline.admin_app.OllamaClient', DummyClient)

    client = TestClient(app)
    response = client.post('/api/ollama/test')
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'ok'
    assert len(data['results']) == 2
    assert calls == ['model', 'structure-model']
