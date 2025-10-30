import json
import pytest
from pathlib import Path
from typing import Optional

from onedrive_ollama_pipeline.config import GraphSettings, OllamaSettings, PipelineSettings
from onedrive_ollama_pipeline.database import get_database
from onedrive_ollama_pipeline.onedrive_client import SortedTreeEntry
from onedrive_ollama_pipeline.structure_service import (
    StructureCache,
    StructureContext,
    StructureService,
    StructureServiceError,
    StructureSource,
)


def build_settings(tmp_path: Path) -> PipelineSettings:
    db_path = tmp_path / "state.db"
    log_path = tmp_path / "pipeline.log"
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    graph = GraphSettings(
        tenant_id="tenant",
        client_id="client",
        client_secret=None,
        user_id="user",
        inbox_folder="_inbox",
        sorted_folder="_sorted",
    )
    ollama = OllamaSettings(
        base_url="http://localhost:11434/v1",
        model="llama3",
        metadata_prompt="prompt",
    )
    return PipelineSettings(
        graph=graph,
        ollama=ollama,
        db_path=db_path,
        log_path=log_path,
        config_path=config_path,
        log_level="INFO",
        poll_interval_seconds=None,
        structure_model="test-structure-model",
    )


class StubOnedrive:
    def __init__(self, tree: list[SortedTreeEntry]):
        self.tree = tree
        self.created_folders: list[str] = []
        self.download_requests: list[str] = []
        self.uploads: list[tuple[str, str, bytes]] = []
        self.deleted: list[str] = []
        self.moves: list[tuple[str, str, Optional[str]]] = []

    def walk_sorted_tree(self, **_kwargs):
        return list(self.tree)

    def ensure_sorted_subfolder(self, path: str):
        self.created_folders.append(path)
        return SortedTreeEntry(
            path=path,
            item_id=f"folder-{path}",
            drive_id=None,
            name=Path(path).name or path,
            is_folder=True,
            download_url=None,
            web_url=None,
            base_path="https://graph.microsoft.com/v1.0/me/drive",
        )

    def download_sorted_file(self, relative_path: str):
        self.download_requests.append(relative_path)
        entry = SortedTreeEntry(
            path=relative_path,
            item_id=f"item-{relative_path}",
            drive_id=None,
            name=Path(relative_path).name,
            is_folder=False,
            download_url=None,
            web_url=None,
            base_path="https://graph.microsoft.com/v1.0/me/drive",
        )
        return entry, b"pdf-bytes"

    def upload_bytes_to_sorted(self, filename: str, content: bytes, *, folder_path: str = ""):
        self.uploads.append((folder_path, filename, content))
        return {"name": filename, "folder": folder_path}

    def delete_sorted_path(self, relative_path: str):
        self.deleted.append(relative_path)

    def move_sorted_item(self, source_relative_path: str, target_folder: str, target_name: Optional[str] = None):
        self.moves.append((source_relative_path, target_folder, target_name))


class FailingOnedrive(StubOnedrive):
    def __init__(self, error: Exception):
        super().__init__(tree=[])
        self._error = error

    def walk_sorted_tree(self, **_kwargs):
        raise self._error


class StubOllama:
    def __init__(
        self,
        payload: dict | None = None,
        raw_output: str | None = None,
        plan_responses: list[str | dict] | None = None,
        validation_response: dict | str | None = None,
    ):
        self.plan_payload = payload or {}
        self.raw_output = raw_output
        self.plan_responses = list(plan_responses or [])
        self.validation_response = validation_response or {"status": "ok", "notes": "Looks good.", "issues": []}
        self.calls: list[dict[str, object]] = []

    def request_structure_plan(
        self,
        prompt: str,
        *,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        json_schema: dict | None = None,
        schema_name: str = "StructurePlan",
        system_prompt: str | None = None,
    ):
        return self.request_json_schema_completion(
            system_prompt=system_prompt or "",
            prompt=prompt,
            model=model,
            json_schema=json_schema,
            schema_name=schema_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def request_json_schema_completion(
        self,
        *,
        system_prompt: str,
        prompt: str,
        model: str,
        json_schema: dict | None,
        schema_name: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        timeout: int = 180,
    ):
        call = {
            "system_prompt": system_prompt,
            "prompt": prompt,
            "model": model,
            "json_schema": json_schema,
            "schema_name": schema_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": timeout,
        }
        self.calls.append(call)
        if schema_name == "StructurePlan":
            if self.plan_responses:
                response = self.plan_responses.pop(0)
                if isinstance(response, dict):
                    return json.dumps(response)
                return response
            if self.raw_output is not None:
                return self.raw_output
            return json.dumps(self.plan_payload)
        if schema_name == "StructureValidation":
            response = self.validation_response
            if isinstance(response, dict):
                return json.dumps(response)
            return response
        return "{}"


def test_structure_source_extracts_dates_from_metadata():
    source = StructureSource(
        source_id="SRC001",
        relative_path="example.pdf",
        name="example.pdf",
        metadata={"document_date": "05.03.2024", "document_type": "Tax document"},
    )
    assert source.document_date() == "2024-03-05"
    assert source.document_year() == "2024"
    target = source.suggested_target_name()
    assert target.startswith("2024-03-05")
    assert target.endswith(".pdf")


def test_structure_source_extracts_dates_from_summary_text():
    source = StructureSource(
        source_id="SRC002",
        relative_path="letter.pdf",
        name="letter.pdf",
        metadata={"summary": "This letter was issued on 7 April 2023.", "document_type": "Insurance"},
    )
    assert source.document_date() == "2023-04-07"
    assert source.document_year() == "2023"
    assert source.suggested_folder() == "Finance/Insurance/2023"


def test_structure_source_actor_included_in_title():
    source = StructureSource(
        source_id="SRC010",
        relative_path="invoice.pdf",
        name="invoice.pdf",
        metadata={
            "document_date": "2025-02-01",
            "title": "Invoice from ACME Corp",
            "issuer": "ACME Corp",
            "document_type": "Invoice",
        },
        locale="de",
    )
    target = source.suggested_target_name()
    assert target.startswith("2025-02-01")
    assert "ACME" in target
    assert "Rechnung" in target


def test_structure_source_government_folder():
    source = StructureSource(
        source_id="SRC011",
        relative_path="letter.pdf",
        name="letter.pdf",
        metadata={
            "document_date": "2024-05-10",
            "document_type": "Official Government Letter",
        },
        locale="de",
    )
    assert source.suggested_folder() == "Verwaltung/Behoerden/2024"


def test_analyze_without_sources(tmp_path):
    settings = build_settings(tmp_path)
    cache = StructureCache(tmp_path / "structure_state.json")
    onedrive = StubOnedrive(tree=[])
    ollama = StubOllama({"summary": "nothing to do", "operations": []})
    service = StructureService(
        settings=settings,
        cache=cache,
        onedrive_client=onedrive,
        ollama_client=ollama,
    )

    plan = service.analyze()
    assert plan["operations"] == []
    assert "No eligible files" in plan["summary"]
    state = cache.read()
    assert state["plan"]["summary"] == plan["summary"]
    assert service._ollama.calls == []  # LLM not invoked when there are no sources


def test_analyze_with_selected_sources(tmp_path):
    settings = build_settings(tmp_path)
    cache = StructureCache(tmp_path / "structure_state.json")
    onedrive = StubOnedrive(
        tree=[
            SortedTreeEntry(
                path="alpha.pdf",
                item_id="file-alpha",
                drive_id=None,
                name="alpha.pdf",
                is_folder=False,
                download_url="https://download/alpha",
                web_url=None,
                base_path="https://graph.microsoft.com/v1.0/me/drive",
            ),
            SortedTreeEntry(
                path="beta.pdf",
                item_id="file-beta",
                drive_id=None,
                name="beta.pdf",
                is_folder=False,
                download_url="https://download/beta",
                web_url=None,
                base_path="https://graph.microsoft.com/v1.0/me/drive",
            ),
        ]
    )
    ollama = StubOllama({"summary": "ok", "operations": []})
    service = StructureService(
        settings=settings,
        cache=cache,
        onedrive_client=onedrive,
        ollama_client=ollama,
    )

    plan = service.analyze(include_relative_paths={"beta.pdf"})
    paths = [source["relative_path"] for source in plan["context"]["sources"]]
    assert paths == ["beta.pdf"]


def test_apply_and_revert_flow(tmp_path):
    settings = build_settings(tmp_path)
    cache = StructureCache(tmp_path / "structure_state.json")
    db = get_database(settings.db_path)
    db.execute(
        """
        INSERT INTO processed_items (onedrive_id, filename, processed_at, status, model, metadata_json, error_message, run_id)
        VALUES (?, ?, datetime('now'), 'success', 'model', ?, NULL, NULL)
        """,
        (
            "item-1",
            "invoice.pdf",
            json.dumps({"document_type": "Tax document", "document_date": "2023-01-15"}),
        ),
    )
    onedrive = StubOnedrive(
        tree=[
            SortedTreeEntry(
                path="2023",
                item_id="folder-2023",
                drive_id=None,
                name="2023",
                is_folder=True,
                download_url=None,
                web_url=None,
                base_path="https://graph.microsoft.com/v1.0/me/drive",
            ),
            SortedTreeEntry(
                path="invoice.pdf",
                item_id="file-1",
                drive_id=None,
                name="invoice.pdf",
                is_folder=False,
                download_url="https://download",
                web_url=None,
                base_path="https://graph.microsoft.com/v1.0/me/drive",
            ),
        ]
    )
    ollama = StubOllama(
        {
            "summary": "Proposed structure for invoices.",
            "operations": [
                {"action": "create_folder", "path": "2023/Taxes", "justification": "Group taxes by year."},
                {
                    "action": "copy_file",
                    "source_id": "SRC001",
                    "target_folder": "2023/Taxes",
                    "target_name": "2023-01-15 Invoice.pdf",
                    "justification": "Rename with date prefix.",
                },
            ],
        }
    )
    service = StructureService(
        settings=settings,
        cache=cache,
        onedrive_client=onedrive,
        ollama_client=ollama,
    )

    plan = service.analyze()
    assert len(plan["operations"]) == 2
    assert "validation" in plan
    assert plan["validation"]["status"] == "ok"
    plan_calls = [call for call in ollama.calls if call["schema_name"] == "StructurePlan"]
    assert plan_calls
    assert plan_calls[0]["model"] == "test-structure-model"
    assert isinstance(plan_calls[0]["json_schema"], dict)
    validation_calls = [call for call in ollama.calls if call["schema_name"] == "StructureValidation"]
    assert validation_calls

    applied = service.apply()
    assert any("Taxes" in folder for folder in onedrive.created_folders)
    assert any(folder == "_done" for folder in onedrive.created_folders)
    assert onedrive.download_requests == ["invoice.pdf"]
    assert onedrive.uploads[0][0] == "2023/Taxes"
    assert onedrive.uploads[0][1] == "2023-01-15 Invoice.pdf"
    assert applied["created_files"] == ["2023/Taxes/2023-01-15 Invoice.pdf"]
    assert onedrive.moves == [("invoice.pdf", "_done", "invoice.pdf")]

    reverted = service.revert()
    assert "2023/Taxes/2023-01-15 Invoice.pdf" in onedrive.deleted
    assert "2023/Taxes" in onedrive.deleted
    assert reverted["removed_files"]


def test_analyze_tolerates_trailing_text(tmp_path):
    settings = build_settings(tmp_path)
    cache = StructureCache(tmp_path / "structure_state.json")
    db = get_database(settings.db_path)
    db.execute(
        """
        INSERT INTO processed_items (onedrive_id, filename, processed_at, status, model, metadata_json, error_message, run_id)
        VALUES (?, ?, datetime('now'), 'success', 'model', ?, NULL, NULL)
        """,
        (
            "item-2",
            "manual.pdf",
            json.dumps({"document_type": "Reference"}),
        ),
    )
    onedrive = StubOnedrive(
        tree=[
            SortedTreeEntry(
                path="manual.pdf",
                item_id="file-2",
                drive_id=None,
                name="manual.pdf",
                is_folder=False,
                download_url="https://download",
                web_url=None,
                base_path="https://graph.microsoft.com/v1.0/me/drive",
            )
        ]
    )
    raw_plan = '{"summary":"ok","operations": []}\nDone.'
    service = StructureService(
        settings=settings,
        cache=cache,
        onedrive_client=onedrive,
        ollama_client=StubOllama(raw_output=raw_plan),
    )

    plan = service.analyze()
    assert plan["summary"] == "ok"
    assert plan["operations"]
    copy_ops = [op for op in plan["operations"] if op["action"] == "copy_file"]
    assert copy_ops
    assert copy_ops[0]["target_folder"] == "Manuals/Reference"
    assert copy_ops[0]["target_name"].lower().startswith("0000-01-01")
    assert plan["validation"]["status"] == "ok"


def test_analyze_generates_fallback_operations(tmp_path):
    settings = build_settings(tmp_path)
    cache = StructureCache(tmp_path / "structure_state.json")
    db = get_database(settings.db_path)
    db.execute(
        """
        INSERT INTO processed_items (onedrive_id, filename, processed_at, status, model, metadata_json, error_message, run_id)
        VALUES (?, ?, datetime('now'), 'success', 'model', ?, NULL, NULL)
        """,
        (
            "item-3",
            "tax.pdf",
            json.dumps({"document_type": "Tax document", "document_date": "2024-02-10", "title": "Tax Certificate"}),
        ),
    )
    onedrive = StubOnedrive(
        tree=[
            SortedTreeEntry(
                path="tax.pdf",
                item_id="file-3",
                drive_id=None,
                name="tax.pdf",
                is_folder=False,
                download_url="https://download",
                web_url=None,
                base_path="https://graph.microsoft.com/v1.0/me/drive",
            )
        ]
    )
    service = StructureService(
        settings=settings,
        cache=cache,
        onedrive_client=onedrive,
        ollama_client=StubOllama({"summary": "ok", "operations": []}),
    )

    plan = service.analyze()
    assert plan["operations"]
    create_ops = [op for op in plan["operations"] if op["action"] == "create_folder"]
    assert any(op["path"] == "Finance/Taxes/2024" for op in create_ops)
    copy_ops = [op for op in plan["operations"] if op["action"] == "copy_file"]
    assert any(
        op["source_id"] == "SRC001" and op["target_folder"] == "Finance/Taxes/2024" and op["target_name"].startswith("2024-02-10")
        for op in copy_ops
    )
    assert plan["validation"]["status"] == "ok"


def test_analyze_repairs_invalid_output(tmp_path):
    settings = build_settings(tmp_path)
    cache = StructureCache(tmp_path / "structure_state.json")
    db = get_database(settings.db_path)
    db.execute(
        """
        INSERT INTO processed_items (onedrive_id, filename, processed_at, status, model, metadata_json, error_message, run_id)
        VALUES (?, ?, datetime('now'), 'success', 'model', ?, NULL, NULL)
        """,
        (
            "item-4",
            "insurance.pdf",
            json.dumps({"document_type": "Insurance", "document_date": "2024-03-05"}),
        ),
    )
    onedrive = StubOnedrive(
        tree=[
            SortedTreeEntry(
                path="insurance.pdf",
                item_id="file-4",
                drive_id=None,
                name="insurance.pdf",
                is_folder=False,
                download_url="https://download",
                web_url=None,
                base_path="https://graph.microsoft.com/v1.0/me/drive",
            )
        ]
    )
    ollama = StubOllama(plan_responses=["not json", {"summary": "fixed", "operations": []}])
    service = StructureService(
        settings=settings,
        cache=cache,
        onedrive_client=onedrive,
        ollama_client=ollama,
    )

    plan = service.analyze()
    assert plan["operations"]
    plan_calls = [call for call in ollama.calls if call["schema_name"] == "StructurePlan"]
    assert len(plan_calls) == 2  # initial + repair attempt
    assert plan["validation"]["status"] == "ok"


def test_analyze_handles_onedrive_failure(tmp_path):
    settings = build_settings(tmp_path)
    cache = StructureCache(tmp_path / "structure_state.json")
    error = RuntimeError("onedrive offline")
    service = StructureService(
        settings=settings,
        cache=cache,
        onedrive_client=FailingOnedrive(error),
        ollama_client=StubOllama({"summary": "", "operations": []}),
    )
    with pytest.raises(StructureServiceError) as exc_info:
        service.analyze()
    assert "onedrive offline" in str(exc_info.value)


def test_parse_plan_with_trailing_junk(tmp_path):
    settings = build_settings(tmp_path)
    cache = StructureCache(tmp_path / "structure_state.json")
    service = StructureService(settings=settings, cache=cache, onedrive_client=StubOnedrive([]), ollama_client=StubOllama({}))

    context = StructureContext(sources=[], existing_folders=set(), folder_examples={})
    raw = """{
        "summary": "Test summary",
        "operations": [
            {
                "action": "create_folder",
                "path": "Folder",
                "justification": "Reason"
            }
        ]
    } extra text"""

    plan = service._plan_engine._parse_plan(raw, context)
    assert plan["summary"] == "Test summary"
    assert plan["operations"][0]["path"] == "Folder"



def test_summary_alternatives(tmp_path):
    settings = build_settings(tmp_path)
    cache = StructureCache(tmp_path / "structure_state.json")
    service = StructureService(settings=settings, cache=cache, onedrive_client=StubOnedrive([]), ollama_client=StubOllama({}))
    context = StructureContext(sources=[], existing_folders=set(), folder_examples={})
    raw = json.dumps({
        'Summary': ['First part', 'Second part'],
        'operations': []
    })
    plan = service._plan_engine._parse_plan(raw, context)
    assert plan['summary'] == 'First part Second part'
