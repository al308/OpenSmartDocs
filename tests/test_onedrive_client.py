import json
import logging
from pathlib import Path

import pytest
import requests

import onedrive_ollama_pipeline.onedrive_client as onedrive_module
from onedrive_ollama_pipeline.config import GraphSettings
from onedrive_ollama_pipeline.onedrive_client import DriveItem, OneDriveClient


class FakeResponse:
    def __init__(self, *, status_code=200, json_payload=None, content=b""):
        self.status_code = status_code
        self._json_payload = json_payload
        self.content = content

    def raise_for_status(self):  # noqa: D401 - mimic requests response
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP error {self.status_code}", response=self)

    def json(self):
        return self._json_payload

    @property
    def ok(self) -> bool:
        return self.status_code < 400


class FakeSession:
    def __init__(self):
        self.get_calls = []
        self.put_calls = []
        self.to_return = []

    def queue_get(self, payload, status_code=200):
        if isinstance(payload, FakeResponse):
            self.to_return.append(payload)
            return
        if isinstance(payload, bytes):
            self.to_return.append(FakeResponse(status_code=status_code, content=payload))
            return
        self.to_return.append(FakeResponse(status_code=status_code, json_payload=payload))

    def get(self, url, headers=None, **kwargs):  # noqa: A003 - align with requests
        self.get_calls.append((url, headers, kwargs))
        return self.to_return.pop(0)

    def put(self, url, headers=None, data=None):
        self.put_calls.append((url, headers, data))
        return FakeResponse(json_payload={"id": "uploaded"})


def make_settings(tmp_path: Path) -> GraphSettings:
    return GraphSettings(
        tenant_id="tenant",
        client_id="client",
        client_secret="secret",
        user_id="user",
        token_cache_path=tmp_path / "token_cache.json",
    )


def test_list_pdfs_in_inbox(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(onedrive_module.OneDriveClient, "_create_app", lambda self: object())
    session = FakeSession()
    session.queue_get(
        {
            "value": [
                {
                    "id": "folder-id",
                    "name": "_inbox",
                    "folder": {},
                }
            ]
        }
    )
    session.queue_get({
        "value": [
            {
                "id": "1",
                "name": "doc.pdf",
                "file": {"mimeType": "application/pdf"},
                "@microsoft.graph.downloadUrl": "https://example/download",
            }
        ]
    })

    client = OneDriveClient(make_settings(tmp_path), session=session)
    monkeypatch.setattr(client, "_get_access_token", lambda: "token")

    items = list(client.list_pdfs_in_inbox())
    assert items[0].name == "doc.pdf"
    assert items[0].item_id == "1"
    assert session.get_calls[0][1]["Authorization"] == "Bearer token"


def test_list_pdfs_in_inbox_remote_items(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(onedrive_module.OneDriveClient, "_create_app", lambda self: object())
    session = FakeSession()
    session.queue_get(
        {
            "value": [
                {
                    "id": "shortcut-id",
                    "name": "_inbox",
                    "folder": {},
                    "remoteItem": {
                        "id": "remote-folder-id",
                        "driveId": "drive-123",
                        "parentReference": {"driveId": "drive-123"},
                    },
                }
            ]
        }
    )
    session.queue_get(
        {
            "value": [
                {
                    "id": "shortcut-file",
                    "name": "Remote.pdf",
                    "remoteItem": {
                        "id": "remote-file-id",
                        "driveId": "drive-123",
                        "file": {"mimeType": "application/pdf"},
                        "@microsoft.graph.downloadUrl": "https://example/download",
                        "webUrl": "https://example/web",
                    },
                }
            ]
        }
    )

    client = OneDriveClient(make_settings(tmp_path), session=session)
    monkeypatch.setattr(client, "_get_access_token", lambda: "token")

    items = list(client.list_pdfs_in_inbox())
    assert items[0].name == "Remote.pdf"
    assert items[0].download_url == "https://example/download"
    assert items[0].drive_id == "drive-123"
    assert items[0].item_id == "remote-file-id"


def test_download_and_upload(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(onedrive_module.OneDriveClient, "_create_app", lambda self: object())
    session = FakeSession()
    session.queue_get(b"pdf-bytes")
    session.queue_get(
        {
            "value": [
                {
                    "id": "sorted-id",
                    "name": "_sorted",
                    "folder": {},
                }
            ]
        }
    )
    client = OneDriveClient(make_settings(tmp_path), session=session)
    monkeypatch.setattr(client, "_get_access_token", lambda: "token")
    item = DriveItem(item_id="1", drive_id=None, name="doc.pdf", download_url="https://file", web_url=None)

    data = client.download_item(item)
    assert data == b"pdf-bytes"
    assert session.get_calls[0][0] == "https://file"

    response = client.upload_pdf_to_sorted("doc.pdf", b"content")
    assert response["id"] == "uploaded"
    assert session.put_calls


def test_download_item_without_direct_url(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(onedrive_module.OneDriveClient, "_create_app", lambda self: object())
    session = FakeSession()
    session.queue_get(FakeResponse(status_code=404))
    session.queue_get(b"pdf-bytes")

    client = OneDriveClient(make_settings(tmp_path), session=session)
    monkeypatch.setattr(client, "_get_access_token", lambda: "token")

    item = DriveItem(
        item_id="remote-item",
        drive_id="drive-123",
        name="doc.pdf",
        download_url=None,
        web_url=None,
    )

    data = client.download_item(item)
    assert data == b"pdf-bytes"
    first_call, first_headers, _ = session.get_calls[0]
    second_call, second_headers, _ = session.get_calls[1]
    assert first_call == "https://graph.microsoft.com/v1.0/users/user/drive/items/remote-item/content"
    assert first_headers["Authorization"] == "Bearer token"
    assert second_call == "https://graph.microsoft.com/v1.0/drives/drive-123/items/remote-item/content"
    assert second_headers["Authorization"] == "Bearer token"


def test_list_pdfs_in_inbox_missing_folder(monkeypatch, caplog, tmp_path: Path):
    monkeypatch.setattr(onedrive_module.OneDriveClient, "_create_app", lambda self: object())
    session = FakeSession()
    session.queue_get({"value": [{"name": "Documents"}]})
    session.queue_get({"value": [{"name": "Documents"}]})

    client = OneDriveClient(make_settings(tmp_path), session=session)
    monkeypatch.setattr(client, "_get_access_token", lambda: "token")

    caplog.set_level(logging.ERROR)
    with pytest.raises(FileNotFoundError):
        list(client.list_pdfs_in_inbox())

    assert "OneDrive folder '_inbox' not found" in caplog.text
    assert "OneDrive root currently contains" in caplog.text
