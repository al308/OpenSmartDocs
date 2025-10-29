"""Lightweight Microsoft Graph OneDrive client."""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any, Iterable, Optional, Tuple

import msal
import requests

from .config import GraphSettings

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DriveItem:
    """Subset of OneDrive item metadata used by the pipeline."""

    item_id: str
    drive_id: Optional[str]
    name: str
    download_url: Optional[str]
    web_url: Optional[str]


@dataclass(frozen=True)
class SortedTreeEntry:
    """Representation of an item stored under the configured sorted folder."""

    path: str
    item_id: str
    drive_id: Optional[str]
    name: str
    is_folder: bool
    download_url: Optional[str]
    web_url: Optional[str]
    base_path: str


class OneDriveClient:
    """Wrapper around Microsoft Graph requests for OneDrive interactions."""

    GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"

    def __init__(self, settings: GraphSettings, session: Optional[requests.Session] = None):
        self._settings = settings
        self._session = session or requests.Session()
        self._cache_path = settings.token_cache_path.expanduser()
        self._token_cache = msal.SerializableTokenCache()
        if self._cache_path.exists():
            try:
                self._token_cache.deserialize(self._cache_path.read_text(encoding="utf-8"))
            except Exception as exc:  # pragma: no cover - defensive
                _LOGGER.warning("Failed to load token cache %s: %s", self._cache_path, exc)
        self._app = self._create_app()

    def _drive_base_path(self) -> str:
        user_id = self._settings.user_id.strip()
        if user_id.lower() == "me":
            return f"{self.GRAPH_BASE_URL}/me/drive"
        return f"{self.GRAPH_BASE_URL}/users/{user_id}/drive"

    @staticmethod
    def _normalize_folder_path(folder: str) -> str:
        path = folder.strip()
        if path.startswith("/"):
            path = path[1:]
        if path.endswith("/"):
            path = path[:-1]
        return path or ""

    @staticmethod
    def _normalize_relative_path(path: str) -> str:
        normalized = (path or "").strip().strip("/")
        return normalized

    def _create_app(self):
        authority = self._settings.authority or f"https://login.microsoftonline.com/{self._settings.tenant_id}"
        if self._settings.client_secret:
            return msal.ConfidentialClientApplication(
                client_id=self._settings.client_id,
                client_credential=self._settings.client_secret,
                authority=authority,
                token_cache=self._token_cache,
            )
        return msal.PublicClientApplication(
            client_id=self._settings.client_id,
            authority=authority,
            token_cache=self._token_cache,
        )

    def _get_access_token(self) -> str:
        scopes = list(self._settings.scopes)
        account = None
        if isinstance(self._app, msal.PublicClientApplication):
            accounts = self._app.get_accounts()
            if accounts:
                account = accounts[0]
                token = self._app.acquire_token_silent(scopes, account=account)
                if token:
                    self._persist_token_cache()
                    return token["access_token"]
            flow = self._app.initiate_device_flow(scopes=scopes)
            if "user_code" not in flow:
                raise RuntimeError("Device flow initialization failed")
            _LOGGER.info("Authorize the application by visiting %s and entering code %s", flow["verification_uri"], flow["user_code"])
            token = self._app.acquire_token_by_device_flow(flow)
        else:
            token = self._app.acquire_token_silent(scopes, account=account)
            if not token:
                token = self._app.acquire_token_for_client(scopes=scopes)
        if "access_token" not in token:
            raise RuntimeError(f"Failed to acquire token: {token.get('error_description', token)}")
        self._persist_token_cache()
        return token["access_token"]

    def _persist_token_cache(self) -> None:
        if not self._token_cache.has_state_changed:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(self._token_cache.serialize(), encoding="utf-8")
        except Exception as exc:  # pragma: no cover - diagnostic only
            _LOGGER.warning("Failed to persist token cache %s: %s", self._cache_path, exc)

    def _auth_headers(self) -> dict[str, str]:
        token = self._get_access_token()
        return {"Authorization": f"Bearer {token}"}

    def list_pdfs_in_inbox(self) -> Iterable[DriveItem]:
        """Yield PDF files stored inside the configured inbox folder."""
        folder = self._normalize_folder_path(self._settings.inbox_folder)
        headers = self._auth_headers()
        base_path, folder_id = self._resolve_folder_reference(folder, headers)
        select_fields = "id,name,file,@microsoft.graph.downloadUrl,folder,parentReference,remoteItem,webUrl"
        if folder_id == "root":
            url = f"{base_path}/root/children?$select={select_fields}"
        else:
            url = f"{base_path}/items/{folder_id}/children?$select={select_fields}"
        while url:
            response = self._session.get(url, headers=headers)
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                if response.status_code == 404:
                    self._log_folder_not_found(folder, headers, base_path, folder_id)
                raise exc
            data = response.json()
            for item in data.get("value", []):
                remote = item.get("remoteItem") or {}
                file_info = item.get("file") or remote.get("file")
                name = item.get("name") or remote.get("name", "")
                _LOGGER.debug(
                    "Found child item '%s' (file=%s remote=%s)",
                    name,
                    bool(file_info),
                    bool(remote),
                )
                if not file_info:
                    continue
                if not name.lower().endswith(".pdf"):
                    continue
                download_url = item.get("@microsoft.graph.downloadUrl") or remote.get("@microsoft.graph.downloadUrl")
                parent_ref = item.get("parentReference") or remote.get("parentReference") or {}
                drive_id = remote.get("driveId") or parent_ref.get("driveId")
                item_id = item.get("id") or remote.get("id") or parent_ref.get("id")
                remote_item_id = remote.get("id")
                if remote_item_id:
                    item_id = remote_item_id
                web_url = item.get("webUrl") or remote.get("webUrl")
                if not item_id:
                    _LOGGER.debug(
                        "Skipping '%s' because item id is missing (keys=%s, remote_keys=%s)",
                        name,
                        list(item.keys()),
                        list(remote.keys()) if remote else [],
                    )
                    continue
                if not download_url:
                    _LOGGER.debug(
                        "Item '%s' has no download URL; will fallback to Graph content API",
                        name,
                    )
                    _LOGGER.debug(
                        "Item '%s' metadata for fallback: parentReference=%s remote=%s",
                        name,
                        parent_ref,
                        remote,
                    )
                yield DriveItem(
                    item_id=item_id,
                    drive_id=drive_id,
                    name=name,
                    download_url=download_url,
                    web_url=web_url,
                )
            url = data.get("@odata.nextLink")

    def download_item(self, item: DriveItem) -> bytes:
        """Download a file's bytes from OneDrive."""
        if item.download_url:
            response = self._session.get(item.download_url, timeout=60)
            response.raise_for_status()
            return response.content

        headers = self._auth_headers()
        candidate_urls = []
        default_base = self._drive_base_path()
        candidate_urls.append(f"{default_base}/items/{item.item_id}/content")
        if item.drive_id:
            candidate_urls.append(f"{self.GRAPH_BASE_URL}/drives/{item.drive_id}/items/{item.item_id}/content")

        last_error: Optional[requests.HTTPError] = None
        for url in candidate_urls:
            _LOGGER.debug("Downloading '%s' via Graph content endpoint %s", item.name, url)
            response = self._session.get(url, headers=headers, timeout=60)
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                last_error = exc
                if exc.response is not None and exc.response.status_code == 404:
                    continue
                raise
            return response.content

        if last_error:
            raise last_error
        raise RuntimeError(f"No download URL resolved for item {item.item_id}")

    def upload_pdf_to_inbox(self, filename: str, content: bytes) -> dict:
        """Upload a PDF into the configured inbox folder, replacing any existing file."""
        folder = self._normalize_folder_path(self._settings.inbox_folder)
        return self._upload_pdf_to_folder(filename, content, folder)

    def upload_pdf_to_sorted(self, filename: str, content: bytes) -> dict:
        """Upload a PDF into the configured sorted folder, replacing any existing file."""
        folder = self._normalize_folder_path(self._settings.sorted_folder)
        return self._upload_pdf_to_folder(filename, content, folder)

    def _upload_pdf_to_folder(self, filename: str, content: bytes, folder: str) -> dict:
        headers = self._auth_headers()
        base_path, folder_id = self._resolve_folder_reference(folder, headers)
        if folder_id == "root":
            url = f"{base_path}/root:/{filename}:/content"
        else:
            url = f"{base_path}/items/{folder_id}:/{filename}:/content"
        response = self._session.put(url, headers=headers, data=content)
        response.raise_for_status()
        return response.json()

    def move_to_sorted(self, item: DriveItem) -> dict:
        """Move an existing item to the sorted folder using the Graph API."""
        folder = self._normalize_folder_path(self._settings.sorted_folder)
        headers = self._auth_headers()
        base_path, folder_id = self._resolve_folder_reference(folder, headers)
        url = f"{base_path}/items/{item.item_id}"
        if folder_id == "root":
            payload = {"parentReference": {"path": "/drive/root"}}
        else:
            payload = {"parentReference": {"id": folder_id}}
        response = self._session.patch(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()

    def _ensure_folder_reference(self, folder: str, headers: dict[str, str]) -> Tuple[str, str]:
        """
        Resolve a folder path, creating intermediate components when missing.

        Returns the base path (to handle remote drives) and the final folder id.
        """
        base_path = self._drive_base_path()
        if not folder:
            return base_path, "root"
        segments = [segment for segment in folder.split("/") if segment]
        current_base = base_path
        current_id = "root"
        for segment in segments:
            if current_id == "root":
                url = f"{current_base}/root/children?$select=id,name,folder,parentReference,remoteItem"
            else:
                url = f"{current_base}/items/{current_id}/children?$select=id,name,folder,parentReference,remoteItem"
            response = self._session.get(url, headers=headers)
            response.raise_for_status()
            payload = response.json()
            match = self._find_child_with_name(payload, segment)
            if not match:
                if current_id == "root":
                    create_url = f"{current_base}/root/children"
                else:
                    create_url = f"{current_base}/items/{current_id}/children"
                create_payload = {"name": segment, "folder": {}, "@microsoft.graph.conflictBehavior": "fail"}
                create_resp = self._session.post(create_url, headers=headers, json=create_payload)
                if create_resp.status_code == 409:
                    # Folder already exists after concurrent creation; reload listing.
                    response = self._session.get(url, headers=headers)
                    response.raise_for_status()
                    payload = response.json()
                    match = self._find_child_with_name(payload, segment)
                    if not match:
                        raise RuntimeError(f"Failed to resolve or create folder segment '{segment}'")
                else:
                    create_resp.raise_for_status()
                    match = create_resp.json()
            remote = match.get("remoteItem")
            if remote:
                parent_ref = remote.get("parentReference", {})
                drive_id = remote.get("driveId") or parent_ref.get("driveId")
                item_id = remote.get("id") or parent_ref.get("id")
                if not drive_id or not item_id:
                    raise RuntimeError("Remote folder reference missing driveId/id")
                current_base = f"{self.GRAPH_BASE_URL}/drives/{drive_id}"
                current_id = item_id
            else:
                if "folder" not in match:
                    raise RuntimeError(f"Path component '{segment}' is not a folder")
                current_id = match["id"]
        return current_base, current_id

    def _resolve_folder_reference(self, folder: str, headers: dict[str, str]) -> Tuple[str, str]:
        base_path = self._drive_base_path()
        if not folder:
            return base_path, "root"
        segments = [segment for segment in folder.split("/") if segment]
        current_base = base_path
        current_id = "root"
        for segment in segments:
            if current_id == "root":
                url = f"{current_base}/root/children?$select=id,name,folder,parentReference,remoteItem"
            else:
                url = f"{current_base}/items/{current_id}/children?$select=id,name,folder,parentReference,remoteItem"
            response = self._session.get(url, headers=headers)
            response.raise_for_status()
            payload = response.json()
            match = self._find_child_with_name(payload, segment)
            if not match:
                self._log_folder_not_found(folder, headers, current_base, current_id)
                raise FileNotFoundError(f"Folder '{folder}' not found in OneDrive")
            remote = match.get("remoteItem")
            if remote:
                parent_ref = remote.get("parentReference", {})
                drive_id = parent_ref.get("driveId") or remote.get("driveId")
                item_id = remote.get("id") or parent_ref.get("id")
                if not drive_id or not item_id:
                    raise RuntimeError("Remote folder reference missing driveId/id")
                current_base = f"{self.GRAPH_BASE_URL}/drives/{drive_id}"
                current_id = item_id
            else:
                if "folder" not in match:
                    raise RuntimeError(f"Path component '{segment}' is not a folder")
                current_id = match["id"]
        return current_base, current_id

    def walk_sorted_tree(self) -> Iterable[SortedTreeEntry]:
        """
        Yield every item (recursively) stored under the configured sorted folder.

        Each entry describes the relative path from the sorted folder root.
        """
        folder = self._normalize_folder_path(self._settings.sorted_folder)
        headers = self._auth_headers()
        base_path, folder_id = self._ensure_folder_reference(folder, headers)
        stack: list[tuple[str, str, str]] = [("", folder_id, base_path)]
        select_fields = "id,name,folder,parentReference,remoteItem,file,webUrl,@microsoft.graph.downloadUrl"
        while stack:
            current_path, current_id, current_base = stack.pop()
            if current_id == "root":
                url = f"{current_base}/root/children?$select={select_fields}"
            else:
                url = f"{current_base}/items/{current_id}/children?$select={select_fields}"
            next_link = url
            while next_link:
                response = self._session.get(next_link, headers=headers)
                response.raise_for_status()
                payload = response.json()
                for item in payload.get("value", []):
                    remote = item.get("remoteItem") or {}
                    name = item.get("name") or remote.get("name", "")
                    if not name:
                        continue
                    folder_info = item.get("folder") or remote.get("folder")
                    download_url = item.get("@microsoft.graph.downloadUrl") or remote.get("@microsoft.graph.downloadUrl")
                    web_url = item.get("webUrl") or remote.get("webUrl")
                    item_id = item.get("id") or remote.get("id") or ""
                    parent_ref = item.get("parentReference") or remote.get("parentReference") or {}
                    drive_id = remote.get("driveId") or parent_ref.get("driveId")
                    relative_path = "/".join(part for part in [current_path, name] if part)
                    entry = SortedTreeEntry(
                        path=relative_path,
                        item_id=item_id,
                        drive_id=drive_id,
                        name=name,
                        is_folder=bool(folder_info),
                        download_url=download_url,
                        web_url=web_url,
                        base_path=current_base,
                    )
                    yield entry
                    if folder_info:
                        if remote:
                            remote_parent = remote.get("parentReference", {})
                            child_drive_id = remote.get("driveId") or remote_parent.get("driveId")
                            child_id = remote.get("id") or remote_parent.get("id")
                            if not child_drive_id or not child_id:
                                raise RuntimeError("Remote folder reference missing driveId/id")
                            next_base = f"{self.GRAPH_BASE_URL}/drives/{child_drive_id}"
                            stack.append((relative_path, child_id, next_base))
                        else:
                            child_id = item.get("id")
                            if child_id:
                                stack.append((relative_path, child_id, current_base))
                next_link = payload.get("@odata.nextLink")

    def ensure_sorted_subfolder(self, relative_path: str) -> SortedTreeEntry:
        """
        Ensure a subfolder exists under the sorted root and return its metadata.
        """
        normalized = self._normalize_relative_path(relative_path)
        folder = self._normalize_folder_path(self._settings.sorted_folder)
        target_folder = "/".join(part for part in [folder, normalized] if part)
        headers = self._auth_headers()
        base_path, folder_id = self._ensure_folder_reference(target_folder, headers)
        if folder_id == "root":
            meta_url = f"{base_path}/root"
        else:
            meta_url = f"{base_path}/items/{folder_id}"
        response = self._session.get(meta_url, headers=headers)
        response.raise_for_status()
        data = response.json()
        name = data.get("name") or normalized.split("/")[-1]
        parent_ref = data.get("parentReference", {})
        drive_id = parent_ref.get("driveId")
        web_url = data.get("webUrl")
        item_id = data.get("id") or folder_id
        return SortedTreeEntry(
            path=normalized,
            item_id=item_id,
            drive_id=drive_id,
            name=name,
            is_folder=True,
            download_url=None,
            web_url=web_url,
            base_path=base_path,
        )

    def resolve_sorted_item(self, relative_path: str, headers: Optional[dict[str, str]] = None) -> SortedTreeEntry:
        """
        Resolve metadata for a file or folder stored under the sorted root.
        """
        normalized = self._normalize_relative_path(relative_path)
        folder = self._normalize_folder_path(self._settings.sorted_folder)
        headers = headers or self._auth_headers()
        base_path, folder_id = self._ensure_folder_reference(folder, headers)
        if folder_id == "root":
            item_url = f"{base_path}/root:/{normalized}"
        else:
            item_url = f"{base_path}/items/{folder_id}:/{normalized}"
        select_fields = "id,name,folder,parentReference,remoteItem,file,webUrl,@microsoft.graph.downloadUrl"
        response = self._session.get(f"{item_url}?$select={select_fields}", headers=headers)
        response.raise_for_status()
        data = response.json()
        remote = data.get("remoteItem") or {}
        folder_info = data.get("folder") or remote.get("folder")
        download_url = data.get("@microsoft.graph.downloadUrl") or remote.get("@microsoft.graph.downloadUrl")
        web_url = data.get("webUrl") or remote.get("webUrl")
        parent_ref = data.get("parentReference") or remote.get("parentReference") or {}
        drive_id = remote.get("driveId") or parent_ref.get("driveId")
        name = data.get("name") or remote.get("name") or normalized.split("/")[-1]
        item_id = data.get("id") or remote.get("id") or folder_id
        if drive_id:
            base_path = f"{self.GRAPH_BASE_URL}/drives/{drive_id}"
        return SortedTreeEntry(
            path=normalized,
            item_id=item_id,
            drive_id=drive_id,
            name=name,
            is_folder=bool(folder_info),
            download_url=download_url,
            web_url=web_url,
            base_path=base_path,
        )

    def download_sorted_file(self, relative_path: str) -> tuple[SortedTreeEntry, bytes]:
        """
        Download a file stored under the sorted root.
        """
        entry = self.resolve_sorted_item(relative_path)
        if entry.is_folder:
            raise RuntimeError(f"Path '{relative_path}' refers to a folder, not a file")
        drive_item = DriveItem(
            item_id=entry.item_id,
            drive_id=entry.drive_id,
            name=entry.name,
            download_url=entry.download_url,
            web_url=entry.web_url,
        )
        content = self.download_item(drive_item)
        return entry, content

    def upload_bytes_to_sorted(self, filename: str, content: bytes, *, folder_path: str = "") -> dict:
        """
        Upload arbitrary bytes into the sorted folder (creating intermediate folders).
        """
        if not filename:
            raise ValueError("filename must not be empty")
        folder = self._normalize_folder_path(self._settings.sorted_folder)
        relative_folder = self._normalize_relative_path(folder_path)
        target_folder = "/".join(part for part in [folder, relative_folder] if part)
        headers = self._auth_headers()
        base_path, folder_id = self._ensure_folder_reference(target_folder, headers)
        if folder_id == "root":
            url = f"{base_path}/root:/{filename}:/content"
        else:
            url = f"{base_path}/items/{folder_id}:/{filename}:/content"
        response = self._session.put(url, headers=headers, data=content)
        response.raise_for_status()
        return response.json()

    def delete_sorted_path(self, relative_path: str) -> None:
        """
        Delete a file or folder that resides under the sorted folder.
        """
        normalized = self._normalize_relative_path(relative_path)
        if not normalized:
            raise ValueError("Cannot delete the sorted root folder")
        entry = self.resolve_sorted_item(normalized)
        headers = self._auth_headers()
        if entry.drive_id:
            base = f"{self.GRAPH_BASE_URL}/drives/{entry.drive_id}"
        else:
            base = entry.base_path
        url = f"{base}/items/{entry.item_id}"
        response = self._session.delete(url, headers=headers)
        if response.status_code == 404:
            return
        if response.status_code not in (200, 202, 204):
            response.raise_for_status()

    def move_sorted_item(self, source_relative_path: str, target_folder: str, target_name: Optional[str] = None) -> dict:
        """Move an existing item under the sorted root into a different subfolder."""
        source_entry = self.resolve_sorted_item(source_relative_path)
        headers = self._auth_headers()

        normalized_folder = self._normalize_relative_path(target_folder)
        sorted_root = self._normalize_folder_path(self._settings.sorted_folder)
        target_path = "/".join(part for part in [sorted_root, normalized_folder] if part)

        dest_base, dest_folder_id = self._ensure_folder_reference(target_path, headers)
        payload: dict[str, Any] = {}

        if dest_folder_id == "root":
            payload["parentReference"] = {"path": "/drive/root"}
        else:
            dest_drive_id = self._extract_drive_id(dest_base)
            if dest_drive_id:
                payload["parentReference"] = {"driveId": dest_drive_id, "id": dest_folder_id}
            else:
                payload["parentReference"] = {"id": dest_folder_id}

        if target_name:
            payload["name"] = target_name

        item_base = source_entry.base_path or self._drive_base_path()
        url = f"{item_base}/items/{source_entry.item_id}"
        response = self._session.patch(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _extract_drive_id(base_path: str) -> Optional[str]:
        marker = "/drives/"
        if marker in base_path:
            return base_path.split(marker, 1)[1].split("/", 1)[0]
        return None
    @staticmethod
    def _find_child_with_name(payload: dict, name: str) -> Optional[dict]:
        target = name.lower()
        for item in payload.get("value", []):
            if item.get("name", "").lower() == target:
                return item
        return None

    def _log_folder_not_found(self, folder: str, headers: dict[str, str], base_path: Optional[str] = None, parent_id: str = "root") -> None:
        folder_label = folder or "/"
        _LOGGER.error("OneDrive folder '%s' not found (HTTP 404)", folder_label)
        try:
            base = base_path or self._drive_base_path()
            select_fields = "name"
            if parent_id == "root":
                root_url = f"{base}/root/children?$select={select_fields}"
            else:
                root_url = f"{base}/items/{parent_id}/children?$select={select_fields}"
            root_resp = self._session.get(root_url, headers=headers)
            if not root_resp.ok:
                _LOGGER.debug(
                    "Failed to list OneDrive root (status %s)",
                    root_resp.status_code,
                )
                return
            names = [item.get("name") for item in root_resp.json().get("value", [])]
            if names:
                _LOGGER.error("OneDrive root currently contains: %s", ", ".join(names))
            else:
                _LOGGER.error("OneDrive root listing succeeded but returned no items")
        except Exception as exc:  # pragma: no cover - diagnostic only
            _LOGGER.debug("Could not fetch OneDrive root contents: %s", exc, exc_info=True)
