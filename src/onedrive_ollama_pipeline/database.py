"""SQLite helpers for the OneDrive → Ollama pipeline."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Iterable, Optional


class Database:
    """Lightweight thread-safe wrapper around sqlite3."""

    def __init__(self, path: Path):
        self._path = Path(path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._path.as_posix(), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL;")
        self._ensure_schema()

    @property
    def path(self) -> Path:
        return self._path

    def _ensure_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS processed_items (
                    onedrive_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    processed_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    model TEXT,
                    metadata_json TEXT,
                    error_message TEXT,
                    run_id TEXT
                );

                CREATE TABLE IF NOT EXISTS config_overrides (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            self._conn.commit()

    def execute(self, sql: str, params: Iterable | None = None) -> None:
        with self._lock:
            self._conn.execute(sql, tuple(params or ()))
            self._conn.commit()

    def query(self, sql: str, params: Iterable | None = None) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params or ()))
            rows = cur.fetchall()
        return rows

    def fetch_one(self, sql: str, params: Iterable | None = None) -> Optional[sqlite3.Row]:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    @staticmethod
    def encode_json(data: object) -> str:
        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def decode_json(value: Optional[str]) -> Optional[object]:
        if value is None:
            return None
        return json.loads(value)


_db_cache: dict[Path, Database] = {}
_db_lock = threading.Lock()


def get_database(path: Path) -> Database:
    path = Path(path).expanduser()
    with _db_lock:
        if path not in _db_cache:
            _db_cache[path] = Database(path)
        return _db_cache[path]
