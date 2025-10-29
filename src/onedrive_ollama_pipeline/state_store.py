"""SQLite-backed state tracking for processed OneDrive items."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .database import Database, get_database


class StateStore:
    """Track processed file identifiers and metadata in SQLite."""

    def __init__(self, db_path: Path):
        self._db: Database = get_database(db_path)

    def is_processed(self, item_id: str) -> bool:
        row = self._db.fetch_one(
            "SELECT status FROM processed_items WHERE onedrive_id = ? AND status = 'success'",
            (item_id,),
        )
        return row is not None

    def record_success(
        self,
        *,
        item_id: str,
        filename: str,
        model: Optional[str],
        metadata: Optional[dict],
        run_id: Optional[str] = None,
    ) -> None:
        processed_at = self._utc_now()
        metadata_json = Database.encode_json(metadata) if metadata is not None else None
        self._db.execute(
            """
            INSERT INTO processed_items (onedrive_id, filename, processed_at, status, model, metadata_json, error_message, run_id)
            VALUES (?, ?, ?, 'success', ?, ?, NULL, ?)
            ON CONFLICT(onedrive_id)
            DO UPDATE SET
                filename = excluded.filename,
                processed_at = excluded.processed_at,
                status = excluded.status,
                model = excluded.model,
                metadata_json = excluded.metadata_json,
                error_message = NULL,
                run_id = excluded.run_id
            """,
            (item_id, filename, processed_at, model, metadata_json, run_id),
        )

    def record_failure(
        self,
        *,
        item_id: str,
        filename: str,
        model: Optional[str],
        error: str,
        run_id: Optional[str] = None,
    ) -> None:
        processed_at = self._utc_now()
        self._db.execute(
            """
            INSERT INTO processed_items (onedrive_id, filename, processed_at, status, model, metadata_json, error_message, run_id)
            VALUES (?, ?, ?, 'failed', ?, NULL, ?, ?)
            ON CONFLICT(onedrive_id)
            DO UPDATE SET
                filename = excluded.filename,
                processed_at = excluded.processed_at,
                status = excluded.status,
                model = excluded.model,
                metadata_json = NULL,
                error_message = excluded.error_message,
                run_id = excluded.run_id
            """,
            (item_id, filename, processed_at, model, error, run_id),
        )

    def recent_items(self, limit: int = 50) -> list[dict]:
        rows = self._db.query(
            """
            SELECT onedrive_id, filename, processed_at, status, model, metadata_json, error_message, run_id
            FROM processed_items
            ORDER BY datetime(processed_at) DESC
            LIMIT ?
            """,
            (limit,),
        )
        results: list[dict] = []
        for row in rows:
            results.append(
                {
                    "onedrive_id": row["onedrive_id"],
                    "filename": row["filename"],
                    "processed_at": row["processed_at"],
                    "status": row["status"],
                    "model": row["model"],
                    "metadata": Database.decode_json(row["metadata_json"]),
                    "error_message": row["error_message"],
                    "run_id": row["run_id"],
                }
            )
        return results

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()
