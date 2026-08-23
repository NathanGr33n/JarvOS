"""Append-only SQLite audit log for executed and cancelled OS actions."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class AuditEntry:
    """One immutable action audit record."""

    id: int
    action_name: str
    argument: str
    status: str
    detail: str
    confirmed: bool
    user_transcript: str
    created_at: str


class ActionAuditLog:
    """Local SQLite store for action outcomes (success, error, cancelled)."""

    def __init__(self, db_path: Path | str, enabled: bool = True):
        self.enabled = enabled
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        if self.enabled:
            self._connect()

    def _connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        # WAL is fine for local single-writer audit traffic.
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        assert self._conn is not None
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS action_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_name TEXT NOT NULL,
                argument TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                detail TEXT NOT NULL DEFAULT '',
                confirmed INTEGER NOT NULL DEFAULT 0,
                user_transcript TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_action_audit_created
                ON action_audit(created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_action_audit_name
                ON action_audit(action_name);
            """
        )
        self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def record(
        self,
        action_name: str,
        *,
        argument: str = "",
        status: str,
        detail: str = "",
        confirmed: bool = False,
        user_transcript: str = "",
    ) -> None:
        """Append one audit row. No update/delete API is exposed."""
        if not self.enabled or self._conn is None:
            return
        normalized_status = (status or "unknown").strip().lower() or "unknown"
        self._conn.execute(
            """
            INSERT INTO action_audit (
                action_name, argument, status, detail, confirmed, user_transcript
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                action_name,
                argument or "",
                normalized_status,
                detail or "",
                1 if confirmed else 0,
                user_transcript or "",
            ),
        )
        self._conn.commit()

    def list_entries(
        self,
        limit: int = 50,
        *,
        action_name: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[AuditEntry]:
        """Return recent entries newest-first."""
        if not self.enabled or self._conn is None or limit <= 0:
            return []

        clauses: list[str] = []
        params: list[object] = []
        if action_name:
            clauses.append("action_name = ?")
            params.append(action_name)
        if status:
            clauses.append("status = ?")
            params.append(status.strip().lower())

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = self._conn.execute(
            f"""
            SELECT id, action_name, argument, status, detail, confirmed,
                   user_transcript, created_at
            FROM action_audit
            {where}
            ORDER BY id DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return [
            AuditEntry(
                id=row["id"],
                action_name=row["action_name"],
                argument=row["argument"],
                status=row["status"],
                detail=row["detail"],
                confirmed=bool(row["confirmed"]),
                user_transcript=row["user_transcript"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def count(self) -> int:
        """Return total number of audit rows."""
        if not self.enabled or self._conn is None:
            return 0
        row = self._conn.execute("SELECT COUNT(*) AS n FROM action_audit").fetchone()
        return int(row["n"]) if row is not None else 0
