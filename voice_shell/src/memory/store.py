"""Lightweight SQLite-backed persistent memory for conversation and facts."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class ConversationTurn:
    """One user/assistant exchange."""

    user_text: str
    assistant_text: str
    created_at: str = ""


class MemoryStore:
    """Local SQLite store for recent turns and simple key/value facts."""

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
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        assert self._conn is not None
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversation_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_text TEXT NOT NULL,
                assistant_text TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS facts (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            """
        )
        self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        """Persist one conversation turn."""
        if not self.enabled or self._conn is None:
            return
        self._conn.execute(
            "INSERT INTO conversation_turns (user_text, assistant_text) VALUES (?, ?)",
            (user_text, assistant_text),
        )
        self._conn.commit()

    def get_recent_turns(self, limit: int = 3) -> List[ConversationTurn]:
        """Return the most recent turns oldest-first."""
        if not self.enabled or self._conn is None or limit <= 0:
            return []
        rows = self._conn.execute(
            """
            SELECT user_text, assistant_text, created_at
            FROM conversation_turns
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        turns = [
            ConversationTurn(
                user_text=row["user_text"],
                assistant_text=row["assistant_text"],
                created_at=row["created_at"],
            )
            for row in rows
        ]
        turns.reverse()
        return turns

    def set_fact(self, key: str, value: str) -> None:
        """Upsert a simple preference/fact."""
        if not self.enabled or self._conn is None:
            return
        self._conn.execute(
            """
            INSERT INTO facts (key, value, updated_at)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = datetime('now')
            """,
            (key, value),
        )
        self._conn.commit()

    def get_fact(self, key: str) -> Optional[str]:
        """Return a stored fact value, or None if missing."""
        if not self.enabled or self._conn is None:
            return None
        row = self._conn.execute(
            "SELECT value FROM facts WHERE key = ?",
            (key,),
        ).fetchone()
        return None if row is None else row["value"]

    def list_facts(self, limit: int = 20) -> List[Tuple[str, str]]:
        """Return up to ``limit`` facts as (key, value) pairs."""
        if not self.enabled or self._conn is None or limit <= 0:
            return []
        rows = self._conn.execute(
            """
            SELECT key, value
            FROM facts
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [(row["key"], row["value"]) for row in rows]

    def clear_turns(self) -> None:
        """Delete all conversation turns (facts are retained)."""
        if not self.enabled or self._conn is None:
            return
        self._conn.execute("DELETE FROM conversation_turns")
        self._conn.commit()
