import re
import sqlite3
from pathlib import Path


DATABASE_PATH = Path(__file__).resolve().parents[1] / "savedMem" / "agent_memory.db"
_MEMORY_CATEGORIES = {"project", "preference", "decision", "lesson", "task", "other"}
_MAX_MEMORY_CONTENT_LENGTH = 900

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'tool')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_messages_session_time
    ON messages(session_id, created_at);

CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY,
    memory_type TEXT NOT NULL CHECK (memory_type IN ('short_term', 'long_term')),
    category TEXT NOT NULL CHECK (
        category IN ('project', 'preference', 'decision', 'lesson', 'task', 'other')
    ),
    content TEXT NOT NULL,
    importance INTEGER NOT NULL DEFAULT 3 CHECK (importance BETWEEN 1 AND 5),
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'superseded')),
    source_session_id INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_memories_active
    ON memories(status, memory_type, category);

CREATE INDEX IF NOT EXISTS idx_memories_expiration
    ON memories(expires_at);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    content='memories',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS memories_ai
AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content)
    VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad
AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS memories_au
AFTER UPDATE OF content ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
    INSERT INTO memories_fts(rowid, content)
    VALUES (new.id, new.content);
END;
"""


def get_connection(
    db_path: str | Path = DATABASE_PATH,
) -> sqlite3.Connection:
    """Open the SQLite database, ensure its schema exists, and return the connection."""
    database_path = Path(db_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(_SCHEMA)
    return connection


def save_long_term_memory(
    content: str,
    category: str = "other",
    importance: int = 3,
    db_path: str | Path = DATABASE_PATH,
) -> str:
    """Persist one long-term memory entry and return its database ID."""
    memory_ids = save_long_term_memories(
        [{"content": content, "category": category, "importance": importance}],
        db_path=db_path,
    )
    return f"Saved long-term memory (id={memory_ids[0]})."


def save_long_term_memories(
    memories: list[dict[str, object]],
    db_path: str | Path = DATABASE_PATH,
) -> list[int]:
    """Persist validated long-term memories atomically, avoiding exact duplicates."""
    validated: list[tuple[str, str, int]] = []
    for memory in memories:
        content = memory.get("content")
        category = memory.get("category", "other")
        importance = memory.get("importance", 3)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Memory content must be a non-empty string.")
        if len(content.strip()) > _MAX_MEMORY_CONTENT_LENGTH:
            raise ValueError(
                f"Memory content cannot exceed {_MAX_MEMORY_CONTENT_LENGTH} characters."
            )
        if not isinstance(category, str) or category not in _MEMORY_CATEGORIES:
            raise ValueError(f"Invalid memory category: {category!r}.")
        if (
            not isinstance(importance, int)
            or isinstance(importance, bool)
            or not 1 <= importance <= 5
        ):
            raise ValueError("Memory importance must be an integer from 1 to 5.")
        validated.append((content.strip(), category, importance))

    if not validated:
        return []

    connection = get_connection(db_path)
    try:
        memory_ids = []
        for content, category, importance in validated:
            existing = connection.execute(
                """
                SELECT id
                FROM memories
                WHERE memory_type = 'long_term'
                  AND category = ?
                  AND lower(content) = lower(?)
                  AND status = 'active'
                LIMIT 1
                """,
                (category, content),
            ).fetchone()
            if existing:
                memory_id = existing["id"]
                connection.execute(
                    """
                    UPDATE memories
                    SET importance = MAX(importance, ?), updated_at = datetime('now')
                    WHERE id = ?
                    """,
                    (importance, memory_id),
                )
            else:
                cursor = connection.execute(
                    """
                    INSERT INTO memories (memory_type, category, content, importance)
                    VALUES ('long_term', ?, ?, ?)
                    """,
                    (category, content, importance),
                )
                memory_id = cursor.lastrowid
            memory_ids.append(memory_id)
        connection.commit()
        return memory_ids
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def search_long_term_memories(
    query: str,
    limit: int = 5,
    db_path: str | Path = DATABASE_PATH,
) -> list[dict[str, object]]:
    """Return active, unexpired long-term memories relevant to a query."""
    if limit < 1:
        raise ValueError("Memory search limit must be at least 1.")

    terms = re.findall(r"\w+", query, flags=re.UNICODE)
    connection = get_connection(db_path)
    try:
        if terms:
            match_query = " OR ".join(f'"{term}"' for term in terms)
            rows = connection.execute(
                """
                SELECT m.id, m.category, m.content, m.importance
                FROM memories_fts
                JOIN memories AS m ON m.id = memories_fts.rowid
                WHERE memories_fts MATCH ?
                  AND m.memory_type = 'long_term'
                  AND m.status = 'active'
                  AND (m.expires_at IS NULL OR m.expires_at > datetime('now'))
                ORDER BY bm25(memories_fts), m.importance DESC, m.updated_at DESC
                LIMIT ?
                """,
                (match_query, limit),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT id, category, content, importance
                FROM memories
                WHERE memory_type = 'long_term'
                  AND status = 'active'
                  AND (expires_at IS NULL OR expires_at > datetime('now'))
                ORDER BY importance DESC, updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()
