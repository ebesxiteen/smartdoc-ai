# db.py
import sqlite3
import uuid
import json
from typing import List, Dict, Optional
from core.configs import (
    NOTEBOOK_DEFAULT_NAME,
    SOURCE_DEFAULT_NAME,
    NOTE_DEFAULT_TITLE,
)
import hashlib

DB_PATH = "./data/smartdoc.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Returns rows as dictionaries
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


# ==========================================
# NOTEBOOKS
# ==========================================
def create_notebook(name: Optional[str], description: Optional[str] = None) -> str:
    notebook_id = str(uuid.uuid4())
    name = name if name else NOTEBOOK_DEFAULT_NAME

    with get_connection() as conn:
        conn.execute(
            "INSERT INTO notebooks (id, name, description) VALUES (?, ?, ?)",
            (notebook_id, name, description),
        )

    return notebook_id


def get_all_notebooks() -> List[Dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM notebooks ORDER BY created_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]


def get_notebook(notebook_id: str) -> Optional[Dict]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM notebooks WHERE id = ?", (notebook_id,)
        ).fetchone()
        return dict(row) if row else None


def delete_notebook(notebook_id: str):
    with get_connection() as conn:
        conn.execute("DELETE FROM notebooks WHERE id = ?", (notebook_id,))


# ==========================================
# SOURCES (Documents)
# ==========================================
def add_source(
    notebook_id: str,
    file_name: Optional[str],
    file_path: str,
    summary: Optional[str] = None,
    suggested_questions: Optional[List[str]] = None,
) -> str:
    source_id = str(uuid.uuid4())
    file_name = file_name if file_name else SOURCE_DEFAULT_NAME

    suggested_questions_str = (
        json.dumps(suggested_questions) if suggested_questions else None
    )

    file_hash = hashlib.md5(file_name.encode()).hexdigest()  # Simplistic hash

    with get_connection() as conn:
        conn.execute(
            "INSERT INTO sources (id, notebook_id, file_name, file_path, file_hash, summary, suggested_questions) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                source_id,
                notebook_id,
                file_name,
                file_path,
                file_hash,
                summary,
                suggested_questions_str,
            ),
        )

    return source_id


def get_sources_for_notebook(notebook_id: str) -> List[Dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM sources WHERE notebook_id = ? ORDER BY created_at ASC",
            (notebook_id,),
        ).fetchall()

        sources = []
        for row in rows:
            src = dict(row)
            src["suggested_questions"] = (
                json.loads(src["suggested_questions"])
                if src["suggested_questions"]
                else []
            )
            sources.append(src)
        return sources


def delete_source(source_id: str):
    with get_connection() as conn:
        conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))


# ==========================================
# CHAT MESSAGES
# ==========================================
def add_chat_message(
    notebook_id: str, role: str, content: str, sources: Optional[List[Dict]] = None
) -> str:
    msg_id = str(uuid.uuid4())
    sources_str = json.dumps(sources) if sources else None

    with get_connection() as conn:
        conn.execute(
            "INSERT INTO chat_messages (id, notebook_id, role, content, sources) VALUES (?, ?, ?, ?, ?)",
            (msg_id, notebook_id, role, content, sources_str),
        )
    return msg_id


def get_chat_history(notebook_id: str) -> List[Dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM chat_messages WHERE notebook_id = ? ORDER BY created_at ASC",
            (notebook_id,),
        ).fetchall()

        history = []
        for row in rows:
            msg = dict(row)
            msg["sources"] = json.loads(msg["sources"]) if msg["sources"] else None
            history.append(msg)
        return history


# ==========================================
# NOTES
# ==========================================
def add_note(notebook_id: str, title: Optional[str], content: str) -> str:
    note_id = str(uuid.uuid4())
    title = title if title else NOTE_DEFAULT_TITLE

    with get_connection() as conn:
        conn.execute(
            "INSERT INTO notes (id, notebook_id, title, content) VALUES (?, ?, ?, ?)",
            (note_id, notebook_id, title, content),
        )

    return note_id


def get_notes_for_notebook(notebook_id: str) -> List[Dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM notes WHERE notebook_id = ? ORDER BY created_at DESC",
            (notebook_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def delete_note(note_id: str):
    with get_connection() as conn:
        conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
