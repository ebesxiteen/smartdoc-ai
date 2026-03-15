# db.py
import sqlite3
import uuid
import json
from typing import List, Dict, Optional, Any
from core.configs import (
    DB_ROOT_PATH,
    NOTEBOOK_DEFAULT_NAME,
    SOURCE_DEFAULT_NAME,
    NOTE_DEFAULT_TITLE,
)


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_ROOT_PATH)
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


def get_all_notebooks() -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM notebooks ORDER BY created_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]


def get_notebook(notebook_id: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM notebooks WHERE id = ?", (notebook_id,)
        ).fetchone()
        return dict(row) if row else None


def delete_notebook(notebook_id: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM notebooks WHERE id = ?", (notebook_id,))


def update_notebook(
    notebook_id: str, name: Optional[str] = None, description: Optional[str] = None
) -> None:
    """Update notebook name and/or description."""
    with get_connection() as conn:
        # Build dynamic UPDATE statement
        updates: list[str] = []
        params: list[str] = []

        if name is not None:
            updates.append("name = ?")
            params.append(name)

        if description is not None:
            updates.append("description = ?")
            params.append(description)

        if not updates:
            return  # Nothing to update

        params.append(notebook_id)

        sql = f"UPDATE notebooks SET {', '.join(updates)}, updated_at = CURRENT_TIMESTAMP WHERE id = ?"
        conn.execute(sql, params)


# ==========================================
# SOURCES (Documents)
# ==========================================
def add_source(
    notebook_id: str,
    file_name: Optional[str],
    file_path: str,
    file_hash: str,
    summary: Optional[str] = None,
    suggested_questions: Optional[List[str]] = None,
    source_id: Optional[str] = None,
) -> str:
    """Add a source to the database.

    Args:
        notebook_id: The notebook ID
        file_name: Original filename
        file_path: Path to vectorstore directory (e.g., "./data/vectorstores/nb_xyz/src_abc")
        file_hash: MD5 hash of file content
        summary: Document summary
        suggested_questions: List of suggested questions
        source_id: Optional pre-generated source ID. If None, generates new UUID.

    Returns:
        The source ID
    """
    source_id = source_id if source_id else str(uuid.uuid4())
    file_name = file_name if file_name else SOURCE_DEFAULT_NAME

    suggested_questions_str = (
        json.dumps(suggested_questions) if suggested_questions else None
    )

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


def get_sources_for_notebook(notebook_id: str) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM sources WHERE notebook_id = ? ORDER BY created_at ASC",
            (notebook_id,),
        ).fetchall()

        sources: List[Dict[str, Any]] = []
        for row in rows:
            src = dict(row)
            src["suggested_questions"] = (
                json.loads(src["suggested_questions"])
                if src["suggested_questions"]
                else []
            )
            sources.append(src)
        return sources


def get_source_by_hash(file_hash: str) -> Optional[Dict[str, Any]]:
    """Check if a file with this hash already exists in ANY notebook."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM sources WHERE file_hash = ? LIMIT 1",
            (file_hash,),
        ).fetchone()
        return dict(row) if row else None


def get_source_by_hash_and_notebook(
    file_hash: str, notebook_id: str
) -> Optional[Dict[str, Any]]:
    """Check if a file with this hash already exists in THIS specific notebook."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM sources WHERE file_hash = ? AND notebook_id = ? LIMIT 1",
            (file_hash, notebook_id),
        ).fetchone()
        return dict(row) if row else None


def delete_source(source_id: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))


# ==========================================
# CHAT MESSAGES
# ==========================================
def add_chat_message(
    notebook_id: str,
    role: str,
    content: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    found_answer: Optional[bool] = None,
) -> str:
    msg_id = str(uuid.uuid4())
    sources_str = json.dumps(sources) if sources else None
    # Convert bool to int for SQLite (True=1, False=0, None=NULL)
    found_answer_int = None if found_answer is None else (1 if found_answer else 0)

    with get_connection() as conn:
        conn.execute(
            "INSERT INTO chat_messages (id, notebook_id, role, content, sources, found_answer) VALUES (?, ?, ?, ?, ?, ?)",
            (msg_id, notebook_id, role, content, sources_str, found_answer_int),
        )
    return msg_id


def get_chat_history(notebook_id: str) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM chat_messages WHERE notebook_id = ? ORDER BY created_at ASC",
            (notebook_id,),
        ).fetchall()

        history: List[Dict[str, Any]] = []
        for row in rows:
            msg = dict(row)
            msg["sources"] = json.loads(msg["sources"]) if msg["sources"] else None
            # Convert found_answer from int back to bool (1=True, 0=False, None=None)
            if "found_answer" in msg:
                msg["found_answer"] = (
                    None if msg["found_answer"] is None else bool(msg["found_answer"])
                )
            else:
                msg["found_answer"] = (
                    True  # Default for old messages without this field
                )
            history.append(msg)
        return history


def delete_chat_messages(notebook_id: str) -> None:
    """Delete all chat messages for a notebook."""
    with get_connection() as conn:
        conn.execute("DELETE FROM chat_messages WHERE notebook_id = ?", (notebook_id,))


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


def get_notes_for_notebook(notebook_id: str) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM notes WHERE notebook_id = ? ORDER BY created_at DESC",
            (notebook_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def delete_note(note_id: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
