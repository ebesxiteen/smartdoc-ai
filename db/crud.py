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


def delete_notebook(notebook_id: str) -> bool:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM notebooks WHERE id = ?", (notebook_id,))
        return cursor.rowcount > 0


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
    file_type: str,
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
        file_type: Type of the file (e.g., 'pdf', 'docx')
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
            "INSERT INTO sources (id, notebook_id, file_name, file_type, file_path, file_hash, summary, suggested_questions) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                source_id,
                notebook_id,
                file_name,
                file_type,
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


def delete_source(source_id: str) -> bool:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sources WHERE id = ?", (source_id,))
        return cursor.rowcount > 0


def update_source(
    source_id: str, file_name: Optional[str] = None, summary: Optional[str] = None
) -> None:
    """Update source file_name and/or summary."""
    with get_connection() as conn:
        # Build dynamic UPDATE statement
        updates: list[str] = []
        params: list[str] = []

        if file_name is not None:
            updates.append("file_name = ?")
            params.append(file_name)

        if summary is not None:
            updates.append("summary = ?")
            params.append(summary)

        if not updates:
            return  # Nothing to update

        params.append(source_id)

        sql = f"UPDATE sources SET {', '.join(updates)}, updated_at = CURRENT_TIMESTAMP WHERE id = ?"
        conn.execute(sql, params)


# ==========================================
# CHAT MESSAGES
# ==========================================
def add_chat_message(
    notebook_id: str,
    role: str,
    content: str,
    self_rag_content: Optional[str] = None,
    self_rag_sources: Optional[List[Dict[str, Any]]] = None,
    self_rag_found_answer: Optional[bool] = None,
    self_rag_confidence_score: Optional[float] = None,
    self_rag_reasoning_trace: Optional[List[str]] = None,
    co_rag_content: Optional[str] = None,
    co_rag_sources: Optional[List[Dict[str, Any]]] = None,
    co_rag_found_answer: Optional[bool] = None,
    co_rag_reasoning_trace: Optional[List[Any]] = None,
) -> str:
    msg_id = str(uuid.uuid4())
    self_rag_sources_str = json.dumps(self_rag_sources) if self_rag_sources else None
    self_rag_reasoning_trace_str = (
        json.dumps(self_rag_reasoning_trace) if self_rag_reasoning_trace else None
    )
    co_rag_sources_str = json.dumps(co_rag_sources) if co_rag_sources else None
    co_rag_reasoning_trace_str = (
        json.dumps(co_rag_reasoning_trace) if co_rag_reasoning_trace else None
    )

    # Convert bool to int for SQLite (True=1, False=0, None=NULL)
    if self_rag_found_answer is None:
        self_rag_found_answer_int = None
    else:
        self_rag_found_answer_int = 1 if self_rag_found_answer else 0

    if co_rag_found_answer is None:
        co_rag_found_answer_int = None
    else:
        co_rag_found_answer_int = 1 if co_rag_found_answer else 0

    with get_connection() as conn:
        conn.execute(
            """INSERT INTO chat_messages (
                id, notebook_id, role, content,
                self_rag_content, self_rag_sources, self_rag_found_answer,
                self_rag_confidence_score, self_rag_reasoning_trace,
                co_rag_content, co_rag_sources, co_rag_found_answer, co_rag_reasoning_trace
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                msg_id,
                notebook_id,
                role,
                content,
                self_rag_content,
                self_rag_sources_str,
                self_rag_found_answer_int,
                self_rag_confidence_score,
                self_rag_reasoning_trace_str,
                co_rag_content,
                co_rag_sources_str,
                co_rag_found_answer_int,
                co_rag_reasoning_trace_str,
            ),
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

            # Parse Self-RAG columns from JSON
            msg["self_rag_sources"] = (
                json.loads(msg["self_rag_sources"])
                if msg.get("self_rag_sources")
                else None
            )
            msg["self_rag_reasoning_trace"] = (
                json.loads(msg["self_rag_reasoning_trace"])
                if msg.get("self_rag_reasoning_trace")
                else None
            )

            # Parse Co-RAG columns from JSON
            msg["co_rag_sources"] = (
                json.loads(msg["co_rag_sources"]) if msg.get("co_rag_sources") else None
            )
            msg["co_rag_reasoning_trace"] = (
                json.loads(msg["co_rag_reasoning_trace"])
                if msg.get("co_rag_reasoning_trace")
                else None
            )

            # Convert self_rag_found_answer from int back to bool (1=True, 0=False, None=None)
            if "self_rag_found_answer" in msg:
                msg["self_rag_found_answer"] = (
                    None
                    if msg["self_rag_found_answer"] is None
                    else bool(msg["self_rag_found_answer"])
                )
            else:
                msg["self_rag_found_answer"] = True  # Default for legacy messages

            # Convert co_rag_found_answer from int back to bool
            if msg.get("co_rag_found_answer") is not None:
                msg["co_rag_found_answer"] = bool(msg["co_rag_found_answer"])

            # Reconstruct confidence_metrics for the UI (from self_rag_confidence_score column)
            if msg.get("self_rag_confidence_score") is not None:
                msg["confidence_metrics"] = {
                    "total_score": msg["self_rag_confidence_score"]
                }

            history.append(msg)
        return history


def delete_chat_messages(notebook_id: str) -> bool:
    """Delete all chat messages for a notebook."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM chat_messages WHERE notebook_id = ?", (notebook_id,)
        )
        return cursor.rowcount > 0


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


def update_note(
    note_id: str, title: Optional[str] = None, content: Optional[str] = None
) -> None:
    updates: list[str] = []
    params: list[str] = []

    if title is not None:
        updates.append("title = ?")
        params.append(title)

    if content is not None:
        updates.append("content = ?")
        params.append(content)

    if not updates:
        return

    params.append(note_id)
    sql = f"UPDATE notes SET {', '.join(updates)}, updated_at = CURRENT_TIMESTAMP WHERE id = ?"

    with get_connection() as conn:
        conn.execute(sql, params)


def delete_note(note_id: str) -> bool:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        return cursor.rowcount > 0


def get_notebook_settings(notebook_id: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM notebook_settings WHERE notebook_id = ?", (notebook_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def delete_notebook_settings(notebook_id: str) -> bool:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM notebook_settings WHERE notebook_id = ?", (notebook_id,)
        )
        return cursor.rowcount > 0


def upsert_notebook_settings(notebook_id: str, settings: Dict[str, Any]) -> None:
    setting_id = str(uuid.uuid4())

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO notebook_settings (
                id, notebook_id, rag_final_context_k, rag_rerank_top_n, rag_retrieval_min_results,
                rag_retrieval_score_threshold, rag_max_chunk_len,
                rag_chunk_overlap, rag_max_ctx_len, max_msg_history,
                llm_model_name, llm_num_ctx, llm_avg_temp, personal_ctx,
                weight_semantic, weight_bm25, self_rag_max_depth, self_rag_candidates,
                self_rag_max_retries_per_hop, self_rag_threshold_issup, self_rag_threshold_isrel,
                self_rag_threshold_isuse, co_rag_max_retries, updated_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP
            ) ON CONFLICT(notebook_id) DO UPDATE SET
                rag_final_context_k=excluded.rag_final_context_k,
                rag_rerank_top_n=excluded.rag_rerank_top_n,
                rag_retrieval_min_results=excluded.rag_retrieval_min_results,
                rag_retrieval_score_threshold=excluded.rag_retrieval_score_threshold,
                rag_max_chunk_len=excluded.rag_max_chunk_len,
                rag_chunk_overlap=excluded.rag_chunk_overlap,
                rag_max_ctx_len=excluded.rag_max_ctx_len,
                max_msg_history=excluded.max_msg_history,
                llm_model_name=excluded.llm_model_name,
                llm_num_ctx=excluded.llm_num_ctx,
                llm_avg_temp=excluded.llm_avg_temp,
                personal_ctx=excluded.personal_ctx,
                weight_semantic=excluded.weight_semantic,
                weight_bm25=excluded.weight_bm25,
                self_rag_max_depth=excluded.self_rag_max_depth,
                self_rag_candidates=excluded.self_rag_candidates,
                self_rag_max_retries_per_hop=excluded.self_rag_max_retries_per_hop,
                self_rag_threshold_issup=excluded.self_rag_threshold_issup,
                self_rag_threshold_isrel=excluded.self_rag_threshold_isrel,
                self_rag_threshold_isuse=excluded.self_rag_threshold_isuse,
                co_rag_max_retries=excluded.co_rag_max_retries,
                updated_at=CURRENT_TIMESTAMP;
            """,
            (
                setting_id,
                notebook_id,
                settings.get("rag_final_context_k"),
                settings.get("rag_rerank_top_n"),
                settings.get("rag_retrieval_min_results"),
                settings.get("rag_retrieval_score_threshold"),
                settings.get("rag_max_chunk_len"),
                settings.get("rag_chunk_overlap"),
                settings.get("rag_max_ctx_len"),
                settings.get("max_msg_history"),
                settings.get("llm_model_name"),
                settings.get("llm_num_ctx"),
                settings.get(
                    "llm_avg_temp"
                ),  # key matches schema column `llm_avg_temp`
                settings.get("personal_ctx"),
                settings.get("weight_semantic"),
                settings.get("weight_bm25"),
                settings.get("self_rag_max_depth"),
                settings.get("self_rag_candidates"),
                settings.get("self_rag_max_retries_per_hop"),
                settings.get("self_rag_threshold_issup"),
                settings.get("self_rag_threshold_isrel"),
                settings.get("self_rag_threshold_isuse"),
                settings.get("co_rag_max_retries"),
            ),
        )
        conn.commit()
