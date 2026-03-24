import logging
from typing import Optional, List, Dict, Any
from db import crud
from core.utils import clean_spaces
from core.configs import (
    USER_ROLE_NAME,
    ASSISTANT_ROLE_NAME,
    MAX_NOTE_TITLE_LEN,
    MAX_FILENAME_LEN,
    MAX_DESCRIPTION_LEN,
    MAX_NOTEBOOK_NAME_LEN,
    MAX_SOURCE_SUMMARY_LEN,
    MAX_SUGGESTED_QUESTION_LEN,
)

logger = logging.getLogger(__name__)


def validate_and_clean_text(
    text: Optional[str], max_len: int, required: bool = False, field_name: str = "Field"
) -> Optional[str]:
    """Helper to clean spaces and validate length."""
    if text is None:
        if required:
            raise ValueError(f"{field_name} is required.")
        return None

    cleaned = clean_spaces(text)
    if required and not cleaned:
        raise ValueError(f"{field_name} cannot be empty.")

    if len(cleaned) > max_len:
        logger.warning(
            f"{field_name} truncated: '{cleaned[:20]}...' exceeded length {max_len}"
        )
        return cleaned[:max_len]

    return cleaned


def create_notebook(name: Optional[str], description: Optional[str]) -> str:
    """Validates notebook name and description before creating."""
    cleaned_name = validate_and_clean_text(
        name, MAX_NOTEBOOK_NAME_LEN, required=True, field_name="Notebook Name"
    )
    cleaned_desc = validate_and_clean_text(
        description, MAX_DESCRIPTION_LEN, required=False, field_name="Description"
    )
    return crud.create_notebook(cleaned_name, cleaned_desc)


def get_all_notebooks() -> List[Dict[str, Any]]:
    return crud.get_all_notebooks()


def get_notebook(notebook_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a notebook by ID without validation since ID is system-generated."""
    return crud.get_notebook(notebook_id)


def delete_notebook(notebook_id: str) -> None:
    return crud.delete_notebook(notebook_id)


def rename_notebook(
    notebook_id: str,
    new_name: Optional[str] = None,
    new_description: Optional[str] = None,
) -> None:
    """Validates and renames a notebook name and/or description."""

    # Validate name if provided
    cleaned_name = None
    if new_name is not None:
        cleaned_name = validate_and_clean_text(
            new_name, MAX_NOTEBOOK_NAME_LEN, required=True, field_name="Notebook Name"
        )

    # Validate description if provided
    cleaned_desc = None
    if new_description is not None:
        cleaned_desc = validate_and_clean_text(
            new_description,
            MAX_DESCRIPTION_LEN,
            required=False,
            field_name="Description",
        )

    # At least one field must be updated
    if new_name is None and new_description is None:
        raise ValueError(
            "At least one field (name or description) must be provided for update."
        )

    # Call CRUD with validated data
    return crud.update_notebook(notebook_id, cleaned_name, cleaned_desc)


def add_source(
    notebook_id: str,
    file_name: str,
    file_path: str,
    file_hash: str,
    summary: Optional[str],
    suggested_questions: Optional[List[str]],
    source_id: Optional[str] = None,
) -> str:
    cleaned_filename = validate_and_clean_text(
        file_name, MAX_FILENAME_LEN, required=True, field_name="Filename"
    )

    cleaned_summary = validate_and_clean_text(
        summary, MAX_SOURCE_SUMMARY_LEN, required=False, field_name="Summary"
    )

    if suggested_questions:
        cleaned_questions: List[str] = []
        for q in suggested_questions:
            cleaned_q = validate_and_clean_text(
                q,
                MAX_SUGGESTED_QUESTION_LEN,
                required=True,
                field_name="Suggested Question",
            )
            if cleaned_q:
                cleaned_questions.append(cleaned_q)
        suggested_questions = cleaned_questions

    return crud.add_source(
        notebook_id,
        cleaned_filename,
        file_path,
        file_hash,
        cleaned_summary,
        suggested_questions,
        source_id,
    )


def get_sources_for_notebook(notebook_id: str) -> List[Dict[str, Any]]:
    return crud.get_sources_for_notebook(notebook_id)


def delete_source(source_id: str) -> None:
    return crud.delete_source(source_id)


def rename_source(source_id: str, new_file_name: Optional[str] = None) -> None:
    """Validates and rename a source file_name."""
    cleaned_name = None
    if new_file_name is not None:
        cleaned_name = validate_and_clean_text(
            new_file_name, MAX_FILENAME_LEN, required=True, field_name="Filename"
        )

    if cleaned_name is None:
        raise ValueError("Filename cannot be empty.")

    return crud.update_source(source_id, file_name=cleaned_name)


def get_source_by_hash(file_hash: str) -> Optional[Dict[str, Any]]:
    """Check if a file with this hash already exists."""
    return crud.get_source_by_hash(file_hash)


def add_chat_message(
    notebook_id: str,
    role: str,
    content: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    found_answer: Optional[bool] = None,
) -> str:
    if role not in [USER_ROLE_NAME, ASSISTANT_ROLE_NAME]:
        raise ValueError(
            f"Invalid role: {role}. Must be '{USER_ROLE_NAME}' or '{ASSISTANT_ROLE_NAME}'."
        )

    if not clean_spaces(content):
        raise ValueError("Content cannot be empty.")

    return crud.add_chat_message(notebook_id, role, content, sources, found_answer)


def get_chat_history(notebook_id: str) -> List[Dict[str, Any]]:
    return crud.get_chat_history(notebook_id)


def delete_chat_history(notebook_id: str) -> None:
    """Delete all chat messages for a notebook."""
    return crud.delete_chat_messages(notebook_id)


def add_note(notebook_id: str, title: str, content: str) -> str:
    cleaned_title = validate_and_clean_text(
        title, MAX_NOTE_TITLE_LEN, required=True, field_name="Note Title"
    )

    # Note content is allowed to be long and maintain whitespace/newlines!
    return crud.add_note(notebook_id, cleaned_title, content)


def get_notes_for_notebook(notebook_id: str) -> List[Dict[str, Any]]:
    return crud.get_notes_for_notebook(notebook_id)


def update_note(
    note_id: str, title: Optional[str] = None, content: Optional[str] = None
) -> None:
    cleaned_title = None
    if title is not None:
        cleaned_title = validate_and_clean_text(
            title, MAX_NOTE_TITLE_LEN, required=True, field_name="Note Title"
        )

    return crud.update_note(note_id, cleaned_title, content)


def delete_note(note_id: str) -> None:
    return crud.delete_note(note_id)
