import logging
from typing import Optional, List, Dict
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


def get_all_notebooks() -> List[Dict]:
    return crud.get_all_notebooks()


def get_notebook(notebook_id: str) -> Optional[Dict]:
    """Retrieves a notebook by ID without validation since ID is system-generated."""
    return crud.get_notebook(notebook_id)


def delete_notebook(notebook_id: str):
    return crud.delete_notebook(notebook_id)


def add_source(
    notebook_id: str,
    file_name: str,
    file_path: str,
    summary: Optional[str],
    suggested_questions: Optional[List[str]],
) -> str:
    cleaned_filename = validate_and_clean_text(
        file_name, MAX_FILENAME_LEN, required=True, field_name="Filename"
    )

    cleaned_summary = validate_and_clean_text(
        summary, MAX_SOURCE_SUMMARY_LEN, required=False, field_name="Summary"
    )

    if suggested_questions:
        cleaned_questions = []
        for q in suggested_questions:
            cleaned_q = validate_and_clean_text(
                q,
                MAX_SUGGESTED_QUESTION_LEN,
                required=True,
                field_name="Suggested Question",
            )
            cleaned_questions.append(cleaned_q)
        suggested_questions = cleaned_questions

    return crud.add_source(
        notebook_id, cleaned_filename, file_path, cleaned_summary, suggested_questions
    )


def get_sources_for_notebook(notebook_id: str) -> List[Dict]:
    return crud.get_sources_for_notebook(notebook_id)


def delete_source(source_id: str):
    return crud.delete_source(source_id)


def add_chat_message(notebook_id: str, role: str, content: str, sources: Optional[List[Dict]] = None) -> str:
    if role not in [USER_ROLE_NAME, ASSISTANT_ROLE_NAME]:
        raise ValueError(
            f"Invalid role: {role}. Must be '{USER_ROLE_NAME}' or '{ASSISTANT_ROLE_NAME}'."
        )

    if not clean_spaces(content):
        raise ValueError("Content cannot be empty.")

    return crud.add_chat_message(notebook_id, role, content, sources)


def get_chat_history(notebook_id: str) -> List[Dict]:
    return crud.get_chat_history(notebook_id)


def add_note(notebook_id: str, title: str, content: str) -> str:
    cleaned_title = validate_and_clean_text(
        title, MAX_NOTE_TITLE_LEN, required=True, field_name="Note Title"
    )
    if not clean_spaces(content):
        raise ValueError("Note content cannot be empty.")

    # Note content is allowed to be long and maintain whitespace/newlines!
    return crud.add_note(notebook_id, cleaned_title, content)


def get_notes_for_notebook(notebook_id: str) -> List[Dict]:
    return crud.get_notes_for_notebook(notebook_id)


def delete_note(note_id: str):
    return crud.delete_note(note_id)
