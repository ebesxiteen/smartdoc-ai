import logging
from typing import Optional, List, Dict, Any
from db import crud
from core.utils import clean_spaces
import core.configs as cfg

logger = logging.getLogger(__name__)


def validate_and_clean_text(
    text: Optional[str],
    max_len: Optional[int],
    required: bool = False,
    field_name: str = "Field",
) -> Optional[str]:
    """Helper to clean spaces and validate length."""
    if text is None:
        if required:
            raise ValueError(f"{field_name} is required.")
        return None

    cleaned = clean_spaces(text)
    if required and not cleaned:
        raise ValueError(f"{field_name} cannot be empty.")

    if max_len is not None and len(cleaned) > max_len:
        raise ValueError(
            f"{field_name} must be {max_len} characters or fewer. (Provided {len(cleaned)} characters)"
        )

    return cleaned


def create_notebook(name: Optional[str], description: Optional[str]) -> str:
    """Validates notebook name and description before creating."""
    cleaned_name = validate_and_clean_text(
        name, cfg.MAX_NOTEBOOK_NAME_LEN, required=True, field_name="Notebook Name"
    )
    cleaned_desc = validate_and_clean_text(
        description, cfg.MAX_DESCRIPTION_LEN, required=False, field_name="Description"
    )
    return crud.create_notebook(cleaned_name, cleaned_desc)


def get_all_notebooks() -> List[Dict[str, Any]]:
    return crud.get_all_notebooks()


def get_notebook(notebook_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a notebook by ID without validation since ID is system-generated."""
    return crud.get_notebook(notebook_id)


def delete_notebook(notebook_id: str) -> bool:
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
            new_name,
            cfg.MAX_NOTEBOOK_NAME_LEN,
            required=True,
            field_name="Notebook Name",
        )

    # Validate description if provided
    cleaned_desc = None
    if new_description is not None:
        cleaned_desc = validate_and_clean_text(
            new_description,
            cfg.MAX_DESCRIPTION_LEN,
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
    file_type: str,
    file_path: str,
    file_hash: str,
    summary: Optional[str],
    suggested_questions: Optional[List[str]],
    source_id: Optional[str] = None,
) -> str:
    cleaned_filename = validate_and_clean_text(
        file_name, cfg.MAX_FILENAME_LEN, required=True, field_name="Filename"
    )

    cleaned_filetype = validate_and_clean_text(
        file_type, 50, required=True, field_name="File Type"
    )

    cleaned_summary = validate_and_clean_text(
        summary, None, required=False, field_name="Summary"
    )

    if suggested_questions:
        cleaned_questions: List[str] = []
        for q in suggested_questions:
            cleaned_q = validate_and_clean_text(
                q,
                None,
                required=True,
                field_name="Suggested Question",
            )
            if cleaned_q:
                cleaned_questions.append(cleaned_q)
        suggested_questions = cleaned_questions

    return crud.add_source(
        notebook_id,
        cleaned_filename,
        cleaned_filetype or "pdf",
        file_path,
        file_hash,
        cleaned_summary,
        suggested_questions,
        source_id,
    )


def get_sources_for_notebook(notebook_id: str) -> List[Dict[str, Any]]:
    return crud.get_sources_for_notebook(notebook_id)


def delete_source(source_id: str) -> bool:
    return crud.delete_source(source_id)


def rename_source(source_id: str, new_file_name: Optional[str] = None) -> None:
    """Validates and rename a source file_name."""
    cleaned_name = None
    if new_file_name is not None:
        cleaned_name = validate_and_clean_text(
            new_file_name, cfg.MAX_FILENAME_LEN, required=True, field_name="Filename"
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
    confidence_score: Optional[float] = None,
    reasoning_trace: Optional[List[str]] = None,
) -> str:
    if role not in [cfg.USER_ROLE_NAME, cfg.ASSISTANT_ROLE_NAME]:
        raise ValueError(
            f"Invalid role: {role}. Must be '{cfg.USER_ROLE_NAME}' or '{cfg.ASSISTANT_ROLE_NAME}'."
        )

    if not clean_spaces(content):
        raise ValueError("Content cannot be empty.")

    return crud.add_chat_message(
        notebook_id,
        role,
        content,
        sources,
        found_answer,
        confidence_score,
        reasoning_trace,
    )


def get_chat_history(notebook_id: str) -> List[Dict[str, Any]]:
    return crud.get_chat_history(notebook_id)


def delete_chat_history(notebook_id: str) -> bool:
    """Delete all chat messages for a notebook."""
    return crud.delete_chat_messages(notebook_id)


def add_note(notebook_id: str, title: str, content: str) -> str:
    cleaned_title = validate_and_clean_text(
        title, cfg.MAX_NOTE_TITLE_LEN, required=True, field_name="Note Title"
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
            title, cfg.MAX_NOTE_TITLE_LEN, required=True, field_name="Note Title"
        )

    return crud.update_note(note_id, cleaned_title, content)


def delete_note(note_id: str) -> bool:
    return crud.delete_note(note_id)


def get_notebook_settings(notebook_id: str) -> Optional[Dict[str, Any]]:
    return crud.get_notebook_settings(notebook_id)


def delete_notebook_settings(notebook_id: str) -> bool:
    return crud.delete_notebook_settings(notebook_id)


def upsert_notebook_settings(notebook_id: str, settings: Dict[str, Any]) -> None:
    # 1. Define the validation schema
    # Format: "key": (min_val, max_val)
    validation_map: Dict[str, tuple[int | float, int | float]] = {
        "rag_final_context_k": (
            cfg.RAG_FINAL_CONTEXT_K_MIN,
            cfg.RAG_FINAL_CONTEXT_K_MAX,
        ),
        "rag_rerank_top_n": (cfg.RAG_RERANK_TOP_N_MIN, cfg.RAG_RERANK_TOP_N_MAX),
        "rag_retrieval_min_results": (
            cfg.RAG_RETRIEVAL_MIN_RESULTS_MIN,
            cfg.RAG_RETRIEVAL_MIN_RESULTS_MAX,
        ),
        "rag_retrieval_score_threshold": (
            cfg.RAG_RETRIEVAL_SCORE_THRESHOLD_MIN,
            cfg.RAG_RETRIEVAL_SCORE_THRESHOLD_MAX,
        ),
        "rag_max_chunk_len": (cfg.RAG_MAX_CHUNK_LEN_MIN, cfg.RAG_MAX_CHUNK_LEN_MAX),
        "rag_chunk_overlap": (cfg.RAG_CHUNK_OVERLAP_MIN, cfg.RAG_CHUNK_OVERLAP_MAX),
        "rag_max_ctx_len": (cfg.RAG_MAX_CTX_LEN_MIN, cfg.RAG_MAX_CTX_LEN_MAX),
        "max_msg_history": (cfg.MAX_MSG_HISTORY_MIN, cfg.MAX_MSG_HISTORY_MAX),
        "llm_num_ctx": (cfg.LLM_NUM_CTX_MIN, cfg.LLM_NUM_CTX_MAX),
        "avg_llm_temp": (cfg.LLM_AVG_TEMP_MIN, cfg.LLM_AVG_TEMP_MAX),
        "weight_semantic": (cfg.WEIGHT_SEMANTIC_MIN, cfg.WEIGHT_SEMANTIC_MAX),
        "self_rag_max_depth": (cfg.SELF_RAG_MAX_DEPTH_MIN, cfg.SELF_RAG_MAX_DEPTH_MAX),
        "self_rag_candidates": (
            cfg.SELF_RAG_CANDIDATES_MIN,
            cfg.SELF_RAG_CANDIDATES_MAX,
        ),
        "self_rag_max_retries_per_hop": (
            cfg.SELF_RAG_MAX_RETRIES_PER_HOP_MIN,
            cfg.SELF_RAG_MAX_RETRIES_PER_HOP_MAX,
        ),
        "self_rag_threshold_issup": (
            cfg.SELF_RAG_THRESHOLD_ISSUP_MIN,
            cfg.SELF_RAG_THRESHOLD_ISSUP_MAX,
        ),
        "self_rag_threshold_isrel": (
            cfg.SELF_RAG_THRESHOLD_ISREL_MIN,
            cfg.SELF_RAG_THRESHOLD_ISREL_MAX,
        ),
        "self_rag_threshold_isuse": (
            cfg.SELF_RAG_THRESHOLD_ISUSE_MIN,
            cfg.SELF_RAG_THRESHOLD_ISUSE_MAX,
        ),
    }

    # 2. Iterate and Validate
    for key, (min_val, max_val) in validation_map.items():
        value = settings.get(key)

        # Only validate if the key is actually present in the dict
        if value is not None and isinstance(value, (int, float)):
            if not (min_val <= value <= max_val):
                raise ValueError(
                    f"Invalid {key}: {value}. Must be between {min_val} and {max_val}."
                )

    # 2b. Cross-parameter constraint: Initial Retrieval Pool must cover Final Context size
    # (rag_rerank_top_n is the wide funnel; rag_final_context_k is how many reach the LLM)
    rerank_top_n = settings.get("rag_rerank_top_n")
    final_context_k = settings.get("rag_final_context_k")
    if (
        rerank_top_n is not None
        and final_context_k is not None
        and isinstance(rerank_top_n, (int, float))
        and isinstance(final_context_k, (int, float))
        and rerank_top_n < final_context_k
    ):
        raise ValueError(
            f"Invalid configuration: Initial Retrieval Pool (rag_rerank_top_n={rerank_top_n}) "
            f"must be >= Final LLM Context (rag_final_context_k={final_context_k}). "
            f"The retrieval pool is the input to the re-ranking funnel — it must be at least as large as the desired output."
        )

    # 3. Save to Database
    try:
        crud.upsert_notebook_settings(notebook_id, settings)
    except Exception as e:
        # Wrap database errors in a more descriptive context
        raise RuntimeError(
            f"Failed to update settings for notebook {notebook_id}: {str(e)}"
        )
