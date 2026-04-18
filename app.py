# app.py
"""
SmartDoc AI - Local NotebookLM-Inspired Document Intelligence System
A privacy-first RAG application for querying documents with source citations.
"""

import re
import time
import html

import streamlit as st
import os
import logging
import uuid
from typing import List, Dict, Any, Optional, Callable, Tuple, cast
from pathlib import Path

from middlewares import db_middleware
from langchain_core.documents import Document

import core.configs as cfg
from core.utils import (
    debug_log,
    print_breaker,
    get_default_notebook_settings,
    hash_file_content,
    check_file_already_exists_in_notebook,
    reload_vectorstore_and_chain,
    get_source_vectorstore_dir,
    load_notebook_settings,
)
from core.self_rag import create_history_aware_rag_chain

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Reduce noise from external libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# ============================================================================
# CONSTANTS & PATHS
# ============================================================================
DATA_DIR = Path("data")

DATA_DIR.mkdir(exist_ok=True)
CHUNKS_PROGRESS_REGEX = r"chunks \d+ to (\d+) of (\d+)"
_VIEW_SOURCES_LABEL = "✨ View sources"

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title=cfg.APP_NAME,
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    /* Progress Bar */
    div[data-testid="stProgressBar"] {
        height: 6px !important;
        margin-bottom: 10px;
    }
    /* The track (background) of the progress bar */
    div[data-testid="stProgressBar"] > div > div {
        background-color: rgba(128, 128, 128, 0.2) !important;
    }
    /* The filled portion of the progress bar */
    div[data-testid="stProgressBar"] > div > div > div > div {
        background-color: var(--primary-color) !important;
    }
    .progress-status-text {
        font-size: 0.9rem;
        margin-bottom: 5px;
        font-weight: 500;
    }

    /* Compact mode - reduce all padding and spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 2000px;
        margin: 0 auto;
    }

    .main-header {
        font-size: 2.2em;
        font-weight: 600;
        margin-bottom: 0.3em;
        margin-top: 0em;
    }

    .source-citation {
        background-color: rgba(128, 128, 128, 0.05) !important;
        padding: 0.8em;
        border-left: 4px solid var(--primary-color);
        border-radius: 4px;
        margin: 0.5em 0;
        font-size: 0.85em;
    }

    .saved-note {
        background-color: rgba(255, 165, 0, 0.1) !important;
        padding: 0.8em;
        border-left: 4px solid var(--primary-color);
        border-radius: 4px;
        margin: 0.4em 0;
        font-size: 0.85em;
    }

    .error-box {
        background-color: rgba(204, 0, 0, 0.1) !important;
        padding: 0.8em;
        border-left: 4px solid #cc0000;
        border-radius: 4px;
        margin: 0.5em 0;
    }

    /* Reduce chat message spacing */
    .stChatMessage {
        margin-bottom: 0.2rem;
        padding: 0.3rem 0.5rem;
    }

    /* Assistant styling customization */
    div[data-testid="stChatMessage"]:not(:has(svg[title="user"])) div[data-testid="stChatMessageContent"] {
        background-color: rgba(128, 128, 128, 0.05) !important;
        border-radius: 5px 15px 15px 15px;
        padding: 10px 15px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }

    /* Hide the ⋮ dots icon (1st svg) on popover trigger buttons, keep the arrow */
    [data-testid="stPopoverButton"] button svg:nth-of-type(1) {
        display: none !important;
    }
    [data-testid="stPopoverButton"] button {
        border: none !important;
        background: transparent !important;
    }

    /* Pending New Files: Process & Cancel as link-style buttons.
       Targets the 2nd/3rd columns of the [3,2,2] per-file rows — the only
       3-column horizontal blocks in the sidebar. */
    section[data-testid="stSidebar"]
    [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(3))
    [data-testid="stColumn"]:nth-child(n+2) button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        opacity: 0.8;
        text-decoration: underline !important;
        white-space: nowrap !important;
        font-size: 0.85rem !important;
        cursor: pointer !important;
    }
    section[data-testid="stSidebar"]
    [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(3))
    [data-testid="stColumn"]:nth-child(n+2) button:hover {
        opacity: 1;
        background: transparent !important;
    }

    /* Global layout padding reduction */
    .stApp > header {
        display: none;
    }
    .css-1544g2n {
        padding-top: 1rem;
    }

    /* Main workspace section backgrounds — scoped via hidden marker divs injected
       at the top of each section function. :has() only matches the top-level
       stColumn that directly contains the marker, not nested inner columns. */
    [data-testid="stColumn"]:has(.source-hub-bg) {
        background-color: rgba(128, 128, 128, 0.05) !important;
        border-radius: 10px;
        padding: 0.75rem !important;
    }
    [data-testid="stColumn"]:has(.chat-section-bg) {
        background-color: rgba(128, 128, 128, 0.03) !important;
        border-radius: 10px;
        padding: 0.75rem !important;
    }
    [data-testid="stColumn"]:has(.notes-panel-bg) {
        background-color: rgba(128, 128, 128, 0.05) !important;
        border-radius: 10px;
        padding: 0.75rem !important;
        position: relative !important;
        padding-bottom: 4rem !important; /* reserve space for fixed button */
    }

    /* Target the marker div's parent container and pin the NEXT container (which holds the button) to bottom */
    div[data-testid="stElementContainer"]:has(.add-note-btn-anchor) + div[data-testid="stElementContainer"] {
        position: absolute;
        bottom: 0.75rem;
        left: 0.75rem;
        right: 0.75rem;
        width: auto !important; /* let left/right dictate width */
    }

    /* Target the wrapper for suggested questions and ensure the buttons inside can wrap their text freely */
    div[data-testid="stVerticalBlock"]:has(.suggested-questions-wrapper) [data-testid="stButton"] button {
        height: auto !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        text-align: left !important;
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }
    div[data-testid="stVerticalBlock"]:has(.suggested-questions-wrapper) [data-testid="stButton"] button p {
        white-space: normal !important;
        word-wrap: break-word !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================================
# PERSISTENCE FUNCTIONS
# ============================================================================
def load_chat_history() -> List[Dict[str, Any]]:
    """Load chat history from database."""
    notebook_id = st.session_state.get("current_notebook_id")
    if notebook_id:
        return db_middleware.get_chat_history(notebook_id)
    return []


def load_saved_notes() -> List[Dict[str, Any]]:
    """Load saved notes from database."""
    notebook_id = st.session_state.get("current_notebook_id")
    if notebook_id:
        return db_middleware.get_notes_for_notebook(notebook_id)
    return []


def load_documents_state() -> Dict[str, Any]:
    """Load documents metadata from database. Key by source ID (not filename, to avoid duplicates)."""
    notebook_id = st.session_state.get("current_notebook_id")
    if not notebook_id:
        return {}

    sources = db_middleware.get_sources_for_notebook(notebook_id)
    docs_dict: Dict[str, Any] = {}
    for src in sources:
        # Use source ID as key to avoid collisions when sources have the same filename
        docs_dict[src["id"]] = {
            "id": src["id"],
            "file_name": src["file_name"],
            "loaded": True,
            "summary": src["summary"],
            "suggested_questions": src["suggested_questions"],
        }
    return docs_dict


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
def load_workspace(notebook_id: str, print_debug: bool = False):
    """Load or reload all workspace state for a specific notebook."""
    from core.utils import load_persisted_vectorstore_filtered

    st.session_state.current_notebook_id = notebook_id
    st.session_state.documents = load_documents_state()

    # Initialize selected sources to all sources by default
    sources = db_middleware.get_sources_for_notebook(notebook_id)
    st.session_state.selected_sources = {src["id"] for src in sources}

    st.session_state.vectorstore = load_persisted_vectorstore_filtered(
        notebook_id, st.session_state.selected_sources
    )

    if st.session_state.vectorstore is not None:
        # Use history-aware RAG chain for better context understanding
        st.session_state.rag_chain = create_history_aware_rag_chain(
            vectorstore=st.session_state.vectorstore,
            print_debug=print_debug,
            notebook_id=st.session_state.current_notebook_id,
        )
    else:
        st.session_state.rag_chain = None

    st.session_state.chat_history = load_chat_history()
    st.session_state.saved_notes = load_saved_notes()
    st.session_state.pending_query = None
    st.session_state.file_uploader_key = 0
    st.session_state.pending_new_uploads = {}


def check_ollama_connection() -> bool:
    """
    Check if Ollama LLM server is running and accessible.

    Attempts to connect to the Ollama API server at the configured base URL.
    This is called during initialization to set ollama_ready status.

    Returns:
        bool: True if Ollama server is online and responding, False otherwise.
    """
    import requests

    try:
        response = requests.get(f"{cfg.OLLAMA_BASE_URL}/api/tags")
        if response.status_code == 200:
            return True
        return False
    except Exception as e:
        debug_log("ERROR", message=f"Ollama connection failed: {str(e)}")
        return False


def initialize_session_state():
    """Initialize all global session state variables on first app run."""
    if "notebooks" not in st.session_state:
        st.session_state.notebooks = db_middleware.get_all_notebooks()

    if "current_notebook_id" not in st.session_state:
        st.session_state.current_notebook_id = None
        debug_log("DEBUG", message="Reset notebook ID on first load")

    if "ollama_ready" not in st.session_state:
        st.session_state.ollama_ready = check_ollama_connection()

    if "file_uploader_key" not in st.session_state:
        st.session_state.file_uploader_key = 0

    if "rename_source_modal_open" not in st.session_state:
        st.session_state.rename_source_modal_open = False

    if "show_notes_panel" not in st.session_state:
        st.session_state.show_notes_panel = False

    if "rename_source_id" not in st.session_state:
        st.session_state.rename_source_id = None

    if "rename_source_name" not in st.session_state:
        st.session_state.rename_source_name = None


def generate_summary(chunks: List[Document], notebook_id: str) -> str:
    """
    Generate a brief summary of the document using Top-K slicing strategy.

    Extracts the top K chunks from a document and prompts the LLM to generate
    a concise summary. Used during document upload for quick overview.

    Args:
        chunks: List of Document objects (from PDF extraction/chunking).
        notebook_id: UUID of the notebook this summary belongs to.

    Returns:
        str: Generated summary text, or error message if generation fails.
    """
    from langchain_ollama import OllamaLLM

    try:
        # Get the first k chunks (or fewer if the document is very short)
        top_k_chunks = chunks[: cfg.TOP_K_CHUNKS_FOR_SUMMARY]
        context_text = "\n\n".join([chunk.page_content for chunk in top_k_chunks])

        summary_prompt = cfg.SUMMARY_PROMPT.format(text=context_text)

        # Initialize standalone LLM
        settings = (
            db_middleware.get_notebook_settings(notebook_id)
            or get_default_notebook_settings()
        )
        llm = OllamaLLM(
            model=settings["llm_model_name"],
            base_url=cfg.OLLAMA_BASE_URL,
            temperature=settings["llm_avg_temp"],
            num_ctx=settings["llm_num_ctx"],
        )

        summary = llm.invoke(summary_prompt)

        return summary.strip()
    except Exception as e:
        debug_log("ERROR", message=f"Summary generation failed: {str(e)}")
        return "Unable to generate summary."


def generate_suggested_questions(chunks: List[Document], notebook_id: str) -> List[str]:
    """
    Generate suggested questions based on document content.

    Extracts the top K chunks from a document and prompts the LLM to generate
    3-4 relevant questions that a user might ask about the document. Displayed
    in the UI for quick reference during onboarding.

    Args:
        chunks: List of Document objects (from PDF extraction/chunking).
        notebook_id: UUID of the notebook this belongs to.

    Returns:
        List[str]: List of suggested questions (max 4), or empty list if generation fails.
    """
    from langchain_ollama import OllamaLLM

    try:
        # Get the first k chunks (or fewer if the document is very short)
        top_k_chunks = chunks[: cfg.TOP_K_CHUNKS_FOR_QUESTIONS]
        context_text = "\n\n".join([chunk.page_content for chunk in top_k_chunks])

        question_prompt = cfg.SUGGESTED_QUESTIONS_PROMPT.format(text=context_text)

        # Initialize standalone LLM
        settings = (
            db_middleware.get_notebook_settings(notebook_id)
            or get_default_notebook_settings()
        )
        llm = OllamaLLM(
            model=settings["llm_model_name"],
            base_url=cfg.OLLAMA_BASE_URL,
            temperature=settings["llm_avg_temp"],
            num_ctx=settings["llm_num_ctx"],
        )

        response = llm.invoke(question_prompt)

        questions: List[str] = []
        for line in str(response).split("\n"):
            if line.strip() and (line[0].isdigit() or line.startswith("-")):
                question = line.lstrip("0123456789.-) ").strip()
                if question and len(question) > 5 and "?" in question:
                    questions.append(question)

        return questions[:4]
    except Exception as e:
        debug_log("ERROR", message=f"Question generation failed: {str(e)}")
        return []


def process_file(
    uploaded_file: Any,
    filename: str,
    print_debug: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> bool:
    """
    Process uploaded document: extract, chunk, embed, and integrate with vectorstore.

    Orchestrates the complete ingestion pipeline: file type detection, chunking,
    embedding, vectorstore creation, summary/question generation, and database
    persistence. Handles errors gracefully and provides progress updates.

    Args:
        uploaded_file: Streamlit UploadedFile or bytes object containing document.
        filename: Display name for the document.
        print_debug: If True, emit structured debug logs throughout the process.
        progress_callback: Optional callback function for progress updates (used in UI).

    Returns:
        bool: True if processing succeeded, False if error occurred or duplicate detected.

    Raises:
        Does not raise; errors are caught and reported via st.error() and logs.
    """
    import tempfile
    from core.utils import (
        detect_file_type,
        chunk_and_process_file,
        try_load_embeddings,
        create_vectorstore_from_chunks,
        merge_vectorstores,
        save_source_to_database,
    )

    tmp_path = None
    file_bytes = None
    try:
        # First, read all file bytes to calculate hash
        if hasattr(uploaded_file, "getvalue"):
            file_bytes = uploaded_file.getvalue()
        elif hasattr(uploaded_file, "getbuffer"):
            file_bytes = uploaded_file.getbuffer()
        elif isinstance(uploaded_file, bytes):
            file_bytes = uploaded_file
        else:
            file_bytes = uploaded_file.read()

        file_hash = hash_file_content(file_bytes)

        try:
            file_type = detect_file_type(file_bytes)
        except ValueError as e:
            st.error(f"❌ {e}")
            return False

        # Check if this exact file was already uploaded to THIS notebook
        existing_source = check_file_already_exists_in_notebook(
            file_hash, st.session_state.current_notebook_id
        )
        if existing_source:
            st.error(
                "❌ This file was already uploaded to this notebook. "
                "You can upload it to another notebook, or replace the existing one."
            )
            return False

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        import contextlib

        @contextlib.contextmanager
        def optional_spinner():
            if progress_callback:
                yield
            else:
                with st.spinner(f"Processing '{filename}'..."):
                    yield

        with optional_spinner():
            notebook_id = st.session_state.current_notebook_id
            settings = (
                db_middleware.get_notebook_settings(notebook_id)
                or get_default_notebook_settings()
            )

            # Load and chunk
            chunks, num_chunks = chunk_and_process_file(
                tmp_path,
                file_type,
                filename,
                print_debug,
                progress_callback=progress_callback,
                chunk_size=int(settings["rag_max_chunk_len"]),
                chunk_overlap=int(settings["rag_chunk_overlap"]),
            )

            # Embed with GPU/CPU fallback
            if print_debug:
                debug_log(
                    "EMBED", message=f"Creating embeddings for {num_chunks} chunks"
                )
            if progress_callback:
                progress_callback("Loading embedding model...")
            embeddings = try_load_embeddings()
            if embeddings is None:
                st.error(
                    "❌\tFailed to load embedding model. Please check your system."
                )
                return False

            if progress_callback:
                progress_callback("Creating vector store...")
            new_vectorstore = create_vectorstore_from_chunks(
                chunks, embeddings, print_debug, progress_callback=progress_callback
            )

            # Setup Database / File Paths mapping
            notebook_id = st.session_state.current_notebook_id

            # Generate source_id upfront so we know the vectorstore path
            source_id = str(uuid.uuid4())

            # Calculate vectorstore directory path
            src_vs_dir = get_source_vectorstore_dir(notebook_id, source_id)
            # Store the relative path for persistence
            vectorstore_path = str(src_vs_dir)

            # CRITICAL FIX: Save the individual source vectorstore BEFORE merging it!
            # Langchain's FAISS `merge_from` empties the target index natively.
            new_vectorstore.save_local(vectorstore_path)

            # Merge into the current session vectorstore (or create if first upload)
            st.session_state.vectorstore = merge_vectorstores(
                st.session_state.vectorstore, new_vectorstore, print_debug
            )
            # NOTE: Do NOT rebuild the RAG chain here — reload_vectorstore_and_chain()
            # below already rebuilds it once with the final selected sources set.

            # Generate summary
            if print_debug:
                debug_log(
                    "PROCESS_START", message=f'Generating summary for "{filename}"'
                )
            summary = generate_summary(chunks, notebook_id=notebook_id)

            # Generate suggested questions
            if print_debug:
                debug_log(
                    "PROCESS_START",
                    message=f'Generating suggested questions for "{filename}"',
                )
            suggested_questions = generate_suggested_questions(
                chunks, notebook_id=notebook_id
            )

            # Add source to database with correct vectorstore path
            source_id = save_source_to_database(
                notebook_id,
                filename,
                file_type,
                file_hash,
                summary,
                suggested_questions,
                vectorstore_path,
                source_id=source_id,
                print_debug=print_debug,
            )

            # Update session state - use source_id as key to avoid filename collisions
            st.session_state.documents[source_id] = {
                "id": source_id,
                "file_name": filename,
                "loaded": True,
                "summary": summary,
                "suggested_questions": suggested_questions,
            }

            # Add the new source to selected sources (auto-select)
            st.session_state.selected_sources.add(source_id)

            if (
                "documents" in st.session_state
                and "selected_sources" in st.session_state
            ):
                all_ids = set(st.session_state.documents.keys())
                st.session_state["select_all_sources"] = (
                    st.session_state.selected_sources == all_ids
                )

            # Reload vectorstore with the new selection
            reload_vectorstore_and_chain(
                notebook_id, st.session_state.selected_sources, print_debug
            )

            if print_debug:
                debug_log("SUCCESS", message=f'Document upload completed: "{filename}"')

        st.success(f'✨ Successfully loaded "{filename}"')

        return True

    except Exception as e:
        debug_log("ERROR", message=f"Document processing failed: {filename} - {str(e)}")
        st.error(f"Error processing file: {str(e)}")
        return False

    finally:
        # Guarantee tmp cleanup
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# ============================================================================
# SIDEBAR: SOURCE HUB
# ============================================================================
def go_back_to_notebooks() -> None:
    """
    Reset session state to return to notebooks dashboard.

    Called when user clicks "Back" button in notebook workspace. Clears
    the current notebook ID to show the main dashboard UI.
    """
    st.session_state.current_notebook_id = None
    debug_log("DEBUG", message="Returned to notebooks dashboard")


def source_hub_ui(print_debug: bool = False) -> None:
    """
    Render the Source Hub sidebar for document management.

    Displays and manages uploaded documents (sources) including:
    - File upload interface with progress feedback
    - List of uploaded sources with metadata
    - Rename, delete, view functionality for each source
    - Source selection checkboxes for search filtering
    - Hardware information display

    Args:
        print_debug: If True, emit debug logs during operations.
    """
    from core.utils import get_system_hardware_info

    hw_info = get_system_hardware_info()

    st.markdown(
        '<div class="source-hub-bg" style="display:none"></div>', unsafe_allow_html=True
    )
    # Header with back arrow and title (vertically centered)
    col1, col2 = st.columns([0.5, 3], vertical_alignment="center")
    with col1:
        if st.button(
            "←",
            use_container_width=True,
            help="Back to Notebooks",
            key="back_to_notebooks_btn",
        ):
            go_back_to_notebooks()
            st.rerun()
    with col2:
        st.markdown("## Source Hub")

    # Show notebook description if available
    notebook = db_middleware.get_notebook(st.session_state.current_notebook_id)
    if notebook and notebook.get("description"):
        st.caption(notebook["description"])

    st.markdown("#### Upload Documents")
    uploaded_files = st.file_uploader(
        "Select PDF or DOCX files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.file_uploader_key}",
        label_visibility="collapsed",
    )

    if "pending_replacements" not in st.session_state:
        st.session_state.pending_replacements = {}

    if "pending_new_uploads" not in st.session_state:
        st.session_state.pending_new_uploads = {}

    # Typed alias — mutations propagate back to session_state via dict reference
    pending_repls: Dict[str, bytes] = st.session_state.pending_replacements  # type: ignore
    pending_new: Dict[str, bytes] = st.session_state.pending_new_uploads  # type: ignore

    if uploaded_files:
        new_pending_replacements = False
        new_pending_new = False

        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            file_bytes = uploaded_file.getvalue()
            file_hash = hash_file_content(file_bytes)

            # Check for duplicates by hash (not filename) to handle renamed files correctly
            existing_source = check_file_already_exists_in_notebook(
                file_hash, st.session_state.current_notebook_id
            )

            if existing_source:
                # This file hash already exists (may have been renamed)
                # Use the source ID as the key to handle duplicate filenames
                existing_source_id = str(existing_source.get("id"))
                if existing_source_id not in pending_repls:
                    pending_repls[existing_source_id] = file_bytes
                    new_pending_replacements = True
            else:
                # No duplicate found
                if filename not in pending_new:
                    pending_new[filename] = file_bytes
                    new_pending_new = True

        # Clear the uploader after reading the batch
        st.session_state.file_uploader_key += 1

        if new_pending_replacements or new_pending_new:
            st.rerun()

    # Show pending new uploads (review queue — user can cancel before processing)
    if pending_new:
        st.markdown("##### Pending New Files")
        st.caption("Review files before processing. Click Cancel to skip a file.")

        pending_new_files: List[str] = list(pending_new.keys())

        action_buttons_container = st.empty()
        process_all_clicked = False
        cancel_all_clicked = False

        with action_buttons_container.container():
            col_process_all, col_cancel_all = st.columns(2)
            with col_process_all:
                process_all_clicked = st.button(
                    "Process All",
                    key="process_all_new",
                    use_container_width=True,
                    type="primary",
                )
            with col_cancel_all:
                cancel_all_clicked = st.button(
                    "Cancel All", key="cancel_all_new", use_container_width=True
                )

        files_list_container = st.empty()

        if cancel_all_clicked:
            pending_new.clear()
            st.rerun()

        if process_all_clicked:
            action_buttons_container.empty()
            files_list_container.empty()

            files_to_process = list(pending_new.items())
            num_files = len(files_to_process)

            status_text = st.empty()
            progress_bar = st.progress(0.0)

            for i, (fname, fbytes) in enumerate(files_to_process):
                del pending_new[fname]

                base_progress = i / num_files
                prog_state = {"val": 0.0}

                def _update_multi_status(
                    status_message: str,
                    status_text: Any = status_text,
                    files_to_process: List[Tuple[str, Any]] = files_to_process,
                    i: int = i,
                    num_files: int = num_files,
                    prog_state: Dict[str, float] = prog_state,
                    progress_bar: Any = progress_bar,
                    base_progress: float = base_progress,
                ) -> None:
                    file_list_html = ""
                    for j, (fname_iter, _) in enumerate(files_to_process):
                        if j < i:
                            # Already processed
                            file_list_html += f"<div style='color: green; font-size: 0.85em; margin-bottom: 2px;'>✨ {fname_iter}</div>"
                        elif j == i:
                            # Currently processing
                            file_list_html += f"<div style='color: var(--text-color); font-weight: 500; font-size: 0.85em; margin-bottom: 2px;'>🚀 {fname_iter} <br><span style='font-size: 0.9em; font-weight: normal; margin-left: 15px; opacity: 0.8;'>↳ <i>{status_message}</i></span></div>"
                        else:
                            # Pending
                            file_list_html += f"<div style='color: gray; font-size: 0.85em; margin-bottom: 2px;'>⚡ {fname_iter}</div>"

                    status_text.markdown(
                        f"<div class='progress-status-text' style='margin-bottom: 10px;'><b>Processing {num_files} files:</b><br>{file_list_html}</div>",
                        unsafe_allow_html=True,
                    )

                    max_val_for_file = 0.95 / num_files

                    match: Optional[re.Match[str]] = re.search(
                        CHUNKS_PROGRESS_REGEX, status_message
                    )
                    if match:
                        current_chunk = float(match.group(1))
                        total_chunks = float(match.group(2))
                        prog_state["val"] = max_val_for_file * (
                            0.1 + (current_chunk / total_chunks) * 0.8
                        )
                    else:
                        remaining: float = max_val_for_file - prog_state["val"]
                        prog_state["val"] += remaining * 0.05

                    current_val: float = min(base_progress + prog_state["val"], 1.0)
                    progress_bar.progress(current_val)

                _update_multi_status("Initializing...")

                process_file(
                    fbytes,
                    fname,
                    cfg.PRINT_DEBUG,
                    progress_callback=_update_multi_status,
                )

                progress_bar.progress(min((i + 1) / num_files, 1.0))

            status_text.markdown(
                "<div class='progress-status-text'>⭐ All documents processed successfully!</div>",
                unsafe_allow_html=True,
            )

            time.sleep(1.5)
            progress_bar.empty()
            status_text.empty()
            st.rerun()

        else:
            process_clicked_file = None
            cancel_clicked_file = None

            with files_list_container.container():
                for filename in pending_new_files:
                    col_name, col_proc, col_cancel = st.columns(
                        [3, 2, 2], vertical_alignment="center"
                    )
                    with col_name:
                        st.markdown(
                            f'<p style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin:0;font-size:0.9em;">📄 {filename}</p>',
                            unsafe_allow_html=True,
                        )
                    with col_proc:
                        if st.button(
                            "Process",
                            key=f"process_new_{filename}",
                            use_container_width=True,
                        ):
                            process_clicked_file = filename
                    with col_cancel:
                        if st.button(
                            "Cancel",
                            key=f"cancel_new_{filename}",
                            use_container_width=True,
                        ):
                            cancel_clicked_file = filename

            if cancel_clicked_file:
                del pending_new[cancel_clicked_file]
                st.rerun()

            if process_clicked_file:
                files_list_container.empty()
                action_buttons_container.empty()

                filename = process_clicked_file
                file_bytes_new: bytes = pending_new[filename]
                del pending_new[filename]

                status_text = st.empty()
                progress_bar = st.progress(0.0)

                prog_state = {"val": 0.0}

                def _update_single_status(
                    status_message: str,
                    status_text: Any = status_text,
                    filename: str = filename,
                    prog_state: Dict[str, float] = prog_state,
                    progress_bar: Any = progress_bar,
                ) -> None:
                    status_text.markdown(
                        f"<div class='progress-status-text'><b>Processing:</b> {filename} <br> ↳ <i>{status_message}</i></div>",
                        unsafe_allow_html=True,
                    )

                    match: Optional[re.Match[str]] = re.search(
                        CHUNKS_PROGRESS_REGEX, status_message
                    )
                    if match:
                        current_chunk = float(match.group(1))
                        total_chunks = float(match.group(2))
                        prog_state["val"] = 0.95 * (
                            0.1 + (current_chunk / total_chunks) * 0.8
                        )
                    else:
                        remaining: float = 0.95 - prog_state["val"]
                        prog_state["val"] += remaining * 0.05
                    progress_bar.progress(prog_state["val"])

                process_file(
                    file_bytes_new,
                    filename,
                    cfg.PRINT_DEBUG,
                    progress_callback=_update_single_status,
                )

                progress_bar.progress(1.0)
                status_text.markdown(
                    "<div class='progress-status-text'>[Success] Document processed successfully!</div>",
                    unsafe_allow_html=True,
                )

                time.sleep(1.0)
                progress_bar.empty()
                status_text.empty()
                st.rerun()

    # Show pending replacements if any are saved in session state
    if pending_repls:
        st.markdown("##### Pending Duplicate Files")
        # Copy keys to a list to safely iterate
        pending_files: List[str] = list(pending_repls.keys())

        for source_id in pending_files:
            # Fetch the source info to display its filename
            source_info = st.session_state.documents.get(source_id, {})
            filename = source_info.get("file_name", "Unknown")

            replacement_container = st.empty()
            replace_clicked = False
            cancel_clicked = False

            with replacement_container.container():
                st.warning(f"**{filename}** is already loaded.")
                col_btn1, col_btn2 = st.columns(2)

                with col_btn1:
                    replace_clicked = st.button(
                        "Replace", key=f"replace_{source_id}", use_container_width=True
                    )
                with col_btn2:
                    cancel_clicked = st.button(
                        "Cancel", key=f"cancel_{source_id}", use_container_width=True
                    )

            if cancel_clicked:
                del pending_repls[source_id]
                st.rerun()

            if replace_clicked:
                replacement_container.empty()
                file_bytes: bytes = pending_repls[source_id]

                # Remove the file from pending immediately so it doesn't get processed again
                del pending_repls[source_id]

                status_text = st.empty()
                progress_bar = st.progress(0.0)

                prog_state = {"val": 0.0}

                def _update_replace_status(
                    status_message: str,
                    status_text: Any = status_text,
                    filename: str = filename,
                    prog_state: Dict[str, float] = prog_state,
                    progress_bar: Any = progress_bar,
                ) -> None:
                    status_text.markdown(
                        f"<div class='progress-status-text'><b>Replacing:</b> {filename} <br> ↳ <i>{status_message}</i></div>",
                        unsafe_allow_html=True,
                    )

                    match: Optional[re.Match[str]] = re.search(
                        CHUNKS_PROGRESS_REGEX, status_message
                    )
                    if match:
                        current_chunk = float(match.group(1))
                        total_chunks = float(match.group(2))
                        prog_state["val"] = 0.95 * (
                            0.1 + (current_chunk / total_chunks) * 0.8
                        )
                    else:
                        remaining = 0.95 - prog_state["val"]
                        prog_state["val"] += remaining * 0.05
                    progress_bar.progress(prog_state["val"])

                # Phase 1: Remove old version
                if source_id in st.session_state.documents:
                    if db_middleware.delete_source(source_id):
                        source_dir = get_source_vectorstore_dir(
                            st.session_state.current_notebook_id, source_id
                        )
                        if source_dir.exists():
                            import shutil

                            shutil.rmtree(source_dir, ignore_errors=True)

                        # Remove from selected sources if it was selected
                        st.session_state.selected_sources.discard(source_id)
                        del st.session_state.documents[source_id]

                # Reload updated vectorstore with filtered selection
                reload_vectorstore_and_chain(
                    st.session_state.current_notebook_id,
                    st.session_state.selected_sources,
                    print_debug,
                )

                # Phase 2: Insert the new one
                process_file(
                    file_bytes,
                    filename,
                    True,
                    progress_callback=_update_replace_status,
                )

                progress_bar.progress(1.0)
                status_text.markdown(
                    "<div class='progress-status-text'>[Success] Document replaced successfully!</div>",
                    unsafe_allow_html=True,
                )

                time.sleep(1.0)
                progress_bar.empty()
                status_text.empty()
                st.rerun()

                if cancel_clicked:
                    del pending_repls[source_id]
                    st.rerun()

    st.markdown("#### Loaded Documents")

    if st.session_state.documents:
        # "Select all sources" row with checkbox on the right (matching NotebookLM style)
        all_ids = {di["id"] for di in st.session_state.documents.values()}
        if "selected_sources" in st.session_state:
            st.session_state["select_all_sources"] = (
                len(st.session_state.selected_sources) == len(all_ids)
                and len(all_ids) > 0
            )

        def on_select_all_change():
            """Handle select-all checkbox state change."""
            checked = st.session_state["select_all_sources"]
            all_ids = {di["id"] for di in st.session_state.documents.values()}
            if checked:
                st.session_state.selected_sources = all_ids
                for doc_id in all_ids:
                    st.session_state[f"checkbox_{doc_id}"] = True
                reload_vectorstore_and_chain(
                    st.session_state.current_notebook_id,
                    st.session_state.selected_sources,
                    print_debug,
                )
            else:
                st.session_state.selected_sources = set()
                for doc_id in all_ids:
                    st.session_state[f"checkbox_{doc_id}"] = False
                st.session_state.vectorstore = None
                st.session_state.rag_chain = None

        col_text, col_check = st.columns([3, 0.5], vertical_alignment="center")
        with col_text:
            st.markdown("**Select all sources**")
        with col_check:
            st.checkbox(
                "Select all sources",
                key="select_all_sources",
                on_change=on_select_all_change,
                label_visibility="collapsed",
            )

        # Display sources with checkbox on the RIGHT (matching NotebookLM style)
        for source_id, doc_info in st.session_state.documents.items():
            # source_id is the key (UUID), get filename from doc_info
            file_name = doc_info.get("file_name", "Unknown")

            # Initialize checkbox state if needed
            checkbox_key = f"checkbox_{source_id}"
            if checkbox_key not in st.session_state:
                st.session_state[checkbox_key] = (
                    source_id in st.session_state.selected_sources
                )

            # Callback to handle checkbox changes
            def on_checkbox_change(src_id: str, checkbox_key_param: str):
                """Handle checkbox state change."""
                if st.session_state[checkbox_key_param]:
                    st.session_state.selected_sources.add(src_id)
                else:
                    st.session_state.selected_sources.discard(src_id)

                # Reload vectorstore
                reload_vectorstore_and_chain(
                    st.session_state.current_notebook_id,
                    st.session_state.selected_sources,
                    print_debug,
                )

                # Update "Select all sources" checkbox state based on current selection
                all_ids = set(st.session_state.documents.keys())
                st.session_state["select_all_sources"] = (
                    st.session_state.selected_sources == all_ids
                )

            # Source row: expander on left, checkbox on right (no border)
            with st.container(border=False):
                col_expand, col_check = st.columns(
                    [3, 0.5], vertical_alignment="center"
                )

                with col_expand:
                    # Expandable details section
                    expander_title = f"📄 {file_name[:40] + '...' if len(file_name) > 40 else file_name}"
                    with st.expander(expander_title, expanded=False):
                        if len(file_name) > 40:
                            st.caption(f"**Full Name:** {file_name}")
                        if doc_info.get("summary"):
                            st.caption(f"**Summary:** {doc_info['summary']}")

                        col_detail1, col_detail2 = st.columns([3, 1])
                        with col_detail1:
                            pass  # Space for alignment
                        with col_detail2:
                            with st.popover("⋮", use_container_width=False):
                                if st.button(
                                    "Rename",
                                    key=f"rename_{source_id}",
                                    use_container_width=True,
                                ):
                                    st.session_state.rename_source_modal_open = True
                                    st.session_state.rename_source_id = source_id
                                    st.session_state.rename_source_name = file_name
                                    st.rerun()
                                if st.button(
                                    "Delete",
                                    key=f"delete_{source_id}",
                                    type="secondary",
                                    use_container_width=True,
                                ):
                                    confirm_delete_source_dialog(
                                        source_id,
                                        file_name,
                                        st.session_state.current_notebook_id,
                                        print_debug,
                                    )

                with col_check:
                    # Checkbox on the RIGHT side
                    st.checkbox(
                        label="Select",
                        key=checkbox_key,
                        on_change=on_checkbox_change,
                        args=(source_id, checkbox_key),
                        label_visibility="collapsed",
                    )
    else:
        st.info("🔴 No documents loaded yet.")

    st.markdown("#### System Status")

    if st.session_state.ollama_ready:
        st.success("🟢 Ollama: Connected")
    else:
        st.error("🔴 Ollama: Offline")
        st.caption("Run: ollama serve")

    # Hardware Dashboard
    st.markdown("#### System Hardware")
    col1, col2 = st.columns(2)
    col1.metric("System RAM", f"{hw_info['ram_gb']} GB")
    col2.metric("GPU VRAM", f"{hw_info['vram_gb']} GB" if hw_info["vram_gb"] else "N/A")
    st.caption(
        f"**OS:** {hw_info['os']} | **CPU:** {hw_info['cpu_cores']} Cores | **GPU:** {hw_info['gpu_name']}"
    )

    st.caption(
        "💡 **Tip:** If you experience GPU out-of-memory errors, try selecting fewer documents before asking questions. "
        "Start with 1-2 documents and gradually add more as needed."
    )


# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================
def render_user_message(content: str) -> None:
    """
    Render a user message in the chat interface.

    Displays user-sent messages with right alignment and distinct styling
    to differentiate from AI assistant responses.

    Args:
        content: The user's message text to display.
    """
    escaped_content = html.escape(content).replace("\n", "<br>")
    st.markdown(
        f"""
        <div style="display: flex; justify-content: flex-end; margin-bottom: 0.5rem; width: 100%;">
            <div style="background-color: var(--secondary-background-color); color: var(--text-color); padding: 0.6rem 1rem; border-radius: 15px 5px 15px 15px; border: 1px solid rgba(128, 128, 128, 0.2); max-width: 80%; text-align: left; box-shadow: 0 1px 2px rgba(0,0,0,0.05); font-family: sans-serif; font-size: 0.95em;">
                {escaped_content}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chat_interface(notebook_name: str, print_debug: bool = False) -> None:
    """
    Render the main chat interface for querying documents.

    Displays the chat history area, chat input field, and processes user queries.
    Integrates with the RAG chain to retrieve relevant document chunks and
    generate answers with citations. Handles document selection and chat history.

    Args:
        notebook_name: Display name of the current notebook.
        print_debug: If True, emit debug logs during query processing.
    """
    st.markdown(
        '<div class="chat-section-bg" style="display:none"></div>',
        unsafe_allow_html=True,
    )
    # Header with dropdown menu on the right
    col1, col2 = st.columns([20, 1], vertical_alignment="center")

    # Check if there are any messages to delete
    has_messages = len(st.session_state.chat_history) > 0

    with col1:
        st.markdown("### Chat")

    with col2:
        with st.popover("⋮", use_container_width=False):
            if st.button(
                "Toggle Notes",
                key="toggle_right_sidebar",
                use_container_width=True,
            ):
                st.session_state.show_notes_panel = (
                    not st.session_state.show_notes_panel
                )
                st.rerun()

            if st.button(
                "Delete history",
                key=f"clear_chat_{st.session_state.current_notebook_id}",
                type="secondary",
                use_container_width=True,
                disabled=not has_messages,
            ):
                confirm_delete_chat_history_dialog(st.session_state.current_notebook_id)

    st.caption(f"Notebook: **{notebook_name}**")

    if not st.session_state.documents:
        st.info("🚀 Upload documents in the Source Hub to get started.")
        return

    # Check if any sources are selected
    if not st.session_state.selected_sources:
        st.warning(
            "⭐ No sources selected. Select at least one source in the Source Hub to ask questions."
        )
        return

    # Scrollable chat history container (height creates native scroll, border=False is clean)
    chat_container = st.container(height=640, border=False)

    # Compute source count
    num_selected = len(st.session_state.selected_sources)
    source_text = f"{num_selected} source{'s' if num_selected != 1 else ''}"

    # Display source count as a caption
    st.caption(f"Chatting with {source_text}")

    # Create chat input
    user_query = st.chat_input("Ask a question about your documents...")

    # Handle pending query from suggested question
    if st.session_state.pending_query:
        user_query = st.session_state.pending_query
        st.session_state.pending_query = None

    # Display chat history in the container
    with chat_container:
        suggestions_placeholder = st.empty()

        # Display suggested questions only if chat is empty and no query is pending/running
        if not st.session_state.chat_history and not user_query:
            with suggestions_placeholder.container():
                import random

                # Gather all suggested questions from specifically selected sources
                all_suggested_questions: List[str] = []
                for src_id in st.session_state.selected_sources:
                    doc_info = st.session_state.documents.get(src_id, {})
                    qs = doc_info.get("suggested_questions")
                    if qs:
                        all_suggested_questions.extend(qs)

                if all_suggested_questions:
                    # Deduplicate and pick up to 3 random questions
                    unique_questions = list(set(all_suggested_questions))
                    num_to_pick = min(3, len(unique_questions))
                    selected_qs = random.sample(unique_questions, num_to_pick)

                    st.markdown("#### 💡 Suggested Questions")
                    with st.container():
                        st.markdown(
                            '<div class="suggested-questions-wrapper"></div>',
                            unsafe_allow_html=True,
                        )
                        for question in selected_qs:
                            if st.button(
                                question,
                                use_container_width=True,
                                key=f"suggested_{hash(question)}",
                            ):
                                st.session_state.pending_query = question
                                st.rerun()

        else:
            suggestions_placeholder.empty()

        # Display chat history (proper left/right layout)
        for _, message in enumerate(st.session_state.chat_history):
            if message["role"] == cfg.USER_ROLE_NAME:
                render_user_message(message["content"])
            else:
                # Assistant message
                has_co_rag = bool(message.get("co_rag_content"))
                if has_co_rag:
                    tab_srag, tab_corag = st.tabs(
                        ["⚡ Self-RAG (Vertical)", "⭐ Co-RAG (Horizontal)"]
                    )
                    with tab_srag:
                        st.markdown(
                            message.get("self_rag_content") or message["content"]
                        )
                        if message.get("self_rag_sources") and message.get(
                            "self_rag_found_answer", True
                        ):
                            with st.expander(_VIEW_SOURCES_LABEL, expanded=False):
                                for j, source in enumerate(message["self_rag_sources"]):
                                    st.markdown(
                                        f"""<div class='source-citation'><strong>{j + 1}. {source["document"]} — Page {source["page"]}</strong><br><small>{source["content"]}</small></div>""",
                                        unsafe_allow_html=True,
                                    )
                        if message.get("self_rag_reasoning_trace"):
                            with st.expander(
                                "💭 Self-RAG Reasoning Trace", expanded=False
                            ):
                                st.markdown("**Multi-Hop Retrieval Execution:**")
                                for i, step in enumerate(
                                    message["self_rag_reasoning_trace"], 1
                                ):
                                    step_text = str(step).strip()
                                    if step_text:
                                        st.markdown(f"**Step {i}:** {step_text}")
                                metrics = message.get("confidence_metrics")
                                if metrics:
                                    st.markdown("---")
                                    st.markdown("**Confidence Scores:**")
                                    st.markdown(
                                        f"- **Total Score:** {metrics.get('total_score', 0.0):.2f}"
                                    )
                                    st.markdown(
                                        f"- Groundedness (ISSUP): {metrics.get('issup', 0.0):.2f}"
                                    )
                                    st.markdown(
                                        f"- Relevance (ISREL): {metrics.get('isrel', 0.0):.2f}"
                                    )
                                    st.markdown(
                                        f"- Utility (ISUSE): {metrics.get('isuse', 0.0):.2f}"
                                    )
                    with tab_corag:
                        st.markdown(message["co_rag_content"])
                        if message.get("co_rag_sources") and message.get(
                            "co_rag_found_answer", True
                        ):
                            with st.expander(_VIEW_SOURCES_LABEL, expanded=False):
                                for j, source in enumerate(message["co_rag_sources"]):
                                    st.markdown(
                                        f"""<div class='source-citation'><strong>{j + 1}. {source["document"]} — Page {source["page"]}</strong><br><small>{source["content"]}</small></div>""",
                                        unsafe_allow_html=True,
                                    )
                        if message.get("co_rag_reasoning_trace"):
                            with st.expander("⭐ Co-RAG Review Trace", expanded=False):
                                st.markdown("**Generator↔Reviewer Collaboration:**")
                                for i, step in enumerate(
                                    message["co_rag_reasoning_trace"], 1
                                ):
                                    if isinstance(step, dict):
                                        step_typed = cast(Dict[str, str], step)
                                        step_label = (
                                            step_typed.get("step") or f"Step {i}"
                                        )
                                        step_action = (
                                            step_typed.get("action") or ""
                                        ).strip()
                                        if step_action:
                                            st.markdown(
                                                f"**{step_label}:** {step_action}"
                                            )
                                    else:
                                        step_text = str(step).strip()
                                        if step_text:
                                            st.markdown(f"**Turn {i}:** {step_text}")
                else:
                    st.markdown(message["content"])

                    if message.get("self_rag_sources") and message.get(
                        "self_rag_found_answer", True
                    ):
                        with st.expander(_VIEW_SOURCES_LABEL, expanded=False):
                            for j, source in enumerate(message["self_rag_sources"]):
                                st.markdown(
                                    f"""
                                        <div class='source-citation'>
                                        <strong>{j + 1}. {source["document"]} — Page {source["page"]}</strong><br>
                                        <small>{source["content"]}</small>
                                        </div>
                                        """,
                                    unsafe_allow_html=True,
                                )

                    # Check for historical trace
                    if message.get("self_rag_reasoning_trace"):
                        with st.expander("💭 Self-RAG Reasoning Trace", expanded=False):
                            st.markdown("**Multi-Hop Retrieval Execution:**")
                            for i, step in enumerate(
                                message["self_rag_reasoning_trace"], 1
                            ):
                                step_text = str(step).strip()
                                if step_text:
                                    st.markdown(f"**Step {i}:** {step_text}")

                            metrics = message.get("confidence_metrics")
                            if metrics:
                                st.markdown("---")
                                st.markdown("**Confidence Scores:**")
                                st.markdown(
                                    f"- **Total Score:** {metrics.get('total_score', 0.0):.2f}"
                                )
                                st.markdown(
                                    f"- Groundedness (ISSUP): {metrics.get('issup', 0.0):.2f}"
                                )
                                st.markdown(
                                    f"- Relevance (ISREL): {metrics.get('isrel', 0.0):.2f}"
                                )
                                st.markdown(
                                    f"- Utility (ISUSE): {metrics.get('isuse', 0.0):.2f}"
                                )

    # DETECT INTERRUPTED GENERATIONS: If the last message in history is from a User without an Assistant response
    needs_answer = False
    query_to_answer = ""
    # We slice out the last message if it's the pending query, but we also handle brand new inputs.
    if user_query:
        query_to_answer = user_query
        needs_answer = True
        # Save user message immediately
        db_middleware.add_chat_message(
            notebook_id=st.session_state.current_notebook_id,
            role=cfg.USER_ROLE_NAME,
            content=user_query,
        )
        st.session_state.chat_history.append(
            {"role": cfg.USER_ROLE_NAME, "content": user_query, "sources": None}
        )
        # Display user message immediately
        with chat_container:
            render_user_message(user_query)
    elif (
        len(st.session_state.chat_history) > 0
        and st.session_state.chat_history[-1]["role"] == cfg.USER_ROLE_NAME
    ):
        needs_answer = True
        query_to_answer = st.session_state.chat_history[-1]["content"]
        # It's already rendered via the loop above!

    if needs_answer:
        # Re-initialize the RAG chain if it was cleared (e.g. by applying new settings)
        if (
            st.session_state.rag_chain is None
            and st.session_state.vectorstore is not None
        ):
            st.session_state.rag_chain = create_history_aware_rag_chain(
                vectorstore=st.session_state.vectorstore,
                print_debug=print_debug,
                notebook_id=st.session_state.current_notebook_id,
            )

        # Generate answer
        if st.session_state.rag_chain is None:
            error_msg = "⚠️ Generation interrupted or RAG Chain not initialized. Please ensure documents are selected and ask your question again."
            db_middleware.add_chat_message(
                notebook_id=st.session_state.current_notebook_id,
                role=cfg.ASSISTANT_ROLE_NAME,
                content=error_msg,
                self_rag_content=error_msg,
                self_rag_sources=None,
                self_rag_found_answer=False,
            )
            st.session_state.chat_history.append(
                {
                    "role": cfg.ASSISTANT_ROLE_NAME,
                    "content": error_msg,
                    "self_rag_content": error_msg,
                    "self_rag_sources": None,
                    "self_rag_found_answer": False,
                }
            )
            st.rerun()
        else:
            from core.rag import run_dual_rag

            with chat_container:
                with st.spinner("🤔 Thinking..."):
                    try:
                        # Get answer and sources via RAG chain with chat history context
                        if print_debug:
                            print_breaker()
                            debug_log(
                                "INFO", "🔡", f"Processing query: {query_to_answer}"
                            )

                            # Log selected sources
                            debug_log("INFO", "📚", "Selected sources for this query:")
                            all_sources = db_middleware.get_sources_for_notebook(
                                st.session_state.current_notebook_id
                            )
                            selected_names = [
                                src.get("file_name", "Unknown")
                                for src in all_sources
                                if src["id"] in st.session_state.selected_sources
                            ]
                            if selected_names:
                                for name in selected_names:
                                    debug_log("INFO", "📄", f"• {name}")
                            else:
                                debug_log(
                                    "INFO",
                                    "📄",
                                    "No sources selected - using general knowledge",
                                )
                            print_breaker()

                        # Exclude the current query from the chat history passed for context!
                        history_context = st.session_state.chat_history[:-1]

                        # Build Self-RAG-isolated history (assistant turns use self_rag_content)
                        self_rag_history = [
                            {
                                "role": m["role"],
                                "content": (
                                    (m.get("self_rag_content") or m.get("content", ""))
                                    if m["role"] == cfg.ASSISTANT_ROLE_NAME
                                    else m.get("content", "")
                                ),
                            }
                            for m in history_context
                        ]

                        # Build Co-RAG-isolated history (assistant turns use co_rag_content)
                        co_rag_history = [
                            {
                                "role": m["role"],
                                "content": (
                                    (m.get("co_rag_content") or m.get("content", ""))
                                    if m["role"] == cfg.ASSISTANT_ROLE_NAME
                                    else m.get("content", "")
                                ),
                            }
                            for m in history_context
                        ]

                        result = run_dual_rag(
                            query_to_answer,
                            st.session_state.rag_chain,
                            st.session_state.vectorstore,
                            self_rag_history,
                            print_debug,
                            st.session_state.current_notebook_id,
                            co_rag_chat_history=co_rag_history,
                        )

                        answer = result["self_rag_content"]
                        sources = result["self_rag_sources"]
                        found_answer = result["self_rag_found_answer"]
                        reasoning_trace = result["self_rag_reasoning_trace"]
                        confidence_score = result["self_rag_confidence_score"]
                        co_rag_content = result["co_rag_content"]
                        co_rag_sources = result["co_rag_sources"]
                        co_rag_found_answer = result["co_rag_found_answer"]
                        co_rag_reasoning_trace = result["co_rag_reasoning_trace"]

                        # Store reasoning trace in session state for optional UI transparency display
                        st.session_state.last_reasoning_trace = reasoning_trace

                        # Display answer with dual-pipeline tabs
                        tab_srag, tab_corag = st.tabs(
                            ["⚡ Self-RAG (Vertical)", "⭐ Co-RAG (Horizontal)"]
                        )
                        with tab_srag:
                            st.markdown(answer)
                            if sources and found_answer:
                                with st.expander(_VIEW_SOURCES_LABEL, expanded=False):
                                    for j, source in enumerate(sources):
                                        st.markdown(
                                            f"""<div class='source-citation'><strong>{j + 1}. {source["document"]} — Page {source["page"]}</strong><br><small>{source["content"]}</small></div>""",
                                            unsafe_allow_html=True,
                                        )
                        with tab_corag:
                            if co_rag_content:
                                st.markdown(co_rag_content)
                                if co_rag_sources and co_rag_found_answer:
                                    with st.expander(
                                        _VIEW_SOURCES_LABEL, expanded=False
                                    ):
                                        for j, source in enumerate(co_rag_sources):
                                            st.markdown(
                                                f"""<div class='source-citation'><strong>{j + 1}. {source["document"]} — Page {source["page"]}</strong><br><small>{source["content"]}</small></div>""",
                                                unsafe_allow_html=True,
                                            )
                            else:
                                st.markdown(
                                    "*(Co-RAG not available for this message.)*"
                                )

                        # Add to history with self_rag_found_answer flag
                        db_middleware.add_chat_message(
                            notebook_id=st.session_state.current_notebook_id,
                            role=cfg.ASSISTANT_ROLE_NAME,
                            content=answer,
                            self_rag_content=answer,
                            self_rag_sources=sources,
                            self_rag_found_answer=found_answer,
                            self_rag_confidence_score=confidence_score,
                            self_rag_reasoning_trace=reasoning_trace,
                            co_rag_content=co_rag_content,
                            co_rag_sources=co_rag_sources,
                            co_rag_found_answer=co_rag_found_answer,
                            co_rag_reasoning_trace=co_rag_reasoning_trace,
                        )
                        st.session_state.chat_history.append(
                            {
                                "role": cfg.ASSISTANT_ROLE_NAME,
                                "content": answer,
                                "self_rag_content": answer,
                                "self_rag_sources": sources,
                                "self_rag_found_answer": found_answer,
                                "self_rag_reasoning_trace": st.session_state.last_reasoning_trace,
                                "confidence_metrics": st.session_state.self_rag_metadata.get(
                                    "confidence_metrics"
                                )
                                if "self_rag_metadata" in st.session_state
                                else None,
                                "co_rag_content": co_rag_content,
                                "co_rag_sources": co_rag_sources,
                                "co_rag_found_answer": co_rag_found_answer,
                                "co_rag_reasoning_trace": co_rag_reasoning_trace,
                            }
                        )

                        # We now render the trace entirely from the chat_history loop above
                        # Clear old transient session states if exist
                        if "last_reasoning_trace" in st.session_state:
                            st.session_state.last_reasoning_trace = []
                        if "self_rag_metadata" in st.session_state:
                            st.session_state.self_rag_metadata = {}

                    except Exception as e:
                        error_msg = str(e)
                        debug_log(
                            "ERROR",
                            message=f"Error during answer generation: {error_msg}",
                        )

                        # Check for CUDA/memory-related errors
                        if (
                            "cuda" in error_msg.lower()
                            or "out of memory" in error_msg.lower()
                        ):
                            display_error = (
                                "❌ GPU Memory Error: The model ran out of GPU memory. "
                                "Please try:\n"
                                "1. Asking a simpler question\n"
                                "2. Reducing the number of documents selected\n"
                                "3. Restarting the Ollama service\n"
                                "4. Using a smaller model\n\n"
                                f"Technical details: {error_msg}"
                            )
                        else:
                            display_error = f"❌ Error: {error_msg}"

                        db_middleware.add_chat_message(
                            notebook_id=st.session_state.current_notebook_id,
                            role=cfg.ASSISTANT_ROLE_NAME,
                            content=display_error,
                            self_rag_content=display_error,
                            self_rag_sources=None,
                            self_rag_found_answer=False,
                        )
                        st.session_state.chat_history.append(
                            {
                                "role": cfg.ASSISTANT_ROLE_NAME,
                                "content": display_error,
                                "self_rag_content": display_error,
                                "self_rag_sources": None,
                                "self_rag_found_answer": False,
                            }
                        )

        # Rerun to refresh chat display
        st.rerun()


# ============================================================================
# NOTES PANEL
# ============================================================================
@st.dialog("New Note")
def create_note_modal() -> None:
    """
    Modal dialog to create a new saved note.

    Prompts user for note title and content, validates input,
    and saves to the database.
    """
    title = st.text_input("Title", placeholder="Note title...")
    content = st.text_area("Content", placeholder="Write your note here...", height=200)
    col1, col2 = st.columns(2)
    with col1:
        cancel_click = st.button("Cancel", use_container_width=True)
    with col2:
        save_click = st.button("Save", type="primary", use_container_width=True)

    if cancel_click:
        st.rerun()
    if save_click:
        try:
            db_middleware.add_note(st.session_state.current_notebook_id, title, content)
            st.session_state.saved_notes = db_middleware.get_notes_for_notebook(
                st.session_state.current_notebook_id
            )
            st.rerun()
        except ValueError as e:
            st.error(str(e))


@st.dialog("Edit Note")
def edit_note_modal(note_id: str, current_title: str, current_content: str) -> None:
    """
    Modal dialog to edit an existing saved note.

    Args:
        note_id: UUID of the note to edit.
        current_title: Current note title (displayed as default).
        current_content: Current note content (displayed as default).
    """
    title = st.text_input("Title", value=current_title)
    content = st.text_area("Content", value=current_content or "", height=200)
    col1, col2 = st.columns(2)
    with col1:
        cancel_click = st.button(
            "Cancel", key="edit_note_cancel", use_container_width=True
        )
    with col2:
        save_click = st.button(
            "Save", key="edit_note_save", type="primary", use_container_width=True
        )

    if cancel_click:
        st.rerun()
    if save_click:
        try:
            db_middleware.update_note(note_id, title=title, content=content)
            st.session_state.saved_notes = db_middleware.get_notes_for_notebook(
                st.session_state.current_notebook_id
            )
            st.rerun()
        except ValueError as e:
            st.error(str(e))


def notes_panel_ui() -> None:
    """
    Render the Notes panel for saving and organizing study notes.

    Displays saved notes from the notebook, with functionality to create,
    edit, and delete notes. Notes are persisted in the database and can be
    exported or used to build study guides.
    """
    from core.utils import format_relative_time

    st.markdown(
        '<div class="notes-panel-bg" style="display:none"></div>',
        unsafe_allow_html=True,
    )
    col_title, col_count = st.columns([4, 1], vertical_alignment="center")
    with col_title:
        st.markdown("### Notes")
    with col_count:
        note_count = len(st.session_state.saved_notes)
        if note_count:
            st.caption(f"{note_count}")

    # Fixed-height scroll area — match the exact Chat column height so "Add Note" stays pinned nicely at the bottom
    with st.container(height=580, border=False):
        if st.session_state.saved_notes:
            for note in st.session_state.saved_notes:
                with st.expander(f"📝 {note['title']}", expanded=False):
                    st.caption(format_relative_time(note.get("created_at", "")))
                    st.markdown(note.get("content") or "*No content*")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(
                            "Edit",
                            key=f"edit_note_{note['id']}",
                            use_container_width=True,
                        ):
                            edit_note_modal(
                                note["id"],
                                note["title"],
                                note.get("content") or "",
                            )
                    with col2:
                        if st.button(
                            "Delete",
                            key=f"delete_note_{note['id']}",
                            type="secondary",
                            use_container_width=True,
                        ):
                            confirm_delete_note_dialog(
                                note["id"], st.session_state.current_notebook_id
                            )
        else:
            st.info("🛸 No notes yet.")

    st.markdown('<div class="add-note-btn-anchor"></div>', unsafe_allow_html=True)
    if st.button("+ Add Note", type="primary", use_container_width=True):
        create_note_modal()


def delete_notebook_callback(nb_id: str) -> None:
    """
    Callback to delete a notebook and its associated data.

    Removes the notebook record from the database and cleans up the
    associated vectorstore directory from disk.

    Args:
        nb_id: UUID of the notebook to delete.
    """
    from core.utils import get_notebook_vectorstore_dir

    if db_middleware.delete_notebook(nb_id):
        # Clean up vectorstore if exists
        vs_dir = get_notebook_vectorstore_dir(nb_id)
        if vs_dir.exists():
            import shutil

            shutil.rmtree(vs_dir, ignore_errors=True)


@st.dialog("Create New Notebook")
def create_notebook_modal() -> None:
    """
    Modal dialog to create a new notebook.

    Prompts user for notebook name and optional description, validates input,
    and creates the notebook in the database.
    """
    new_name = st.text_input(
        "Notebook Name", placeholder="e.g., Biology 101, Project Phoenix..."
    )
    new_desc = st.text_input(
        "Description (optional)", placeholder="Brief description..."
    )
    if st.button("Create Notebook", type="primary"):
        if not new_name.strip():
            st.error("Notebook name cannot be empty.")
        else:
            try:
                new_id = db_middleware.create_notebook(new_name, new_desc)
                st.session_state.loading_notebook_id = new_id
                st.rerun()
            except ValueError as e:
                st.error(f"❌ {str(e)}")


def _compute_settings_warnings(
    snap: Dict[str, Any],
    hw_info: Dict[str, Any],
) -> Tuple[List[Tuple[str, str]], bool]:
    """
    Compute all validation messages for the notebook settings UI.

    Evaluates hardware compatibility, logical consistency, performance impact,
    and RAG quality trade-offs across both Self-RAG and Co-RAG pipelines.

    Args:
        snap: Flat dict containing all current setting values:
              model, temp, num_ctx, personal_ctx, rerank_n, k, threshold,
              history, chunk_len, chunk_ovl, min_res, max_ctx,
              self_rag_max_depth, self_rag_candidates, self_rag_max_retries,
              self_rag_threshold_issup, self_rag_threshold_isrel,
              self_rag_threshold_isuse, co_rag_max_retries.
        hw_info: Hardware info dict from get_system_hardware_info().

    Returns:
        Tuple of:
            issues: List of (level, message) where level is "error" | "warning" | "info".
            is_forbidden: True when at least one issue is a logical impossibility
                          that must be resolved before settings can be applied.
    """
    issues: List[Tuple[str, str]] = []
    is_forbidden = False

    # Unpack settings snapshot with explicit casts for type safety
    model: str = str(snap.get("model", ""))
    temp: float = float(snap.get("temp", 0.7))
    num_ctx: int = int(snap.get("num_ctx", 4096))
    personal_ctx: str = str(snap.get("personal_ctx", ""))
    rerank_n: int = int(snap.get("rerank_n", 10))
    k: int = int(snap.get("k", 8))
    threshold: float = float(snap.get("threshold", 15.0))
    history: int = int(snap.get("history", 10))
    chunk_len: int = int(snap.get("chunk_len", 1000))
    chunk_ovl: int = int(snap.get("chunk_ovl", 100))
    min_res: int = int(snap.get("min_res", 1))
    max_ctx: int = int(snap.get("max_ctx", 8000))
    self_rag_max_depth: int = int(snap.get("self_rag_max_depth", 2))
    self_rag_candidates: int = int(snap.get("self_rag_candidates", 3))
    self_rag_max_retries: int = int(snap.get("self_rag_max_retries", 2))
    self_rag_threshold_issup: float = float(snap.get("self_rag_threshold_issup", 0.7))
    self_rag_threshold_isrel: float = float(snap.get("self_rag_threshold_isrel", 0.7))
    self_rag_threshold_isuse: float = float(snap.get("self_rag_threshold_isuse", 0.7))
    co_rag_max_retries: int = int(snap.get("co_rag_max_retries", 3))

    vram: float = float(hw_info.get("vram_gb") or 0.0)
    ram: float = float(hw_info.get("ram_gb") or 0.0)
    model_lower = model.lower()

    # ------------------------------------------------------------------ #
    # A. Hardware & Memory Capacity
    # ------------------------------------------------------------------ #

    # VRAM guardrails by model family size
    is_heavy_model = any(tag in model_lower for tag in ("70b", "72b", "32b"))
    is_medium_model = not is_heavy_model and any(
        tag in model_lower for tag in ("14b", "13b", "20b")
    )
    is_light_model = not is_heavy_model and not is_medium_model

    if is_heavy_model and vram < 24.0:
        issues.append(
            (
                "error",
                f"⚠️ Hardware Mismatch: Your GPU has {vram:.1f} GB VRAM. Models 32B+ typically "
                "require >24 GB. Expect severe lag or Out-Of-Memory crashes.",
            )
        )
    elif is_medium_model and vram < 12.0:
        issues.append(
            (
                "warning",
                f"⚠️ Memory Pressure: Your GPU has {vram:.1f} GB VRAM. Models 14B–20B typically "
                "require >12 GB. You may experience significant slowdowns or OOM errors.",
            )
        )
    elif is_light_model and 0.0 < vram < 6.0:
        issues.append(
            (
                "warning",
                f"⚠️ Low VRAM: Your GPU has {vram:.1f} GB VRAM. Models up to 8B benefit from "
                ">6 GB VRAM. Consider a smaller quantized model for faster inference.",
            )
        )

    # High context window on low-RAM hardware
    if num_ctx > 8192 and (ram < 16.0 or vram < 8.0):
        issues.append(
            (
                "warning",
                f"⚠️ Memory Warning: A context window of {num_ctx:,} tokens consumes significant "
                "RAM/VRAM. You may experience crashes on your current hardware.",
            )
        )

    # Context bottleneck: estimated text usage vs context window size
    # Usage = (chunks × chars/chunk) + (history turns × ~250 chars/turn) + system overhead
    estimated_usage = (k * chunk_len) + (history * 250) + 500
    if estimated_usage > num_ctx:
        issues.append(
            (
                "warning",
                "⚠️ Context Bottleneck: Your Retrieval K and Chat History generate more text than "
                "your Context Window can hold. The AI will 'forget' older document chunks or chat "
                f"history. Estimated: ~{estimated_usage:,} chars vs window: {num_ctx:,} tokens.",
            )
        )

    # ------------------------------------------------------------------ #
    # B. Logical Consistency — Forbidden States (block Apply)
    # ------------------------------------------------------------------ #

    # Retrieval funnel: cannot select more than retrieved
    if k > rerank_n:
        issues.append(
            (
                "error",
                f"❌ Logic Error: Final Context K ({k}) cannot exceed the Initial Retrieval Pool "
                f"({rerank_n}). The system cannot select more chunks than it initially retrieved "
                "for re-ranking.",
            )
        )
        is_forbidden = True

    # Chunking infinity loop
    if chunk_ovl >= chunk_len:
        issues.append(
            (
                "error",
                f"❌ Chunking Error: Chunk Overlap ({chunk_ovl}) must be strictly less than "
                f"Chunk Length ({chunk_len}). This setting would cause an infinite document-"
                "processing loop.",
            )
        )
        is_forbidden = True

    # Fallback conflict
    if min_res > k:
        issues.append(
            (
                "error",
                f"❌ Fallback Conflict: Minimum required results ({min_res}) cannot be higher than "
                f"Final Context K ({k}). The fallback cannot guarantee more results than K allows.",
            )
        )
        is_forbidden = True

    # Self-RAG threshold floor — any gate at zero disables the scoring logic
    # Use < 0.01 instead of == 0.0 to avoid float equality check (slider step is 0.05)
    gates_at_zero = [
        name
        for name, val in (
            ("ISSUP", self_rag_threshold_issup),
            ("ISREL", self_rag_threshold_isrel),
            ("ISUSE", self_rag_threshold_isuse),
        )
        if val < 0.01
    ]
    if gates_at_zero:
        gate_list = ", ".join(gates_at_zero)
        issues.append(
            (
                "error",
                f"❌ Invalid Threshold: Setting {gate_list} to 0.0 disables the Self-RAG repair "
                "logic for that gate entirely. Use at least 0.1 to keep quality gates active.",
            )
        )
        is_forbidden = True

    # ------------------------------------------------------------------ #
    # C. Retrieval & RAG Quality Warnings
    # ------------------------------------------------------------------ #

    if threshold < 10.0:
        issues.append(
            (
                "warning",
                f"⚠️ High Strictness: A Score Threshold of {threshold:.1f} is very strict. "
                "The AI may frequently say 'I cannot find the answer' even when related text "
                "exists in your documents.",
            )
        )
    elif threshold > 15.0:
        issues.append(
            (
                "warning",
                f"⚠️ Low Strictness: A Score Threshold of {threshold:.1f} is very loose. "
                "The AI may retrieve irrelevant noise and produce hallucinated answers.",
            )
        )

    # High rerank pool on CPU-only hardware
    if rerank_n > 40 and vram < 0.1:
        issues.append(
            (
                "warning",
                "⚠️ High Retrieval Pool Without GPU: Cross-Encoder Re-ranking over a large "
                "candidate pool will heavily impact Time to First Token on CPU-only hardware. "
                "Decrease Top-N for faster responses.",
            )
        )

    # Chunk length overflows RAG max context
    if chunk_len > max_ctx:
        issues.append(
            (
                "warning",
                f"⚠️ Embedding Truncation: Max Chunk Length ({chunk_len:,}) exceeds RAG Max "
                f"Context Length ({max_ctx:,}). The tail of each document chunk will be "
                "truncated and lost during embedding.",
            )
        )

    # ------------------------------------------------------------------ #
    # D. Self-RAG Performance & Quality
    # ------------------------------------------------------------------ #

    # High Complexity: depth × candidates = total LLM calls per query
    llm_calls = self_rag_max_depth * self_rag_candidates
    if llm_calls > 6:
        issues.append(
            (
                "warning",
                f"⚠️ Complex Search: Search depth ({self_rag_max_depth}) × candidates "
                f"({self_rag_candidates}) = {llm_calls} LLM calls per query. "
                "Expect significant latency.",
            )
        )

    # Perfectionist Trap: deep search + extremely strict gates → likely never passes
    if self_rag_max_depth > 3 and (
        self_rag_threshold_issup > 0.8 or self_rag_threshold_isrel > 0.8
    ):
        issues.append(
            (
                "warning",
                "⚠️ Perfectionist Trap: High search depth combined with extremely strict quality "
                "gates (ISSUP or ISREL > 0.8) may lead to frequent 'Answer Not Found' results "
                "or excessive repair loops.",
            )
        )

    # Redundant Retries: many retries with very lenient gates → first result always accepted
    if (
        self_rag_max_retries > 2
        and self_rag_threshold_issup < 0.4
        and self_rag_threshold_isrel < 0.4
    ):
        issues.append(
            (
                "info",
                "💡 Redundant Retries: High retries per hop combined with very low quality "
                "thresholds — the system will likely accept the first low-quality result, "
                "making extra retries a waste of compute.",
            )
        )

    # High branching: each extra candidate adds a full LLM generation
    if self_rag_candidates > 3:
        issues.append(
            (
                "warning",
                f"⚡ High Branching: Generating {self_rag_candidates} candidate answers per hop "
                "will spike CPU/GPU usage during the Self-RAG candidate generation phase.",
            )
        )

    # Loose Evidence gate: low ISSUP means weak grounding is accepted
    if 0.01 < self_rag_threshold_issup < 0.5:
        issues.append(
            (
                "info",
                f"📊 Loose Evidence Gate: ISSUP threshold of {self_rag_threshold_issup:.2f} allows "
                "answers with weak document support. Recommended ≥ 0.7 for research-grade accuracy.",
            )
        )

    # ------------------------------------------------------------------ #
    # E. Dual-Pipeline Parallel Fatigue
    # ------------------------------------------------------------------ #

    if self_rag_max_depth > 3 and co_rag_max_retries > 3:
        issues.append(
            (
                "warning",
                f"🐢 Dual-Pipeline Latency: Both Recursive Search (Self-RAG depth="
                f"{self_rag_max_depth}) and Peer-Review (Co-RAG turns={co_rag_max_retries}) "
                "are set to high-intensity mode. Total response time will be exceptionally slow.",
            )
        )

    # ------------------------------------------------------------------ #
    # F. LLM Behavior
    # ------------------------------------------------------------------ #

    if temp > 0.8:
        issues.append(
            (
                "info",
                f"💡 High Creativity: Temperature {temp:.2f} increases linguistic variety but may "
                "cause the Co-RAG Reviewer to miss factual inconsistencies and produce less "
                "grounded answers.",
            )
        )
    elif temp > 0.7:
        issues.append(
            (
                "info",
                f"💡 Elevated Temperature: Temperature {temp:.2f} makes the AI more creative but "
                "slightly increases the risk of hallucinating facts outside the document context.",
            )
        )

    if len(personal_ctx) > 1000:
        issues.append(
            (
                "info",
                "💡 Long Personal Context: Your persona instructions are quite long. They are safely "
                "injected into the system prompt but consume part of your available Context Window.",
            )
        )

    # ------------------------------------------------------------------ #
    # G. Document Chunking Quality
    # ------------------------------------------------------------------ #

    # High overlap redundancy (only when not already a forbidden-state overlap)
    if chunk_len > 0 and chunk_ovl < chunk_len and chunk_ovl > chunk_len * 0.5:
        issues.append(
            (
                "info",
                f"💡 High Overlap Redundancy: Chunk Overlap ({chunk_ovl}) exceeds 50% of Chunk "
                f"Length ({chunk_len}). This wastes memory by embedding the same text segments "
                "multiple times during indexing.",
            )
        )

    return issues, is_forbidden


def render_notebook_settings_sidebar(notebook_id: str) -> None:
    """
    Render the notebook settings sidebar for RAG parameter tuning.

    Displays configurable settings for:
    - LLM parameters (model, temperature, context window)
    - RAG parameters (retrieval K, score threshold, chunk size)
    - Personal context / system instructions
    - Retrieval strategy balance (semantic vs keyword)

    Includes validation warnings for configuration conflicts and unsafe settings.
    Settings are persisted to the database and applied to the current notebook.

    Args:
        notebook_id: UUID of the notebook to configure.
    """
    from core.utils import get_installed_ollama_models, get_system_hardware_info

    hw_info = get_system_hardware_info()
    settings = (
        db_middleware.get_notebook_settings(notebook_id)
        or get_default_notebook_settings()
    )
    models: List[str] = get_installed_ollama_models()

    # Defaults in case the models list is empty or the selected model is not in the list
    if settings["llm_model_name"] not in models:
        if settings["llm_model_name"]:
            models.append(settings["llm_model_name"])
        else:
            models.append(cfg.LLM_MODEL_NAME)

    if not models:
        models = [cfg.LLM_MODEL_NAME]

    # Use a revision key for all inputs to force complete UI reset when clicking Reset to Default
    rev_key = f"settings_rev_{notebook_id}"
    if rev_key not in st.session_state:
        st.session_state[rev_key] = 0
    k_suf = f"_{notebook_id}_{st.session_state[rev_key]}"

    # Handle cross-reload toasts
    if "settings_toast" in st.session_state and st.session_state.settings_toast:
        st.toast(st.session_state.settings_toast)
        st.session_state.settings_toast = None

    st.sidebar.markdown("## ⚙️ Notebook Settings")
    st.sidebar.markdown(
        'Configure parameters for this notebook. Changes are saved only when you click "Apply Settings".'
    )

    with st.sidebar.container():
        st.markdown("### LLM Configuration")
        new_model = st.selectbox(
            "Model Name",
            key=f"model{k_suf}",
            options=models,
            index=(
                models.index(settings["llm_model_name"])
                if settings["llm_model_name"] in models
                else 0
            ),
            placeholder=cfg.LLM_MODEL_NAME,
            help=cfg.LLM_MODEL_NAME_HELP_MSG,
        )

        new_temp = st.slider(
            "Temperature",
            key=f"temp{k_suf}",
            min_value=cfg.LLM_AVG_TEMP_MIN,
            max_value=cfg.LLM_AVG_TEMP_MAX,
            step=cfg.LLM_AVG_TEMP_STEP,
            value=float(settings["llm_avg_temp"]),
            help=cfg.LLM_AVG_TEMP_HELP_MSG,
        )

        new_num_ctx = st.number_input(
            "Context Window (num_ctx)",
            key=f"ctx{k_suf}",
            min_value=cfg.LLM_NUM_CTX_MIN,
            max_value=cfg.LLM_NUM_CTX_MAX,
            step=cfg.LLM_NUM_CTX_STEP,
            value=int(settings["llm_num_ctx"]),
            placeholder=str(cfg.LLM_NUM_CTX),
            help=cfg.LLM_NUM_CTX_HELP_MSG,
        )

        st.markdown("### Personal Context (Optional)")
        new_personal_ctx = st.text_area(
            "Personal Background & Instructions",
            key=f"pctx{k_suf}",
            value=settings.get("personal_ctx", "") or "",
            height=150,
            placeholder=cfg.PERSONAL_CTX_PLACEHOLDER,
            help=cfg.PERSONAL_CTX_HELP_MSG,
        )

        st.markdown("### RAG & Retrieval")
        new_weight_semantic = st.slider(
            "Retrieval Strategy Balance",
            key=f"weights{k_suf}",
            min_value=cfg.WEIGHT_SEMANTIC_MIN,
            max_value=cfg.WEIGHT_SEMANTIC_MAX,
            step=cfg.WEIGHT_SEMANTIC_STEP,
            value=float(settings.get("weight_semantic", cfg.WEIGHT_SEMANTIC)),
            help=cfg.WEIGHT_SEMANTIC_HELP_MSG,
        )

        new_rerank_n = st.number_input(
            "Initial Retrieval Pool (Top-N)",
            key=f"rerank{k_suf}",
            min_value=cfg.RAG_RERANK_TOP_N_MIN,
            max_value=cfg.RAG_RERANK_TOP_N_MAX,
            step=cfg.RAG_RERANK_TOP_N_STEP,
            value=int(settings.get("rag_rerank_top_n", cfg.RAG_RERANK_TOP_N)),
            placeholder=str(cfg.RAG_RERANK_TOP_N),
            help=cfg.RAG_RERANK_TOP_N_HELP_MSG,
        )

        new_k = st.number_input(
            "Final LLM Context (Top-K)",
            key=f"k{k_suf}",
            min_value=cfg.RAG_FINAL_CONTEXT_K_MIN,
            max_value=cfg.RAG_FINAL_CONTEXT_K_MAX,
            step=cfg.RAG_FINAL_CONTEXT_K_STEP,
            value=int(settings["rag_final_context_k"]),
            placeholder=str(cfg.RAG_FINAL_CONTEXT_K),
            help=cfg.RAG_FINAL_CONTEXT_K_HELP_MSG,
        )

        new_threshold = st.slider(
            "Score Threshold",
            key=f"thresh{k_suf}",
            min_value=cfg.RAG_RETRIEVAL_SCORE_THRESHOLD_MIN,
            max_value=cfg.RAG_RETRIEVAL_SCORE_THRESHOLD_MAX,
            step=cfg.RAG_RETRIEVAL_SCORE_THRESHOLD_STEP,
            value=float(settings["rag_retrieval_score_threshold"]),
            help=cfg.RAG_RETRIEVAL_SCORE_THRESHOLD_HELP_MSG,
        )

        new_history = st.number_input(
            "Max Chat History",
            key=f"hist{k_suf}",
            min_value=cfg.MAX_MSG_HISTORY_MIN,
            max_value=cfg.MAX_MSG_HISTORY_MAX,
            step=cfg.MAX_MSG_HISTORY_STEP,
            value=int(settings["max_msg_history"]),
            placeholder=str(cfg.MAX_MSG_HISTORY),
            help=cfg.MAX_MSG_HISTORY_HELP_MSG,
        )

        st.markdown("### Advanced Document Settings")
        new_chunk_len = st.number_input(
            "Max Chunk Length",
            key=f"clen{k_suf}",
            min_value=cfg.RAG_MAX_CHUNK_LEN_MIN,
            max_value=cfg.RAG_MAX_CHUNK_LEN_MAX,
            step=cfg.RAG_MAX_CHUNK_LEN_STEP,
            value=int(settings["rag_max_chunk_len"]),
            placeholder=str(cfg.RAG_MAX_CHUNK_LEN),
            help=cfg.RAG_MAX_CHUNK_LEN_HELP_MSG,
        )
        new_chunk_ovl = st.number_input(
            "Chunk Overlap",
            key=f"covl{k_suf}",
            min_value=cfg.RAG_CHUNK_OVERLAP_MIN,
            max_value=cfg.RAG_CHUNK_OVERLAP_MAX,
            step=cfg.RAG_CHUNK_OVERLAP_STEP,
            value=int(settings["rag_chunk_overlap"]),
            placeholder=str(cfg.RAG_CHUNK_OVERLAP),
            help=cfg.RAG_CHUNK_OVERLAP_HELP_MSG,
        )
        new_min_res = st.number_input(
            "Min Match Results (Fallback)",
            key=f"mres{k_suf}",
            min_value=cfg.RAG_RETRIEVAL_MIN_RESULTS_MIN,
            max_value=cfg.RAG_RETRIEVAL_MIN_RESULTS_MAX,
            step=cfg.RAG_RETRIEVAL_MIN_RESULTS_STEP,
            value=int(settings["rag_retrieval_min_results"]),
            placeholder=str(cfg.RAG_RETRIEVAL_MIN_RESULTS),
            help=cfg.RAG_RETRIEVAL_MIN_RESULTS_HELP_MSG,
        )
        new_max_ctx = st.number_input(
            "RAG Max Context Length",
            key=f"mctx{k_suf}",
            min_value=cfg.RAG_MAX_CTX_LEN_MIN,
            max_value=cfg.RAG_MAX_CTX_LEN_MAX,
            step=cfg.RAG_MAX_CTX_LEN_STEP,
            value=int(settings["rag_max_ctx_len"]),
            placeholder=str(cfg.RAG_MAX_CTX_LEN),
            help=cfg.RAG_MAX_CTX_LEN_HELP_MSG,
        )

        st.markdown("### Self-RAG Configuration")
        st.markdown("**Multi-hop retrieval with quality-based repair**")

        new_self_rag_max_depth = st.number_input(
            "Max Retrieval Depth",
            key=f"srag_depth{k_suf}",
            min_value=cfg.SELF_RAG_MAX_DEPTH_MIN,
            max_value=cfg.SELF_RAG_MAX_DEPTH_MAX,
            step=cfg.SELF_RAG_MAX_DEPTH_STEP,
            value=int(settings.get("self_rag_max_depth", cfg.SELF_RAG_MAX_DEPTH)),
            help=cfg.SELF_RAG_MAX_DEPTH_HELP_MSG,
        )

        new_self_rag_candidates = st.number_input(
            "Candidate Answers per Hop",
            key=f"srag_cand{k_suf}",
            min_value=cfg.SELF_RAG_CANDIDATES_MIN,
            max_value=cfg.SELF_RAG_CANDIDATES_MAX,
            step=cfg.SELF_RAG_CANDIDATES_STEP,
            value=int(settings.get("self_rag_candidates", cfg.SELF_RAG_CANDIDATES)),
            help=cfg.SELF_RAG_CANDIDATES_HELP_MSG,
        )

        new_self_rag_max_retries = st.number_input(
            "Retries per Query (Surgical Retry)",
            key=f"srag_retry{k_suf}",
            min_value=cfg.SELF_RAG_MAX_RETRIES_PER_HOP_MIN,
            max_value=cfg.SELF_RAG_MAX_RETRIES_PER_HOP_MAX,
            step=cfg.SELF_RAG_MAX_RETRIES_PER_HOP_STEP,
            value=int(
                settings.get(
                    "self_rag_max_retries_per_hop", cfg.SELF_RAG_MAX_RETRIES_PER_HOP
                )
            ),
            help=cfg.SELF_RAG_MAX_RETRIES_PER_HOP_HELP_MSG,
        )

        st.markdown("**Quality Gates (0.0-1.0)**")
        new_self_rag_threshold_issup = st.slider(
            "Groundedness (ISSUP)",
            key=f"srag_issup{k_suf}",
            min_value=cfg.SELF_RAG_THRESHOLD_ISSUP_MIN,
            max_value=cfg.SELF_RAG_THRESHOLD_ISSUP_MAX,
            step=cfg.SELF_RAG_THRESHOLD_ISSUP_STEP,
            value=float(settings.get("self_rag_threshold_issup", 0.70)),
            help=cfg.SELF_RAG_THRESHOLD_ISSUP_HELP_MSG,
        )

        new_self_rag_threshold_isrel = st.slider(
            "Relevance (ISREL)",
            key=f"srag_isrel{k_suf}",
            min_value=cfg.SELF_RAG_THRESHOLD_ISREL_MIN,
            max_value=cfg.SELF_RAG_THRESHOLD_ISREL_MAX,
            step=cfg.SELF_RAG_THRESHOLD_ISREL_STEP,
            value=float(settings.get("self_rag_threshold_isrel", 0.70)),
            help=cfg.SELF_RAG_THRESHOLD_ISREL_HELP_MSG,
        )

        new_self_rag_threshold_isuse = st.slider(
            "Utility (ISUSE)",
            key=f"srag_isuse{k_suf}",
            min_value=cfg.SELF_RAG_THRESHOLD_ISUSE_MIN,
            max_value=cfg.SELF_RAG_THRESHOLD_ISUSE_MAX,
            step=cfg.SELF_RAG_THRESHOLD_ISUSE_STEP,
            value=float(settings.get("self_rag_threshold_isuse", 0.70)),
            help=cfg.SELF_RAG_THRESHOLD_ISUSE_HELP_MSG,
        )

        st.markdown("### Co-RAG Configuration")
        new_co_rag_max_retries = st.number_input(
            "Max Collaboration Turns",
            key=f"corag_retry{k_suf}",
            min_value=cfg.CO_RAG_MAX_RETRIES_MIN,
            max_value=cfg.CO_RAG_MAX_RETRIES_MAX,
            step=cfg.CO_RAG_MAX_RETRIES_STEP,
            value=int(settings.get("co_rag_max_retries", cfg.CO_RAG_MAX_RETRIES)),
            help=cfg.CO_RAG_MAX_RETRIES_HELP_MSG,
        )

        # --- Settings Validation ---
        issues, is_forbidden = _compute_settings_warnings(
            snap={
                "model": str(new_model),
                "temp": float(new_temp),
                "num_ctx": int(new_num_ctx),
                "personal_ctx": str(new_personal_ctx) if new_personal_ctx else "",
                "rerank_n": int(new_rerank_n),
                "k": int(new_k),
                "threshold": float(new_threshold),
                "history": int(new_history),
                "chunk_len": int(new_chunk_len),
                "chunk_ovl": int(new_chunk_ovl),
                "min_res": int(new_min_res),
                "max_ctx": int(new_max_ctx),
                "self_rag_max_depth": int(new_self_rag_max_depth),
                "self_rag_candidates": int(new_self_rag_candidates),
                "self_rag_max_retries": int(new_self_rag_max_retries),
                "self_rag_threshold_issup": float(new_self_rag_threshold_issup),
                "self_rag_threshold_isrel": float(new_self_rag_threshold_isrel),
                "self_rag_threshold_isuse": float(new_self_rag_threshold_isuse),
                "co_rag_max_retries": int(new_co_rag_max_retries),
            },
            hw_info=hw_info,
        )
        for level, msg in issues:
            if level == "error":
                st.error(msg)
            elif level == "warning":
                st.warning(msg)
            else:
                st.info(msg)
        if not issues:
            st.success(
                "🤗 System configurations are mathematically sound and optimal for your current hardware."
            )

        col_reset, col_apply = st.columns(2)
        with col_reset:
            reset_clicked = st.button("Reset to Default", use_container_width=True)
        with col_apply:
            apply_clicked = st.button(
                "Apply Settings", type="primary", use_container_width=True
            )

    if reset_clicked:
        with st.spinner("Resetting to defaults..."):
            is_deleted = db_middleware.delete_notebook_settings(notebook_id)
        if is_deleted:
            st.session_state.rag_chain = None

        st.session_state[rev_key] += 1
        st.session_state.settings_toast = "🤗 Settings reset to defaults"
        st.rerun()

    if apply_clicked:
        if is_forbidden:
            st.session_state.settings_toast = "❌ Cannot apply settings due to configuration conflicts. Please adjust the parameters and try again."
            return

        updated_settings: Dict[str, Any] = {
            "llm_model_name": str(new_model),
            "llm_avg_temp": float(new_temp),
            "llm_num_ctx": int(new_num_ctx),
            "rag_final_context_k": int(new_k),
            "rag_rerank_top_n": int(new_rerank_n),
            "rag_retrieval_score_threshold": float(new_threshold),
            "max_msg_history": int(new_history),
            "rag_max_chunk_len": int(new_chunk_len),
            "rag_chunk_overlap": int(new_chunk_ovl),
            "rag_retrieval_min_results": int(new_min_res),
            "rag_max_ctx_len": int(new_max_ctx),
            "personal_ctx": (
                str(new_personal_ctx).strip() if new_personal_ctx else None
            ),
            "weight_semantic": float(new_weight_semantic),
            "weight_bm25": 1.0 - float(new_weight_semantic),
            "self_rag_max_depth": int(new_self_rag_max_depth),
            "self_rag_candidates": int(new_self_rag_candidates),
            "self_rag_max_retries_per_hop": int(new_self_rag_max_retries),
            "self_rag_threshold_issup": float(new_self_rag_threshold_issup),
            "self_rag_threshold_isrel": float(new_self_rag_threshold_isrel),
            "self_rag_threshold_isuse": float(new_self_rag_threshold_isuse),
            "co_rag_max_retries": int(new_co_rag_max_retries),
        }

        # Check if settings actually changed
        for k, v in updated_settings.items():
            if k not in settings or settings[k] != v:
                with st.spinner("Applying settings..."):
                    db_middleware.upsert_notebook_settings(
                        notebook_id, updated_settings
                    )
                # Invalidate the cached settings so the next query loads fresh values
                load_notebook_settings.clear()
                st.session_state.rag_chain = None
                break

        st.session_state.settings_toast = "🤗 Settings saved successfully"
        st.rerun()


@st.dialog("Rename Notebook")
def rename_notebook_modal(
    notebook_id: str, current_name: str, current_description: Optional[str] = None
) -> None:
    """
    Modal dialog to rename a notebook and update its description.

    Args:
        notebook_id: UUID of the notebook to rename.
        current_name: Current notebook name (displayed as default).
        current_description: Current description (if any).
    """
    st.markdown("Edit notebook details:")

    new_name = st.text_input(
        "Notebook Name",
        value=current_name,
        placeholder="e.g., Biology 101, Project Phoenix...",
    )

    new_description = st.text_input(
        "Description (optional)",
        value=current_description or "",
        placeholder="Brief description...",
    )

    col1, col2 = st.columns(2)
    with col1:
        cancel_click = st.button("Cancel", use_container_width=True)
    with col2:
        save_click = st.button("Save", type="primary", use_container_width=True)

    if cancel_click:
        st.rerun()
    if save_click:
        try:
            # Pass through middleware for validation
            db_middleware.rename_notebook(
                notebook_id,
                new_name if new_name != current_name else None,
                (
                    new_description
                    if new_description != (current_description or "")
                    else None
                ),
            )
            st.success("Notebook renamed successfully!")
            st.session_state.notebooks = (
                db_middleware.get_all_notebooks()
            )  # Reload notebook list
            st.rerun()
        except ValueError as e:
            st.error(f"❌ {str(e)}")
        except Exception as e:
            debug_log("ERROR", message=f"Error renaming notebook: {str(e)}")
            st.error(f"Error renaming notebook: {str(e)}")


def rename_source_modal(source_id: str, current_name: str) -> None:
    """
    Modal dialog to rename a source document.

    Args:
        source_id: UUID of the source to rename.
        current_name: Current source filename.
    """
    st.markdown("Rename source:")

    new_name = st.text_input(
        "Filename",
        value=current_name,
        placeholder="e.g., Biology Textbook, Research Paper...",
    )

    col1, col2 = st.columns(2)
    with col1:
        cancel_click = st.button("Cancel", use_container_width=True)
    with col2:
        save_click = st.button("Save", type="primary", use_container_width=True)

    if cancel_click:
        st.session_state.rename_source_modal_open = False
        st.rerun()
    if save_click:
        try:
            if new_name == current_name:
                st.warning("No changes made.")
            else:
                # Pass through middleware for validation
                db_middleware.rename_source(source_id, new_name)
                st.success("Source renamed successfully!")

                # Update session state documents to reflect change (keyed by source_id)
                if source_id in st.session_state.documents:
                    st.session_state.documents[source_id]["file_name"] = new_name

                st.session_state.rename_source_modal_open = False
                st.rerun()
        except ValueError as e:
            st.error(f"❌ {str(e)}")
        except Exception as e:
            debug_log("ERROR", message=f"Error renaming source: {str(e)}")
            st.error(f"Error renaming source: {str(e)}")


@st.dialog("Rename Source", width="small")
def show_rename_source_dialog() -> None:
    """
    Display the rename source modal dialog.

    Wrapper that triggers the rename_source_modal() with session state data.
    """
    if st.session_state.rename_source_id:
        rename_source_modal(
            st.session_state.rename_source_id, st.session_state.rename_source_name or ""
        )


# ============================================================================
# DELETION CONFIRMATION DIALOGS
# ============================================================================


@st.dialog("Confirm Deletion", width="small")
def confirm_delete_notebook_dialog(nb_id: str, nb_name: str) -> None:
    """
    Confirmation dialog for notebook deletion.

    Args:
        nb_id: UUID of the notebook to delete.
        nb_name: Display name of the notebook (shown in warning message).
    """
    st.warning(
        f"Are you sure you want to delete the notebook **{nb_name}**? This action cannot be undone."
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Cancel", key=f"cancel_del_nb_{nb_id}", use_container_width=True):
            st.rerun()
    with col2:
        if st.button(
            "Confirm",
            key=f"confirm_del_nb_{nb_id}",
            type="primary",
            use_container_width=True,
        ):
            delete_notebook_callback(nb_id)
            st.rerun()


@st.dialog("Confirm Deletion", width="small")
def confirm_delete_source_dialog(
    source_id: str, file_name: str, notebook_id: str, print_debug: bool
) -> None:
    """
    Confirmation dialog for source document deletion.

    Removes source from database, deletes vectorstore files, and reloads RAG chain.

    Args:
        source_id: UUID of the source to delete.
        file_name: Display name of the document.
        notebook_id: UUID of the parent notebook (for vectorstore cleanup).
        print_debug: If True, emit debug logs during deletion.
    """
    st.warning(
        f"Are you sure you want to delete **{file_name}**? This action cannot be undone."
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "Cancel", key=f"cancel_del_src_{source_id}", use_container_width=True
        ):
            st.rerun()
    with col2:
        if st.button(
            "Confirm",
            key=f"confirm_del_src_{source_id}",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner(f"Removing {file_name}..."):
                if print_debug:
                    debug_log(
                        "PROCESS_START", message=f"Deleting document: {file_name}"
                    )

                if db_middleware.delete_source(source_id):
                    source_dir = get_source_vectorstore_dir(notebook_id, source_id)
                    if source_dir.exists():
                        import shutil

                        shutil.rmtree(source_dir, ignore_errors=True)
                    st.session_state.selected_sources.discard(source_id)
                    if source_id in st.session_state.documents:
                        del st.session_state.documents[source_id]
                    reload_vectorstore_and_chain(
                        notebook_id, st.session_state.selected_sources, print_debug
                    )
            st.rerun()


@st.dialog("Confirm Deletion", width="small")
def confirm_delete_chat_history_dialog(notebook_id: str) -> None:
    """
    Confirmation dialog to clear all chat history for a notebook.

    Args:
        notebook_id: UUID of the notebook whose chat history should be deleted.
    """
    st.warning(
        "Are you sure you want to delete all chat history for this notebook? This action cannot be undone."
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Cancel", key="cancel_clear_chat", use_container_width=True):
            st.rerun()
    with col2:
        if st.button(
            "Confirm",
            key="confirm_clear_chat",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Clearing chat history..."):
                if db_middleware.delete_chat_history(notebook_id):
                    st.session_state.chat_history = []
            st.rerun()


@st.dialog("Confirm Deletion", width="small")
def confirm_delete_note_dialog(note_id: str, notebook_id: str) -> None:
    """
    Confirmation dialog to delete a saved note.

    Args:
        note_id: UUID of the note to delete.
        notebook_id: UUID of the parent notebook (for session state update).
    """
    st.warning(
        "Are you sure you want to delete this note? This action cannot be undone."
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "Cancel", key=f"cancel_del_note_{note_id}", use_container_width=True
        ):
            st.rerun()
    with col2:
        if st.button(
            "Confirm",
            key=f"confirm_del_note_{note_id}",
            type="primary",
            use_container_width=True,
        ):
            if db_middleware.delete_note(note_id):
                st.session_state.saved_notes = db_middleware.get_notes_for_notebook(
                    notebook_id
                )
            st.rerun()


# ============================================================================
# NOTEBOOK DASHBOARD
# ============================================================================
def render_dashboard() -> None:
    """
    Render the main dashboard with notebook grid.

    Displays all user notebooks in a 3-column grid layout with:
    - Notebook name, description, creation date, source count
    - Open button to load notebook workspace
    - Edit/Delete options in dropdown menu
    - Button to create new notebook
    """
    header_col1, header_col2 = st.columns([4, 1], vertical_alignment="bottom")
    with header_col1:
        st.markdown(
            f"<h1 class='main-header'>{cfg.APP_NAME}</h1>", unsafe_allow_html=True
        )
        st.caption("Your personalized NotebookLM-inspired AI assistant")
    with header_col2:
        if st.button("+ Create New Notebook", type="primary", use_container_width=True):
            create_notebook_modal()

    st.subheader("Your Notebooks")

    # Reload notebooks to ensure we have the latest
    notebooks = db_middleware.get_all_notebooks()

    if not notebooks:
        st.info(
            f"🚀 Welcome to {cfg.APP_NAME}! Create your first notebook below to get started."
        )
    else:
        # Display as a grid using columns
        cols = st.columns(3)
        for i, nb in enumerate(notebooks):
            col = cols[i % 3]
            with col:
                with st.container(border=True):
                    st.markdown(f"### {nb['name']}")
                    # Always render caption to keep all cards the same height
                    st.caption(nb["description"] or "No description")
                    source_count = len(db_middleware.get_sources_for_notebook(nb["id"]))
                    st.markdown(
                        f"<small>{nb['created_at'][:10]}  ·  {source_count} source{'s' if source_count != 1 else ''}</small>",
                        unsafe_allow_html=True,
                    )

                    c1, c2 = st.columns([3, 1])
                    with c1:
                        if st.button(
                            "Open",
                            key=f"open_{nb['id']}",
                            type="primary",
                        ):
                            st.session_state.loading_notebook_id = nb["id"]
                            st.rerun()
                    with c2:
                        with st.popover("⋮", use_container_width=True):
                            if st.button(
                                "Delete",
                                key=f"delete_{nb['id']}",
                                type="secondary",
                                use_container_width=True,
                            ):
                                confirm_delete_notebook_dialog(nb["id"], nb["name"])
                            if st.button(
                                "Edit",
                                key=f"edit_{nb['id']}",
                                use_container_width=True,
                            ):
                                rename_notebook_modal(
                                    nb["id"], nb["name"], nb.get("description")
                                )


# ============================================================================
# MAIN APP
# ============================================================================
def main() -> None:
    """
    Main application entry point and orchestrator.

    Initializes session state, handles navigation between dashboard and
    notebook workspaces, and renders the appropriate UI layout.

    Flow:
    1. Initialize session state on first app run
    2. If loading notebook: show spinner and load workspace
    3. If no notebook selected: render dashboard
    4. If notebook selected: render workspace (sidebar + chat + notes)
    """
    initialize_session_state()

    if (
        "loading_notebook_id" in st.session_state
        and st.session_state.loading_notebook_id
    ):
        st.markdown(
            """<style>[data-testid="stSidebar"] {display: none;}</style>""",
            unsafe_allow_html=True,
        )
        with st.spinner("Loading notebook workspace..."):
            load_workspace(st.session_state.loading_notebook_id, cfg.PRINT_DEBUG)
            st.session_state.loading_notebook_id = None
            st.rerun()

    if st.session_state.current_notebook_id is None:
        # Hide sidebar on dashboard
        st.markdown(
            """
            <style>
                [data-testid="collapsedControl"] {
                    display: none;
                }
                [data-testid="stSidebar"] {
                    display: none;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )
        render_dashboard()
    else:
        # Get notebook details
        notebook = db_middleware.get_notebook(st.session_state.current_notebook_id)
        notebook_name = notebook["name"] if notebook else "Unknown Notebook"

        render_notebook_settings_sidebar(st.session_state.current_notebook_id)

        # Display rename source modal if needed
        if (
            st.session_state.rename_source_modal_open
            and st.session_state.rename_source_id
        ):
            show_rename_source_dialog()

        if st.session_state.show_notes_panel:
            # Split layout into 3 sections (Source Hub | Chat | Notes)
            col1, col2, col3 = st.columns([1.2, 2.6, 1.2])

            with col1:
                source_hub_ui(cfg.PRINT_DEBUG)

            with col2:
                chat_interface(notebook_name, cfg.PRINT_DEBUG)

            with col3:
                notes_panel_ui()
        else:
            # Split layout into 2 sections (Source Hub | Chat)
            col1, col2 = st.columns([1.2, 3.8])

            with col1:
                source_hub_ui(cfg.PRINT_DEBUG)

            with col2:
                chat_interface(notebook_name, cfg.PRINT_DEBUG)


if __name__ == "__main__":
    # Check if the database exists to create or not
    if not os.path.exists(cfg.DB_ROOT_PATH):
        from db.setup import init_db

        if cfg.PRINT_DEBUG:
            debug_log(
                "INFO",
                "🗄️",
                f'Database file not found at "{cfg.DB_ROOT_PATH}". Initializing new database.',
            )

        init_db(cfg.DB_ROOT_PATH, cfg.PRINT_DEBUG)

    main()
