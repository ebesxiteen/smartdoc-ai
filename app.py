# app.py
"""
SmartDoc AI - Local NotebookLM-Inspired Document Intelligence System
A privacy-first RAG application for querying documents with source citations.
"""

import time
import html

import streamlit as st
import os
import tempfile
import logging
import requests
import uuid
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path

from middlewares import db_middleware as db
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document

from core.configs import (
    APP_NAME,
    DB_ROOT_PATH,
    PRINT_DEBUG,
    USER_ROLE_NAME,
    ASSISTANT_ROLE_NAME,
    LLM_BASE_URL,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    LLM_NUM_CTX,
)
from core.utils import (
    format_relative_time,
    hash_file_content,
    detect_file_type,
    check_file_already_exists_in_notebook,
    chunk_and_process_file,
    create_vectorstore_from_chunks,
    merge_vectorstores,
    save_source_to_database,
    process_user_query,
    reload_vectorstore_and_chain,
    load_persisted_vectorstore_filtered,
    get_notebook_vectorstore_dir,
    get_source_vectorstore_dir,
    try_load_embeddings,
    create_history_aware_rag_chain,  # ← For history-aware retrieval
)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS & PATHS
# ============================================================================
DATA_DIR = Path("data")
OLLAMA_BASE_URL = LLM_BASE_URL


DATA_DIR.mkdir(exist_ok=True)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title=APP_NAME,
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    /* Progress Bar */
    div[data-testid="stProgress"] {
        height: 6px !important;
        margin-bottom: 10px;
    }
    /* The track (background) of the progress bar */
    div[data-testid="stProgress"] > div > div {
        background-color: #cccccc !important;
    }
    /* The filled portion of the progress bar */
    div[data-testid="stProgress"] > div > div > div > div {
        background-color: #000000 !important;
    }
    .progress-status-text {
        font-size: 0.9rem;
        color: #444444;
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
        color: #111111;
    }

    .source-citation {
        background-color: #f8f8f8;
        padding: 0.8em;
        border-left: 4px solid #333333;
        border-radius: 4px;
        margin: 0.5em 0;
        font-size: 0.85em;
    }

    .saved-note {
        background-color: #fff9e6;
        padding: 0.8em;
        border-left: 4px solid #ffa500;
        border-radius: 4px;
        margin: 0.4em 0;
        font-size: 0.85em;
    }

    .error-box {
        background-color: #ffe6e6;
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
        background-color: #f5f5f5;
        border-radius: 5px 15px 15px 15px;
        padding: 10px 15px;
        border: 1px solid #ddd;
    }

    /* Black buttons with white text (Streamlit 1.45+ uses stBaseButton-{type}) */
    button[data-testid="stBaseButton-primary"] {
        background-color: black !important;
        color: #ffffff !important;
        border: 1px solid black !important;
    }

    button[data-testid="stBaseButton-primary"]:hover {
        background-color: #333333 !important;
        border-color: #333333 !important;
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
        color: #555 !important;
        text-decoration: underline !important;
        white-space: nowrap !important;
        font-size: 0.85rem !important;
        cursor: pointer !important;
    }
    section[data-testid="stSidebar"]
    [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(3))
    [data-testid="stColumn"]:nth-child(n+2) button:hover {
        color: #000 !important;
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
        background-color: #f0f4f9;
        border-radius: 10px;
        padding: 0.75rem !important;
    }
    [data-testid="stColumn"]:has(.chat-section-bg) {
        background-color: #f9f9fb;
        border-radius: 10px;
        padding: 0.75rem !important;
    }
    [data-testid="stColumn"]:has(.notes-panel-bg) {
        background-color: #f4faf2;
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
        return db.get_chat_history(notebook_id)
    return []


def load_saved_notes() -> List[Dict[str, Any]]:
    """Load saved notes from database."""
    notebook_id = st.session_state.get("current_notebook_id")
    if notebook_id:
        return db.get_notes_for_notebook(notebook_id)
    return []


def load_documents_state() -> Dict[str, Any]:
    """Load documents metadata from database. Key by source ID (not filename, to avoid duplicates)."""
    notebook_id = st.session_state.get("current_notebook_id")
    if not notebook_id:
        return {}

    sources = db.get_sources_for_notebook(notebook_id)
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
    st.session_state.current_notebook_id = notebook_id
    st.session_state.documents = load_documents_state()

    # Initialize selected sources to all sources by default
    sources = db.get_sources_for_notebook(notebook_id)
    st.session_state.selected_sources = {src["id"] for src in sources}

    st.session_state.vectorstore = load_persisted_vectorstore_filtered(
        notebook_id, st.session_state.selected_sources
    )

    if st.session_state.vectorstore is not None:
        # Use history-aware RAG chain for better context understanding
        st.session_state.chat_history = load_chat_history()
        st.session_state.rag_chain = create_history_aware_rag_chain(
            vectorstore=st.session_state.vectorstore,
            chat_history=st.session_state.chat_history,
            print_debug=print_debug,
        )
    else:
        st.session_state.rag_chain = None

    st.session_state.chat_history = load_chat_history()
    st.session_state.saved_notes = load_saved_notes()
    st.session_state.pending_query = None
    st.session_state.file_uploader_key = 0
    st.session_state.pending_new_uploads = {}


def check_ollama_connection():
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        if response.status_code == 200:
            return True
        return False
    except Exception as e:
        logger.error(f"Ollama connection failed: {str(e)}")
        return False


def initialize_session_state():
    """Initialize all global session state variables on first app run."""
    if "notebooks" not in st.session_state:
        st.session_state.notebooks = db.get_all_notebooks()

    if "current_notebook_id" not in st.session_state:
        st.session_state.current_notebook_id = None

    if "ollama_ready" not in st.session_state:
        st.session_state.ollama_ready = check_ollama_connection()

    if "file_uploader_key" not in st.session_state:
        st.session_state.file_uploader_key = 0

    if "rename_source_modal_open" not in st.session_state:
        st.session_state.rename_source_modal_open = False

    if "rename_source_id" not in st.session_state:
        st.session_state.rename_source_id = None

    if "rename_source_name" not in st.session_state:
        st.session_state.rename_source_name = None


def generate_summary(chunks: List[Document]) -> str:
    """Generate a brief summary of the document using Top-K Slicing."""
    try:
        # Get the first 3 chunks (or fewer if the document is very short)
        top_k_chunks = chunks[:3]
        context_text = "\n\n".join([chunk.page_content for chunk in top_k_chunks])

        summary_prompt = (
            f"Please read the following text extracted from the beginning of a document.\n"
            f"Provide a brief 2-3 sentence summary of the main topics in this document.\n\n"
            f"TEXT:\n{context_text}\n\n"
            f"SUMMARY:"
        )

        # Initialize standalone LLM
        llm = OllamaLLM(
            model=LLM_MODEL_NAME,
            base_url=LLM_BASE_URL,
            temperature=LLM_TEMPERATURE,
            num_ctx=LLM_NUM_CTX,
        )

        summary = llm.invoke(summary_prompt)

        summary = (
            summary.replace("[FOUND_ANSWER: true]", "")
            .replace("[FOUND_ANSWER: false]", "")
            .replace("[FOUND_ANSWER: general]", "")
            .strip()
        )
        return summary
    except Exception as e:
        logger.error(f"Summary generation failed: {str(e)}")
        return "Unable to generate summary."


def generate_suggested_questions(rag_chain: Any) -> List[str]:
    """Generate 3-4 suggested questions based on document content."""
    try:
        question_prompt = (
            "Generate exactly 3 specific and interesting questions that a reader might ask about this document. "
            "Format as: 1. Question? 2. Question? 3. Question?"
        )
        response = rag_chain.invoke(question_prompt)

        if isinstance(response, str):
            response = (
                response.replace("[FOUND_ANSWER: true]", "")
                .replace("[FOUND_ANSWER: false]", "")
                .replace("[FOUND_ANSWER: general]", "")
                .strip()
            )

        questions: List[str] = []
        for line in response.split("\n"):
            if line.strip() and (line[0].isdigit() or line.startswith("-")):
                question = line.lstrip("0123456789.-) ").strip()
                if question and len(question) > 5 and "?" in question:
                    questions.append(question)

        return questions[:4]
    except Exception as e:
        logger.error(f"Question generation failed: {str(e)}")
        return []


def process_file(
    uploaded_file: Any,
    filename: str,
    print_debug: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> bool:
    """Process a PDF: extract, chunk, embed, merge with existing data."""
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
            # Load and chunk
            chunks, num_chunks = chunk_and_process_file(
                tmp_path,
                file_type,
                filename,
                print_debug,
                progress_callback=progress_callback,
            )

            # Embed with GPU/CPU fallback
            logger.info(f"Creating embeddings for {num_chunks} chunks")
            if progress_callback:
                progress_callback("Loading embedding model...")
            embeddings = try_load_embeddings()
            if embeddings is None:
                st.error("Failed to load embedding model. Please check your system.")
                return False

            if progress_callback:
                progress_callback("Creating vector store...")
            new_vectorstore = create_vectorstore_from_chunks(
                chunks, embeddings, print_debug
            )

            # Merge or create, then recreate RAG chain for summary/question generation
            st.session_state.vectorstore = merge_vectorstores(
                st.session_state.vectorstore, new_vectorstore, print_debug
            )
            # Use history-aware RAG chain (with or without history)
            st.session_state.rag_chain = create_history_aware_rag_chain(
                vectorstore=st.session_state.vectorstore,
                chat_history=st.session_state.get(
                    "chat_history"
                ),  # May be None on first upload
                print_debug=print_debug,
            )

            # Generate summary
            logger.info(f"Generating summary for {filename}")
            summary = generate_summary(chunks)

            # Generate suggested questions
            logger.info(f"Generating suggested questions for {filename}")
            suggested_questions = generate_suggested_questions(
                st.session_state.rag_chain
            )

            # Save to Database
            notebook_id = st.session_state.current_notebook_id

            # Generate source_id upfront so we know the vectorstore path
            source_id = str(uuid.uuid4())

            # Calculate vectorstore directory path
            src_vs_dir = get_source_vectorstore_dir(notebook_id, source_id)
            # Store the relative path for persistence
            vectorstore_path = str(src_vs_dir)

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

            # Save the individual source vectorstore to the path we calculated above
            new_vectorstore.save_local(vectorstore_path)

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

            logger.info(f"Successfully processed {filename}")

        st.success(f"Successfully loaded '{filename}'")

        return True

    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}")
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
def go_back_to_notebooks():
    st.session_state.current_notebook_id = None


def source_hub_ui(print_debug: bool = False):
    """The 'Source Hub' for document management."""
    st.markdown(
        '<div class="source-hub-bg" style="display:none"></div>', unsafe_allow_html=True
    )
    # Header with back arrow and title (vertically centered)
    col1, col2 = st.columns([0.5, 3], vertical_alignment="center")
    with col1:
        if st.button("←", use_container_width=True, help="Back to Notebooks"):
            go_back_to_notebooks()
            st.rerun()
    with col2:
        st.markdown("## Source Hub")

    # Show notebook description if available
    notebook = db.get_notebook(st.session_state.current_notebook_id)
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
    pending_repls: Dict[str, bytes] = st.session_state.pending_replacements  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    pending_new: Dict[str, bytes] = st.session_state.pending_new_uploads  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

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

                def update_granular_status(status_message: str):
                    file_list_html = ""
                    for j, (fname_iter, _) in enumerate(files_to_process):
                        if j < i:
                            # Already processed
                            file_list_html += f"<div style='color: green; font-size: 0.85em; margin-bottom: 2px;'>✨ {fname_iter}</div>"
                        elif j == i:
                            # Currently processing
                            file_list_html += f"<div style='color: black; font-weight: 500; font-size: 0.85em; margin-bottom: 2px;'>🚀 {fname_iter} <br><span style='font-size: 0.9em; font-weight: normal; color: #444; margin-left: 15px;'>↳ <i>{status_message}</i></span></div>"
                        else:
                            # Pending
                            file_list_html += f"<div style='color: #999; font-size: 0.85em; margin-bottom: 2px;'>⚡ {fname_iter}</div>"

                    status_text.markdown(
                        f"<div class='progress-status-text' style='margin-bottom: 10px;'><b>Processing {num_files} files:</b><br>{file_list_html}</div>",
                        unsafe_allow_html=True,
                    )

                    max_val_for_file = 0.95 / num_files
                    remaining = max_val_for_file - prog_state["val"]
                    prog_state["val"] += remaining * 0.1

                    current_val = min(base_progress + prog_state["val"], 1.0)
                    progress_bar.progress(current_val)

                update_granular_status("Initializing...")

                process_file(
                    fbytes,
                    fname,
                    PRINT_DEBUG,
                    progress_callback=update_granular_status,
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

                def update_granular_status(status_message: str):
                    status_text.markdown(
                        f"<div class='progress-status-text'><b>Processing:</b> {filename} <br> ↳ <i>{status_message}</i></div>",
                        unsafe_allow_html=True,
                    )
                    remaining = 0.95 - prog_state["val"]
                    prog_state["val"] += remaining * 0.1
                    progress_bar.progress(prog_state["val"])

                update_granular_status("Initializing...")

                process_file(
                    file_bytes_new,
                    filename,
                    PRINT_DEBUG,
                    progress_callback=update_granular_status,
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

                def update_granular_status(status_message: str):
                    status_text.markdown(
                        f"<div class='progress-status-text'><b>Replacing:</b> {filename} <br> ↳ <i>{status_message}</i></div>",
                        unsafe_allow_html=True,
                    )
                    remaining = 0.95 - prog_state["val"]
                    prog_state["val"] += remaining * 0.1
                    progress_bar.progress(prog_state["val"])

                update_granular_status("Removing old version...")

                # Phase 1: Remove old version
                if source_id in st.session_state.documents:
                    db.delete_source(source_id)
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
                    progress_callback=update_granular_status,
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
                                    with st.spinner(f"Removing {file_name}..."):
                                        logger.info(f"Removing document: {file_name}")
                                        db.delete_source(source_id)
                                        source_dir = get_source_vectorstore_dir(
                                            st.session_state.current_notebook_id,
                                            source_id,
                                        )
                                        if source_dir.exists():
                                            import shutil

                                            shutil.rmtree(
                                                source_dir, ignore_errors=True
                                            )

                                        # Remove from selected sources if selected
                                        st.session_state.selected_sources.discard(
                                            source_id
                                        )

                                        del st.session_state.documents[source_id]

                                        # Reload vectorstore
                                        reload_vectorstore_and_chain(
                                            st.session_state.current_notebook_id,
                                            st.session_state.selected_sources,
                                            print_debug,
                                        )
                                        st.rerun()

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

    st.caption(
        "💡 **Tip:** If you experience GPU out-of-memory errors, try selecting fewer documents before asking questions. "
        "Start with 1-2 documents and gradually add more as needed."
    )


# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================
def render_user_message(content: str):
    """Renders a chat message from the user aligned to the right."""
    escaped_content = html.escape(content).replace("\n", "<br>")
    st.markdown(
        f"""
        <div style="display: flex; justify-content: flex-end; margin-bottom: 0.5rem; width: 100%;">
            <div style="background-color: #f0f0f0; color: #111; padding: 0.6rem 1rem; border-radius: 15px 5px 15px 15px; border: 1px solid #ddd; max-width: 80%; text-align: left; box-shadow: 0 1px 2px rgba(0,0,0,0.05); font-family: sans-serif; font-size: 0.95em;">
                {escaped_content}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chat_interface(notebook_name: str, print_debug: bool = False):
    """Main chat area - NotebookLM style."""
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
                "Delete chat history",
                key=f"clear_chat_{st.session_state.current_notebook_id}",
                type="secondary",
                use_container_width=True,
                disabled=not has_messages,
            ):
                with st.spinner("Clearing chat history..."):
                    # Delete from database
                    db.delete_chat_history(st.session_state.current_notebook_id)
                    # Clear session state
                    st.session_state.chat_history = []
                st.success("✅ Chat history cleared!")
                st.rerun()

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
            if message["role"] == USER_ROLE_NAME:
                render_user_message(message["content"])
            else:
                # Assistant message
                st.markdown(message["content"])

                if message.get("sources") and message.get("found_answer", True):
                    with st.expander("✨ View sources", expanded=False):
                        for j, source in enumerate(message["sources"]):
                            st.markdown(
                                f"""
                                    <div class='source-citation'>
                                    <strong>{j + 1}. {source["document"]} — Page {source["page"]}</strong><br>
                                    <small>{source["content"]}</small>
                                    </div>
                                    """,
                                unsafe_allow_html=True,
                            )

        # Auto-scroll marker: scroll to this element when new messages arrive
        st.markdown('<div id="chat-scroll-anchor"></div>', unsafe_allow_html=True)

        # Inject JavaScript to auto-scroll to bottom when new messages are added
        st.markdown(
            """
            <script>
            // Auto-scroll to bottom of chat container
            const chatAnchors = document.querySelectorAll('div#chat-scroll-anchor');
            if (chatAnchors.length > 0) {
                const lastAnchor = chatAnchors[chatAnchors.length - 1];
                lastAnchor.parentElement.parentElement.parentElement.scrollTop = lastAnchor.parentElement.parentElement.parentElement.scrollHeight;
            }
            </script>
            """,
            unsafe_allow_html=True,
        )

    # DETECT INTERRUPTED GENERATIONS: If the last message in history is from a User without an Assistant response
    needs_answer = False
    query_to_answer = ""
    # We slice out the last message if it's the pending query, but we also handle brand new inputs.
    if user_query:
        query_to_answer = user_query
        needs_answer = True
        # Save user message immediately
        db.add_chat_message(
            notebook_id=st.session_state.current_notebook_id,
            role=USER_ROLE_NAME,
            content=user_query,
        )
        st.session_state.chat_history.append(
            {"role": USER_ROLE_NAME, "content": user_query, "sources": None}
        )
        # Display user message immediately
        with chat_container:
            render_user_message(user_query)
    elif (
        len(st.session_state.chat_history) > 0
        and st.session_state.chat_history[-1]["role"] == USER_ROLE_NAME
    ):
        needs_answer = True
        query_to_answer = st.session_state.chat_history[-1]["content"]
        # It's already rendered via the loop above!

    if needs_answer:
        # Generate answer
        if st.session_state.rag_chain is None:
            error_msg = "⚠️ Generation interrupted or RAG Chain not initialized. Please ensure documents are selected and ask your question again."
            db.add_chat_message(
                notebook_id=st.session_state.current_notebook_id,
                role=ASSISTANT_ROLE_NAME,
                content=error_msg,
                sources=None,
                found_answer=False,
            )
            st.session_state.chat_history.append(
                {
                    "role": ASSISTANT_ROLE_NAME,
                    "content": error_msg,
                    "sources": None,
                    "found_answer": False,
                }
            )
            st.rerun()
        else:
            with chat_container:
                with st.spinner("🤔 Thinking..."):
                    try:
                        # Get answer and sources via RAG chain with chat history context
                        logger.info(f"Processing query: {query_to_answer[:50]}...")
                        # Exclude the current query from the chat history passed for context!
                        history_context = st.session_state.chat_history[:-1]

                        answer, sources, found_answer = process_user_query(
                            query_to_answer,
                            st.session_state.rag_chain,
                            st.session_state.vectorstore,
                            chat_history=history_context,
                            print_debug=print_debug,
                        )

                        # Display answer
                        st.markdown(answer)

                        # Add to history with found_answer flag
                        db.add_chat_message(
                            notebook_id=st.session_state.current_notebook_id,
                            role=ASSISTANT_ROLE_NAME,
                            content=answer,
                            sources=sources,
                            found_answer=found_answer,
                        )
                        st.session_state.chat_history.append(
                            {
                                "role": ASSISTANT_ROLE_NAME,
                                "content": answer,
                                "sources": sources,
                                "found_answer": found_answer,
                            }
                        )

                        # Display sources (only if LLM found relevant context)
                        if sources and found_answer:
                            with st.expander("✨ View sources", expanded=False):
                                for j, source in enumerate(sources):
                                    st.markdown(
                                        f"""
                                        <div class='source-citation'>
                                        <strong>{j + 1}. {source["document"]} — Page {source["page"]}</strong><br>
                                        <small>{source["content"]}</small>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                    except Exception as e:
                        error_msg = str(e)
                        logger.error(f"Error: {error_msg}")

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

                        db.add_chat_message(
                            notebook_id=st.session_state.current_notebook_id,
                            role=ASSISTANT_ROLE_NAME,
                            content=display_error,
                            sources=None,
                            found_answer=False,
                        )
                        st.session_state.chat_history.append(
                            {
                                "role": ASSISTANT_ROLE_NAME,
                                "content": display_error,
                                "sources": None,
                                "found_answer": False,
                            }
                        )

        # Rerun to refresh chat display
        st.rerun()


# ============================================================================
# NOTES PANEL
# ============================================================================
@st.dialog("New Note")
def create_note_modal():
    title = st.text_input("Title", placeholder="Note title...")
    content = st.text_area("Content", placeholder="Write your note here...", height=200)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Cancel", use_container_width=True):
            st.rerun()
    with col2:
        if st.button("Save", type="primary", use_container_width=True):
            try:
                db.add_note(st.session_state.current_notebook_id, title, content)
                st.session_state.saved_notes = db.get_notes_for_notebook(
                    st.session_state.current_notebook_id
                )
                st.rerun()
            except ValueError as e:
                st.error(str(e))


@st.dialog("Edit Note")
def edit_note_modal(note_id: str, current_title: str, current_content: str):
    title = st.text_input("Title", value=current_title)
    content = st.text_area("Content", value=current_content or "", height=200)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Cancel", key="edit_note_cancel", use_container_width=True):
            st.rerun()
    with col2:
        if st.button(
            "Save", key="edit_note_save", type="primary", use_container_width=True
        ):
            try:
                db.update_note(note_id, title=title, content=content)
                st.session_state.saved_notes = db.get_notes_for_notebook(
                    st.session_state.current_notebook_id
                )
                st.rerun()
            except ValueError as e:
                st.error(str(e))


def notes_panel_ui():
    """Dedicated Notes panel — lists all notes, supports add/edit/delete."""
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
            st.caption(f"{note_count} note{'s' if note_count != 1 else ''}")

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
                            db.delete_note(note["id"])
                            st.session_state.saved_notes = db.get_notes_for_notebook(
                                st.session_state.current_notebook_id
                            )
                            st.rerun()
        else:
            st.info("🛸 No notes yet.")

    st.markdown('<div class="add-note-btn-anchor"></div>', unsafe_allow_html=True)
    if st.button("+ Add Note", type="primary", use_container_width=True):
        create_note_modal()


def delete_notebook_callback(nb_id: str):
    db.delete_notebook(nb_id)
    # Clean up vectorstore if exists
    vs_dir = get_notebook_vectorstore_dir(nb_id)
    if vs_dir.exists():
        import shutil

        shutil.rmtree(vs_dir, ignore_errors=True)


@st.dialog("Create New Notebook")
def create_notebook_modal():
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
                new_id = db.create_notebook(new_name, new_desc)
                st.session_state.loading_notebook_id = new_id
                st.rerun()
            except ValueError as e:
                st.error(f"❌ {str(e)}")


@st.dialog("Rename Notebook")
def rename_notebook_modal(
    notebook_id: str, current_name: str, current_description: Optional[str] = None
):
    """Modal dialog to rename a notebook."""
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
        if st.button("Cancel", use_container_width=True):
            st.rerun()

    with col2:
        if st.button("Save", type="primary", use_container_width=True):
            try:
                # Pass through middleware for validation
                db.rename_notebook(
                    notebook_id,
                    new_name if new_name != current_name else None,
                    (
                        new_description
                        if new_description != (current_description or "")
                        else None
                    ),
                )
                st.success("✅ Notebook renamed successfully!")
                st.session_state.notebooks = (
                    db.get_all_notebooks()
                )  # Reload notebook list
                st.rerun()
            except ValueError as e:
                st.error(f"❌ {str(e)}")
            except Exception as e:
                logger.error(f"Error renaming notebook {notebook_id}: {str(e)}")
                st.error(f"Error renaming notebook: {str(e)}")


def rename_source_modal(source_id: str, current_name: str):
    """Modal dialog to rename a source file."""
    st.markdown("Rename source:")

    new_name = st.text_input(
        "Filename",
        value=current_name,
        placeholder="e.g., Biology Textbook, Research Paper...",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Cancel", use_container_width=True):
            st.session_state.rename_source_modal_open = False
            st.rerun()

    with col2:
        if st.button("Save", type="primary", use_container_width=True):
            try:
                if new_name == current_name:
                    st.warning("No changes made.")
                    return

                # Pass through middleware for validation
                db.rename_source(source_id, new_name)
                st.success("✅ Source renamed successfully!")

                # Update session state documents to reflect change (keyed by source_id)
                if source_id in st.session_state.documents:
                    st.session_state.documents[source_id]["file_name"] = new_name

                st.session_state.rename_source_modal_open = False
                st.rerun()
            except ValueError as e:
                st.error(f"❌ {str(e)}")
            except Exception as e:
                logger.error(f"Error renaming source {source_id}: {str(e)}")
                st.error(f"Error renaming source: {str(e)}")


@st.dialog("Rename Source", width="small")
def show_rename_source_dialog():
    """Display rename source modal dialog."""
    if st.session_state.rename_source_id:
        rename_source_modal(
            st.session_state.rename_source_id, st.session_state.rename_source_name or ""
        )


# ============================================================================
# NOTEBOOK DASHBOARD
# ============================================================================
def render_dashboard():
    """Renders the grid of notebooks to select or create."""
    header_col1, header_col2 = st.columns([4, 1], vertical_alignment="bottom")
    with header_col1:
        st.markdown(f"<h1 class='main-header'>{APP_NAME}</h1>", unsafe_allow_html=True)
        st.caption("Your personalized NotebookLM-inspired AI assistant")
    with header_col2:
        if st.button("+ Create New Notebook", type="primary", use_container_width=True):
            create_notebook_modal()

    st.subheader("Your Notebooks")

    # Reload notebooks to ensure we have the latest
    notebooks = db.get_all_notebooks()

    if not notebooks:
        st.info(
            f"🚀 Welcome to {APP_NAME}! Create your first notebook below to get started."
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
                    source_count = len(db.get_sources_for_notebook(nb["id"]))
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
                                delete_notebook_callback(nb["id"])
                                st.rerun()
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
def main():
    """Main application entry point."""
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
            load_workspace(st.session_state.loading_notebook_id, PRINT_DEBUG)
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
        # Hide the sidebar in the workspace as well to use a custom 2-column layout
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

        # Get notebook details
        notebook = db.get_notebook(st.session_state.current_notebook_id)
        notebook_name = notebook["name"] if notebook else "Unknown Notebook"

        # Display rename source modal if needed
        if (
            st.session_state.rename_source_modal_open
            and st.session_state.rename_source_id
        ):
            show_rename_source_dialog()

        # Split layout into 3 sections (Source Hub | Chat | Notes)
        col1, col2, col3 = st.columns([1.2, 2.8, 1.2])

        with col1:
            source_hub_ui(PRINT_DEBUG)

        with col2:
            chat_interface(notebook_name, PRINT_DEBUG)

        with col3:
            notes_panel_ui()


if __name__ == "__main__":
    # Check if the database exists to create or not
    if not os.path.exists(DB_ROOT_PATH):
        from db.setup import init_db

        if PRINT_DEBUG:
            print(
                f"⚠️ Database file not found at {DB_ROOT_PATH}. Initializing new database."
            )

        init_db(DB_ROOT_PATH, PRINT_DEBUG)

    main()
