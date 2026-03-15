# app.py
"""
SmartDoc AI - Local NotebookLM-Inspired Document Intelligence System
A privacy-first RAG application for querying documents with source citations.
"""

import html

import streamlit as st
import os
import tempfile
import logging
import requests
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path

from middlewares import db_middleware as db
from core.configs import (
    APP_NAME,
    DB_ROOT_PATH,
    PRINT_DEBUG,
    USER_ROLE_NAME,
    ASSISTANT_ROLE_NAME,
    LLM_BASE_URL,
)
from core.utils import (
    hash_pdf_file,
    check_file_already_exists_in_notebook,
    chunk_and_process_pdf,
    create_vectorstore_from_chunks,
    merge_vectorstores,
    save_source_to_database,
    process_user_query,
    save_answer_as_note,
    reload_vectorstore_and_chain,
    load_persisted_vectorstore_filtered,
    get_notebook_vectorstore_dir,
    get_source_vectorstore_dir,
    try_load_embeddings,
    create_rag_chain,
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
    /* Compact mode - reduce all padding and spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 1400px;
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

    /* Global layout padding reduction */
    .stApp > header {
        display: none;
    }
    .css-1544g2n {
        padding-top: 1rem;
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
    """Load documents metadata from database."""
    notebook_id = st.session_state.get("current_notebook_id")
    if not notebook_id:
        return {}

    sources = db.get_sources_for_notebook(notebook_id)
    docs_dict: Dict[str, Any] = {}
    for src in sources:
        docs_dict[src["file_name"]] = {
            "loaded": True,
            "summary": src["summary"],
            "suggested_questions": src["suggested_questions"],
            "id": src["id"],
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
        st.session_state.rag_chain = create_rag_chain(
            st.session_state.vectorstore, print_debug
        )
    else:
        st.session_state.rag_chain = None

    st.session_state.chat_history = load_chat_history()
    st.session_state.saved_notes = load_saved_notes()
    st.session_state.pending_query = None
    st.session_state.file_uploader_key = 0


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


def generate_summary(rag_chain: Any) -> str:
    """Generate a brief summary of the document."""
    try:
        summary_prompt = (
            "Provide a brief 2-3 sentence summary of the main topics in this document."
        )
        summary = rag_chain.invoke(summary_prompt)
        return summary[:300]  # Limit length
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


def process_pdf(uploaded_file: Any, filename: str, print_debug: bool = False) -> bool:
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

        file_hash = hash_pdf_file(file_bytes)

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

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        with st.spinner(f"Processing '{filename}'..."):
            # Load and chunk
            chunks, num_chunks = chunk_and_process_pdf(tmp_path, filename, print_debug)

            # Embed with GPU/CPU fallback
            logger.info(f"Creating embeddings for {num_chunks} chunks")
            embeddings = try_load_embeddings()
            if embeddings is None:
                st.error("Failed to load embedding model. Please check your system.")
                return False

            new_vectorstore = create_vectorstore_from_chunks(
                chunks, embeddings, print_debug
            )

            # Merge or create, then recreate RAG chain for summary/question generation
            st.session_state.vectorstore = merge_vectorstores(
                st.session_state.vectorstore, new_vectorstore, print_debug
            )
            st.session_state.rag_chain = create_rag_chain(st.session_state.vectorstore)

            # Generate summary
            logger.info(f"Generating summary for {filename}")
            summary = generate_summary(st.session_state.rag_chain)

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
                file_hash,
                summary,
                suggested_questions,
                vectorstore_path,
                source_id=source_id,
                print_debug=print_debug,
            )

            # Update session state
            st.session_state.documents[filename] = {
                "loaded": True,
                "summary": summary,
                "suggested_questions": suggested_questions,
                "id": source_id,
            }

            # Save the individual source vectorstore to the path we calculated above
            new_vectorstore.save_local(vectorstore_path)

            # Add the new source to selected sources (auto-select)
            st.session_state.selected_sources.add(source_id)

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
        "Select PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.file_uploader_key}",
        label_visibility="collapsed",
    )

    if "pending_replacements" not in st.session_state:
        st.session_state.pending_replacements = {}

    # Typed alias — mutations propagate back to session_state via dict reference
    pending_repls: Dict[str, bytes] = st.session_state.pending_replacements  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

    if uploaded_files:
        processed_any = False
        new_pending_replacements = False

        for uploaded_file in uploaded_files:
            filename = uploaded_file.name

            if filename in st.session_state.documents:
                # Store in memory to ask user, if not already asking
                if filename not in pending_repls:
                    pending_repls[filename] = uploaded_file.getvalue()
                    new_pending_replacements = True
            else:
                # Process right away
                if process_pdf(uploaded_file, filename, True):
                    processed_any = True

        # Clear the uploader after reading the batch
        st.session_state.file_uploader_key += 1

        if processed_any or new_pending_replacements:
            st.rerun()

    # Show pending replacements if any are saved in session state
    if pending_repls:
        st.markdown("##### Pending Duplicate Files")
        # Copy keys to a list to safely iterate
        pending_files: List[str] = list(pending_repls.keys())

        for filename in pending_files:
            container = st.container()
            with container:
                st.warning(f"**{filename}** is already loaded.")
                col_btn1, col_btn2 = st.columns(2)

                # We use a placeholder for spinner so it doesn't get constrained simply to the column
                status_placeholder = st.empty()

                with col_btn1:
                    replace_clicked = st.button(
                        "Replace", key=f"replace_{filename}", use_container_width=True
                    )
                with col_btn2:
                    cancel_clicked = st.button(
                        "Cancel", key=f"cancel_{filename}", use_container_width=True
                    )

                if replace_clicked:
                    status_placeholder.info(
                        f"⏳ Replacing '{filename}'... Please wait."
                    )
                    file_bytes: bytes = pending_repls[filename]

                    # Remove the file from pending immediately so it doesn't get processed again
                    del pending_repls[filename]

                    # Phase 1: Remove old version
                    doc_info = st.session_state.documents.get(filename, {})
                    if "id" in doc_info:
                        db.delete_source(doc_info["id"])
                        source_dir = get_source_vectorstore_dir(
                            st.session_state.current_notebook_id, doc_info["id"]
                        )
                        if source_dir.exists():
                            import shutil

                            shutil.rmtree(source_dir, ignore_errors=True)

                        # Remove from selected sources if it was selected
                        st.session_state.selected_sources.discard(doc_info["id"])

                    if filename in st.session_state.documents:
                        del st.session_state.documents[filename]

                    # Reload updated vectorstore with filtered selection
                    reload_vectorstore_and_chain(
                        st.session_state.current_notebook_id,
                        st.session_state.selected_sources,
                        print_debug,
                    )

                    # Phase 2: Insert the new one
                    process_pdf(file_bytes, filename, True)

                    status_placeholder.empty()
                    st.rerun()

                if cancel_clicked:
                    del pending_repls[filename]
                    st.rerun()

    st.markdown("#### Loaded Documents")

    if st.session_state.documents:
        # "Select all sources" row with checkbox on the right (matching NotebookLM style)
        all_selected = (
            len(st.session_state.selected_sources) == len(st.session_state.documents)
            and len(st.session_state.documents) > 0
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
                value=all_selected,
                key="select_all_sources",
                on_change=on_select_all_change,
                label_visibility="collapsed",
            )

        # Display sources with checkbox on the RIGHT (matching NotebookLM style)
        for doc_name, doc_info in st.session_state.documents.items():
            doc_id = doc_info.get("id")

            # Initialize checkbox state if needed
            checkbox_key = f"checkbox_{doc_id}"
            if checkbox_key not in st.session_state:
                st.session_state[checkbox_key] = (
                    doc_id in st.session_state.selected_sources
                )

            # Callback to handle checkbox changes
            def on_checkbox_change(source_id: str, checkbox_key_param: str):
                """Handle checkbox state change."""
                if st.session_state[checkbox_key_param]:
                    st.session_state.selected_sources.add(source_id)
                else:
                    st.session_state.selected_sources.discard(source_id)

                # Reload vectorstore
                reload_vectorstore_and_chain(
                    st.session_state.current_notebook_id,
                    st.session_state.selected_sources,
                    print_debug,
                )

                # Update "Select all sources" checkbox state based on current selection
                all_ids = {di["id"] for di in st.session_state.documents.values()}
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
                    with st.expander(f"📄 {doc_name}", expanded=False):
                        if doc_info.get("summary"):
                            st.caption(f"**Summary:** {doc_info['summary']}")

                        col_detail1, col_detail2 = st.columns([3, 1])
                        with col_detail1:
                            pass  # Space for alignment
                        with col_detail2:
                            with st.popover("⋮", use_container_width=False):
                                if st.button(
                                    "Delete",
                                    key=f"delete_{doc_name}",
                                    type="secondary",
                                    use_container_width=True,
                                ):
                                    with st.spinner(f"Removing {doc_name}..."):
                                        logger.info(f"Removing document: {doc_name}")
                                        if doc_id:
                                            db.delete_source(doc_id)
                                            source_dir = get_source_vectorstore_dir(
                                                st.session_state.current_notebook_id,
                                                doc_id,
                                            )
                                            if source_dir.exists():
                                                import shutil

                                                shutil.rmtree(
                                                    source_dir, ignore_errors=True
                                                )

                                            # Remove from selected sources if selected
                                            st.session_state.selected_sources.discard(
                                                doc_id
                                            )

                                        del st.session_state.documents[doc_name]

                                        # Reload vectorstore
                                        reload_vectorstore_and_chain(
                                            st.session_state.current_notebook_id,
                                            st.session_state.selected_sources,
                                            print_debug,
                                        )
                                        st.rerun()
                                st.button(
                                    "Rename",
                                    key=f"rename_{doc_name}",
                                    disabled=True,
                                    use_container_width=True,
                                    help="Coming soon",
                                )

                with col_check:
                    # Checkbox on the RIGHT side
                    st.checkbox(
                        label="Select",
                        key=checkbox_key,
                        on_change=on_checkbox_change,
                        args=(doc_id, checkbox_key),
                        label_visibility="collapsed",
                    )
    else:
        st.info("No documents loaded yet.")

    st.markdown("#### System Status")

    if st.session_state.ollama_ready:
        st.success("Ollama: Connected")
    else:
        st.error("Ollama: Offline")
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
        st.info("Upload documents in the Source Hub to get started.")
        return

    # Check if any sources are selected
    if not st.session_state.selected_sources:
        st.warning(
            "⚠️ No sources selected. Select at least one source in the Source Hub to ask questions."
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
                first_doc = list(st.session_state.documents.keys())[0]
                doc_info = st.session_state.documents[first_doc]

                if doc_info.get("suggested_questions"):
                    st.markdown("#### 💡 Suggested Questions")
                    for question in doc_info["suggested_questions"]:
                        if st.button(
                            question,
                            use_container_width=False,
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

    if user_query:
        # Add user message to history
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

        # Generate answer
        if st.session_state.rag_chain is None:
            st.error(
                "❌ RAG Chain not initialized. Please ensure documents are loaded."
            )
        else:
            with chat_container:
                with st.spinner("🤔 Thinking..."):
                    try:
                        # Get answer and sources via RAG chain
                        logger.info(f"Processing query: {user_query[:50]}...")
                        answer, sources, found_answer = process_user_query(
                            user_query,
                            st.session_state.rag_chain,
                            st.session_state.vectorstore,
                            print_debug,
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

                        # Save as note
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            if st.button(
                                "💾 Save Note",
                                key=f"save_{len(st.session_state.chat_history)}",
                                use_container_width=True,
                            ):
                                save_answer_as_note(
                                    st.session_state.current_notebook_id,
                                    user_query,
                                    answer,
                                    print_debug,
                                )
                                # Refresh notes from DB
                                st.session_state.saved_notes = (
                                    db.get_notes_for_notebook(
                                        st.session_state.current_notebook_id
                                    )
                                )
                                st.success("✅ Saved to study notes")

                    except Exception as e:
                        error_msg = str(e)
                        logger.error(f"Error: {error_msg}")

                        # Check for CUDA/memory-related errors
                        if (
                            "cuda" in error_msg.lower()
                            or "out of memory" in error_msg.lower()
                        ):
                            st.error(
                                "❌ GPU Memory Error: The model ran out of GPU memory. "
                                "Please try:\n"
                                "1. Asking a simpler question\n"
                                "2. Reducing the number of documents selected\n"
                                "3. Restarting the Ollama service\n"
                                "4. Using a smaller model\n\n"
                                f"Technical details: {error_msg}"
                            )
                        else:
                            st.error(f"❌ Error: {error_msg}")

        # Rerun to refresh chat display
        st.rerun()

    # Study notes section at the bottom
    if st.session_state.saved_notes:
        st.markdown("### 📝 Study Notes")

        # Show saved notes count
        with st.expander(f"View {len(st.session_state.saved_notes)} saved note(s)"):
            for _, note in enumerate(st.session_state.saved_notes):
                col1, col2 = st.columns([20, 1])

                with col1:
                    st.markdown(
                        f"""
                        <div class='saved-note'>
                        <strong>Q: {note.get("title", "N/A")}</strong><br>
                        <small>{note.get("content", "N/A")}</small>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col2:
                    if st.button(
                        "🗑️", key=f"delete_note_{note['id']}", help="Delete note"
                    ):
                        db.delete_note(note["id"])
                        st.session_state.saved_notes = db.get_notes_for_notebook(
                            st.session_state.current_notebook_id
                        )
                        st.rerun()


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
            new_id = db.create_notebook(new_name, new_desc)
            st.session_state.loading_notebook_id = new_id
            st.rerun()


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
                    new_description
                    if new_description != (current_description or "")
                    else None,
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
        if st.button("Create New Notebook", type="primary", use_container_width=True):
            create_notebook_modal()

    st.subheader("Your Notebooks")

    # Reload notebooks to ensure we have the latest
    notebooks = db.get_all_notebooks()

    if not notebooks:
        st.info(
            f"Welcome to {APP_NAME}! Create your first notebook below to get started."
        )
    else:
        # Display as a grid using columns
        cols = st.columns(3)
        for i, nb in enumerate(notebooks):
            col = cols[i % 3]
            with col:
                with st.container(border=True):
                    st.markdown(f"### {nb['name']}")
                    if nb["description"]:
                        st.caption(nb["description"])
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

        # Split layout into 2 sections (Source Hub vs Chat)
        col1, col2 = st.columns([1.2, 2.5])

        with col1:
            source_hub_ui(PRINT_DEBUG)

        with col2:
            chat_interface(notebook_name, PRINT_DEBUG)


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
