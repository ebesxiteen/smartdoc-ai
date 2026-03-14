# app.py
"""
SmartDoc AI - Local NotebookLM-Inspired Document Intelligence System
A privacy-first RAG application for querying documents with source citations.
"""

import streamlit as st
import os
import tempfile
import logging
import html
import requests
from typing import List, Dict
from pathlib import Path

from ai.ingest import load_and_chunk_pdf
from ai.rag_chain import create_rag_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from middlewares import db_middleware as db
from core.configs import USER_ROLE_NAME, ASSISTANT_ROLE_NAME

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
OLLAMA_BASE_URL = "http://localhost:11434"


DATA_DIR.mkdir(exist_ok=True)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="SmartDoc AI",
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
        color: #1f77b4;
    }

    .source-citation {
        background-color: #f0f2f6;
        padding: 0.8em;
        border-left: 4px solid #1f77b4;
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
        background-color: #f8f9fa;
        border-radius: 5px 15px 15px 15px;
        padding: 10px 15px;
        border: 1px solid #e9ecef;
    }

    /* Compact dividers */
    hr {
        margin: 0.3rem 0;
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
def load_chat_history() -> List[Dict]:
    """Load chat history from database."""
    notebook_id = st.session_state.get("current_notebook_id")
    if notebook_id:
        return db.get_chat_history(notebook_id)
    return []


def save_chat_history():
    """Save chat history to database."""
    # We no longer save the entire history. Individual messages are saved directly
    # when they are added to chat. This function is kept for backward compatibility
    # but does nothing.
    pass


def load_saved_notes() -> List[Dict]:
    """Load saved notes from database."""
    notebook_id = st.session_state.get("current_notebook_id")
    if notebook_id:
        return db.get_notes_for_notebook(notebook_id)
    return []


def save_notes():
    """Save notes to database."""
    # We no longer dump all notes. Individual notes are added via db.add_note directly.
    pass


def load_documents_state() -> Dict:
    """Load documents metadata from database."""
    notebook_id = st.session_state.get("current_notebook_id")
    if not notebook_id:
        return {}

    sources = db.get_sources_for_notebook(notebook_id)
    docs_dict = {}
    for src in sources:
        docs_dict[src["file_name"]] = {
            "loaded": True,
            "summary": src["summary"],
            "suggested_questions": src["suggested_questions"],
            "id": src["id"],
        }
    return docs_dict


def save_documents_state():
    """Save documents metadata to disk."""
    # Docs are now saved directly via db.add_source
    pass


def get_notebook_vectorstore_dir(notebook_id: str) -> Path:
    """Get the path to the vectorstores directory for a specific notebook."""
    base_dir = DATA_DIR / "vectorstores"
    base_dir.mkdir(exist_ok=True)
    nb_dir = base_dir / f"nb_{notebook_id}"
    nb_dir.mkdir(exist_ok=True)
    return nb_dir


def get_source_vectorstore_dir(notebook_id: str, source_id: str) -> Path:
    """Get the path to the isolated vectorstore for a specific source."""
    nb_dir = get_notebook_vectorstore_dir(notebook_id)
    src_dir = nb_dir / f"src_{source_id}"
    src_dir.mkdir(exist_ok=True)
    return src_dir


def load_persisted_vectorstore(notebook_id: str):
    """Load the persisted vectorstores for all sources in the notebook."""
    sources = db.get_sources_for_notebook(notebook_id)
    merged_vs = None
    if not sources:
        return None

    embeddings = try_load_embeddings()
    if not embeddings:
        return None

    for source in sources:
        source_id = source["id"]
        vs_dir = get_source_vectorstore_dir(notebook_id, source_id)
        if (vs_dir / "index.faiss").exists():
            try:
                vs = FAISS.load_local(
                    str(vs_dir),
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
                if merged_vs is None:
                    merged_vs = vs
                else:
                    merged_vs.merge_from(vs)
            except Exception as e:
                logger.warning(
                    f"Failed to load vectorstore for source {source_id}: {e}"
                )
    return merged_vs


def save_vectorstore(notebook_id: str):
    # This is obsolete, replaced by saving individual source vectorstore
    pass


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
def load_workspace(notebook_id: str):
    """Load or reload all workspace state for a specific notebook."""
    st.session_state.current_notebook_id = notebook_id
    st.session_state.documents = load_documents_state()
    st.session_state.vectorstore = load_persisted_vectorstore(notebook_id)

    if st.session_state.vectorstore is not None:
        st.session_state.rag_chain = create_rag_chain(st.session_state.vectorstore)
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


def try_load_embeddings():
    """Try to load embeddings with GPU, fallback to CPU on OOM."""
    try:
        logger.info("Attempting to load embeddings on GPU")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_kwargs={"device": "cuda"},
        )
        return embeddings
    except Exception as e:
        logger.warning(f"GPU loading failed: {e}. Falling back to CPU...")
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                model_kwargs={"device": "cpu"},
            )
            return embeddings
        except Exception as e2:
            logger.error(f"CPU loading also failed: {e2}")
            return None


def generate_summary(rag_chain, document_name: str) -> str:
    """Generate a brief summary of the document."""
    try:
        summary_prompt = (
            f"Provide a brief 2-3 sentence summary of the main topics in this document."
        )
        summary = rag_chain.invoke(summary_prompt)
        return summary[:300]  # Limit length
    except Exception as e:
        logger.error(f"Summary generation failed: {str(e)}")
        return "Unable to generate summary."


def generate_suggested_questions(rag_chain, document_name: str) -> List[str]:
    """Generate 3-4 suggested questions based on document content."""
    try:
        question_prompt = (
            "Generate exactly 3 specific and interesting questions that a reader might ask about this document. "
            "Format as: 1. Question? 2. Question? 3. Question?"
        )
        response = rag_chain.invoke(question_prompt)

        questions = []
        for line in response.split("\n"):
            if line.strip() and (line[0].isdigit() or line.startswith("-")):
                question = line.lstrip("0123456789.-) ").strip()
                if question and len(question) > 5 and "?" in question:
                    questions.append(question)

        return questions[:4]
    except Exception as e:
        logger.error(f"Question generation failed: {str(e)}")
        return []


def process_pdf(uploaded_file, filename: str) -> bool:
    """Process a PDF: extract, chunk, embed, merge with existing data."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            if hasattr(uploaded_file, "getvalue"):
                tmp.write(uploaded_file.getvalue())
            elif hasattr(uploaded_file, "getbuffer"):
                tmp.write(uploaded_file.getbuffer())
            elif isinstance(uploaded_file, bytes):
                tmp.write(uploaded_file)
            else:
                tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner(f"Processing '{filename}'..."):
            # Load and chunk
            logger.info(f"Loading and chunking PDF: {filename}")
            chunks = load_and_chunk_pdf(tmp_path)
            num_chunks = len(chunks)

            for chunk in chunks:
                chunk.metadata["document"] = filename

            # Embed with GPU/CPU fallback
            logger.info(f"Creating embeddings for {num_chunks} chunks")
            embeddings = try_load_embeddings()
            if embeddings is None:
                st.error("Failed to load embedding model. Please check your system.")
                return False

            new_vectorstore = FAISS.from_documents(chunks, embeddings)

            # Merge or create
            if st.session_state.vectorstore is None:
                st.session_state.vectorstore = new_vectorstore
            else:
                st.session_state.vectorstore.merge_from(new_vectorstore)

            # Recreate RAG chain
            st.session_state.rag_chain = create_rag_chain(st.session_state.vectorstore)

            # Generate summary
            logger.info(f"Generating summary for {filename}")
            summary = generate_summary(st.session_state.rag_chain, filename)

            # Generate suggested questions
            logger.info(f"Generating suggested questions for {filename}")
            suggested_questions = generate_suggested_questions(
                st.session_state.rag_chain, filename
            )

            # Save to Database
            notebook_id = st.session_state.current_notebook_id

            # FIXME Need to get file path
            source_id = db.add_source(
                notebook_id=notebook_id,
                file_name=filename,
                file_path=filename,
                summary=summary,
                suggested_questions=suggested_questions,
            )

            # Update session state
            st.session_state.documents[filename] = {
                "loaded": True,
                "summary": summary,
                "suggested_questions": suggested_questions,
                "id": source_id,
            }

            # Persist state
            save_documents_state()

            # Save the individual source vectorstore
            src_vs_dir = get_source_vectorstore_dir(notebook_id, source_id)
            new_vectorstore.save_local(str(src_vs_dir))

            logger.info(f"Successfully processed {filename}")

        st.success(f"Successfully loaded '{filename}'")

        return True

    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}")
        st.error(f"Error processing file: {str(e)}")
        return False

    finally:
        # Guarantee tmp cleanup
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass


# ============================================================================
# SIDEBAR: SOURCE HUB
# ============================================================================
def go_back_to_notebooks():
    st.session_state.current_notebook_id = None


def source_hub_ui():
    """The 'Source Hub' for document management."""
    st.button(
        "← Back to Notebooks", use_container_width=True, on_click=go_back_to_notebooks
    )

    st.markdown("### Source Hub")
    st.divider()

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

    if uploaded_files:
        processed_any = False
        new_pending_replacements = False

        for uploaded_file in uploaded_files:
            filename = uploaded_file.name

            if filename in st.session_state.documents:
                # Store in memory to ask user, if not already asking
                if filename not in st.session_state.pending_replacements:
                    st.session_state.pending_replacements[filename] = (
                        uploaded_file.getvalue()
                    )
                    new_pending_replacements = True
            else:
                # Process right away
                if process_pdf(uploaded_file, filename):
                    processed_any = True

        # Clear the uploader after reading the batch
        st.session_state.file_uploader_key += 1

        if processed_any or new_pending_replacements:
            st.rerun()

    # Show pending replacements if any are saved in session state
    if getattr(st.session_state, "pending_replacements", None):
        st.markdown("##### Pending Duplicate Files")
        # Copy keys to a list to safely iterate
        pending_files = list(st.session_state.pending_replacements.keys())

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
                    file_bytes = st.session_state.pending_replacements[filename]

                    # Remove the file from pending immediately so it doesn't get processed again
                    del st.session_state.pending_replacements[filename]

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

                    if filename in st.session_state.documents:
                        del st.session_state.documents[filename]
                        save_documents_state()

                    # Reload updated vectorstore without the deleted source
                    st.session_state.vectorstore = load_persisted_vectorstore(
                        st.session_state.current_notebook_id
                    )
                    if st.session_state.vectorstore is not None:
                        st.session_state.rag_chain = create_rag_chain(
                            st.session_state.vectorstore
                        )
                    else:
                        st.session_state.rag_chain = None

                    # Phase 2: Insert the new one
                    process_pdf(file_bytes, filename)

                    status_placeholder.empty()
                    st.rerun()

                if cancel_clicked:
                    del st.session_state.pending_replacements[filename]
                    st.rerun()

    st.divider()
    st.markdown("#### Loaded Documents")

    if st.session_state.documents:
        for doc_name, doc_info in list(st.session_state.documents.items()):
            with st.expander(f"{doc_name}", expanded=False):
                if doc_info.get("summary"):
                    st.caption(f"**Summary:** {doc_info['summary']}")

                if st.button(
                    "Remove", key=f"remove_{doc_name}", use_container_width=True
                ):
                    logger.info(f"Removing document: {doc_name}")
                    if "id" in doc_info:
                        db.delete_source(doc_info["id"])
                        source_dir = get_source_vectorstore_dir(
                            st.session_state.current_notebook_id, doc_info["id"]
                        )
                        if source_dir.exists():
                            import shutil

                            shutil.rmtree(source_dir, ignore_errors=True)

                    del st.session_state.documents[doc_name]
                    save_documents_state()

                    # Reload vectorstore skipping the deleted source
                    st.session_state.vectorstore = load_persisted_vectorstore(
                        st.session_state.current_notebook_id
                    )
                    if st.session_state.vectorstore is not None:
                        st.session_state.rag_chain = create_rag_chain(
                            st.session_state.vectorstore
                        )
                    else:
                        st.session_state.rag_chain = None

                    st.rerun()
    else:
        st.info("No documents loaded yet.")

    st.divider()
    st.markdown("#### System Status")

    if st.session_state.ollama_ready:
        st.success("Ollama: Connected")
    else:
        st.error("Ollama: Offline")
        st.caption("Run: ollama serve")


# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================
def render_user_message(content: str):
    """Renders a chat message from the user aligned to the right."""
    escaped_content = html.escape(content).replace("\n", "<br>")
    st.markdown(
        f"""
        <div style="display: flex; justify-content: flex-end; margin-bottom: 0.5rem; width: 100%;">
            <div style="background-color: #e3f2fd; color: #000; padding: 0.6rem 1rem; border-radius: 15px 5px 15px 15px; border: 1px solid #cce5ff; max-width: 80%; text-align: left; box-shadow: 0 1px 2px rgba(0,0,0,0.05); font-family: sans-serif; font-size: 0.95em;">
                {escaped_content}
            </div>
            <div style="width: 35px; height: 35px; background-color: #0d6efd; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-left: 10px; color: white; font-weight: bold; flex-shrink: 0; font-size: 0.9em; margin-top: 2px;">
                U
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chat_interface(notebook_name: str):
    """Main chat area - NotebookLM style."""
    st.markdown(
        f"<h1 class='main-header'>SmartDoc AI <span style='font-size: 0.5em; color: gray;'>/ {notebook_name}</span></h1>",
        unsafe_allow_html=True,
    )
    st.caption("Intelligent document Q&A with source citations")
    st.divider()

    if not st.session_state.documents:
        st.info("Upload documents in the Source Hub to get started.")
        return

    # Create a container for chat messages
    chat_container = st.container()

    # Chat input at the bottom
    st.divider()
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
                    st.divider()
        else:
            suggestions_placeholder.empty()

        # Display chat history (proper left/right layout)
        for message in st.session_state.chat_history:
            if message["role"] == USER_ROLE_NAME:
                render_user_message(message["content"])
            else:
                with st.chat_message(ASSISTANT_ROLE_NAME, avatar="🤖"):
                    st.markdown(message["content"])

                    if message.get("sources"):
                        with st.expander("📎 View sources", expanded=False):
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
                with st.chat_message(ASSISTANT_ROLE_NAME, avatar="🤖"):
                    with st.spinner("🤔 Thinking..."):
                        try:
                            # Get answer
                            logger.info(f"Processing query: {user_query[:50]}...")
                            answer = st.session_state.rag_chain.invoke(user_query)
                            st.markdown(answer)

                            # Retrieve sources directly from vectorstore
                            source_docs = (
                                st.session_state.vectorstore.similarity_search(
                                    user_query, k=5
                                )
                            )

                            sources = []
                            for doc in source_docs:
                                sources.append(
                                    {
                                        "document": doc.metadata.get(
                                            "document", "Unknown"
                                        ),
                                        "page": doc.metadata.get("page", "N/A"),
                                        "content": (
                                            doc.page_content[:200] + "..."
                                            if len(doc.page_content) > 200
                                            else doc.page_content
                                        ),
                                    }
                                )

                            # Add to history
                            db.add_chat_message(
                                notebook_id=st.session_state.current_notebook_id,
                                role=ASSISTANT_ROLE_NAME,
                                content=answer,
                                sources=sources,
                            )
                            st.session_state.chat_history.append(
                                {
                                    "role": ASSISTANT_ROLE_NAME,
                                    "content": answer,
                                    "sources": sources,
                                }
                            )

                            # Display sources
                            if sources:
                                with st.expander("📎 View sources", expanded=False):
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
                                    note_id = db.add_note(
                                        notebook_id=st.session_state.current_notebook_id,
                                        title=user_query,
                                        content=answer,
                                    )
                                    # Refresh notes from DB
                                    st.session_state.saved_notes = (
                                        db.get_notes_for_notebook(
                                            st.session_state.current_notebook_id
                                        )
                                    )
                                    st.success("✅ Saved to study notes")

                        except Exception as e:
                            logger.error(f"Error: {str(e)}")
                            st.error(f"❌ Error: {str(e)}")

        # Rerun to refresh chat display
        st.rerun()

    # Study notes section at the bottom
    if st.session_state.saved_notes:
        st.divider()
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


# ============================================================================
# NOTEBOOK DASHBOARD
# ============================================================================
def render_dashboard():
    """Renders the grid of notebooks to select or create."""
    header_col1, header_col2 = st.columns([4, 1], vertical_alignment="bottom")
    with header_col1:
        st.markdown("<h1 class='main-header'>SmartDoc AI</h1>", unsafe_allow_html=True)
        st.caption("Your personalized NotebookLM-inspired AI assistant")
    with header_col2:
        if st.button("Create New Notebook", type="primary", use_container_width=True):
            create_notebook_modal()

    st.divider()

    st.subheader("Your Notebooks")

    # Reload notebooks to ensure we have the latest
    notebooks = db.get_all_notebooks()

    if not notebooks:
        st.info(
            "Welcome to SmartDoc AI! Create your first notebook below to get started."
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
                    st.markdown(
                        f"<small>Created: {nb['created_at'][:10]}</small>",
                        unsafe_allow_html=True,
                    )

                    c1, c2 = st.columns([3, 1])
                    with c1:
                        if st.button(
                            "Open",
                            key=f"open_{nb['id']}",
                            use_container_width=True,
                            type="primary",
                        ):
                            st.session_state.loading_notebook_id = nb["id"]
                            st.rerun()
                    with c2:
                        st.button(
                            "🗑️",
                            key=f"delete_{nb['id']}",
                            use_container_width=True,
                            on_click=delete_notebook_callback,
                            args=(nb["id"],),
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
            load_workspace(st.session_state.loading_notebook_id)
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
            source_hub_ui()

        with col2:
            chat_interface(notebook_name)


if __name__ == "__main__":
    main()
