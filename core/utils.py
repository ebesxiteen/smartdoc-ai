"""
Centralized utility functions for RAG application.
Includes: PDF processing, chat handling, vectorstore management, and text cleaning.
"""

from datetime import datetime, timezone
import hashlib
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
import logging

import streamlit as st
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import core.configs as cfg
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
import re
from langdetect import (  # pyright: ignore[reportMissingTypeStubs]
    detect,  # pyright: ignore[reportUnknownVariableType]
)
import psutil
import platform

logger = logging.getLogger(__name__)


# ============================================================================
# HARDWARE DETECTION UTILITY
# ============================================================================


@st.cache_data(show_spinner=False)
def get_system_hardware_info() -> Dict[str, Any]:
    """Gather system hardware capabilities for dynamic UI warnings."""
    # OS & CPU & RAM
    os_name = platform.system()
    cpu_cores = psutil.cpu_count(logical=True)
    total_ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)

    # GPU
    gpu_name = "None/Integrated"
    total_vram_gb = 0.0

    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            # torch limits might not reflect exact system VRAM,
            # but usually get_device_properties gives total_memory in bytes
            props: Any = torch.cuda.get_device_properties(0)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            total_memory_bytes: int = getattr(props, "total_memory", 0)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
            total_vram_gb = round(total_memory_bytes / (1024**3), 1)
    except Exception as e:
        logger.debug(f"Failed to detect GPU via torch: {e}")

    return {
        "os": os_name,
        "cpu_cores": cpu_cores,
        "ram_gb": total_ram_gb,
        "gpu_name": gpu_name,
        "vram_gb": total_vram_gb,
    }


def get_default_notebook_settings() -> Dict[str, Any]:
    return {
        "rag_retrieval_k": cfg.RAG_RETRIEVAL_K,
        "rag_retrieval_min_results": cfg.RAG_RETRIEVAL_MIN_RESULTS,
        "rag_retrieval_score_threshold": cfg.RAG_RETRIEVAL_SCORE_THRESHOLD,
        "rag_max_chunk_len": cfg.RAG_MAX_CHUNK_LEN,
        "rag_chunk_overlap": cfg.RAG_CHUNK_OVERLAP,
        "rag_max_ctx_len": cfg.RAG_MAX_CTX_LEN,
        "max_msg_history": cfg.MAX_MSG_HISTORY,
        "llm_model_name": cfg.LLM_MODEL_NAME,
        "llm_num_ctx": cfg.LLM_NUM_CTX,
        "llm_temp": cfg.LLM_TEMPERATURE,
        "personal_ctx": None,
    }


def _load_notebook_settings(
    notebook_id: Optional[str],
) -> Dict[str, Any]:
    defaults = get_default_notebook_settings()
    if not notebook_id:
        return defaults

    from middlewares.db_middleware import get_notebook_settings

    settings = get_notebook_settings(notebook_id)
    if not settings:
        return defaults

    return {
        "rag_retrieval_k": settings.get("rag_retrieval_k"),
        "rag_retrieval_min_results": settings.get("rag_retrieval_min_results"),
        "rag_retrieval_score_threshold": settings.get("rag_retrieval_score_threshold"),
        "rag_max_chunk_len": settings.get("rag_max_chunk_len"),
        "rag_chunk_overlap": settings.get("rag_chunk_overlap"),
        "rag_max_ctx_len": settings.get("rag_max_ctx_len"),
        "max_msg_history": settings.get("max_msg_history"),
        "llm_model_name": settings.get("llm_model_name"),
        "llm_num_ctx": settings.get("llm_num_ctx"),
        "llm_temp": settings.get("llm_temp"),
        "personal_ctx": settings.get("personal_ctx"),
    }


# ============================================================================
# TEXT UTILITIES
# ============================================================================


def clean_spaces(text: str) -> str:
    """
    Normalizes whitespace: removes leading/trailing spaces and
    collapses multiple internal spaces into a single one.
    """
    if not text:
        return ""
    # .split() without arguments splits by any whitespace (space, \n, \t)
    # ' '.join(...) puts exactly one space between each word
    return " ".join(text.split())


# ============================================================================
# PDF PROCESSING UTILITIES
# ============================================================================


def load_and_chunk_file(
    file_path: str,
    file_type: str,
    chunk_size: int = cfg.RAG_MAX_CHUNK_LEN,
    chunk_overlap: int = cfg.RAG_CHUNK_OVERLAP,
    print_debug: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
):
    """
    Load a file (PDF or DOCX) and chunk it into overlapping segments.

    Args:
      file_path: Path to the file
      file_type: Type of the file ('pdf' or 'docx')

    Returns:
      List of Document objects with chunked text
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Step 1: Load the Document
    documents: List[Document] = []
    if file_type == "pdf":
        from langchain_community.document_loaders import PyMuPDFLoader

        loader = PyMuPDFLoader(file_path)
        for i, page in enumerate(loader.lazy_load()):
            documents.append(page)
            if progress_callback:
                progress_callback(f"Reading page {i + 1}...")
    elif file_type == "docx":
        import docx

        if progress_callback:
            progress_callback("Reading DOCX file...")
        doc = docx.Document(file_path)
        full_text: List[str] = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        full_text.append(cell.text)
        text = "\n".join(full_text)
        if text.strip():
            documents = [Document(page_content=text, metadata={"source": file_path})]
    else:
        raise ValueError(f"Unsupported file type for loading: {file_type}")

    if print_debug:
        print(f"✅ Loaded {file_type.upper()}: {file_path}")
        print(f"   Total documents/pages: {len(documents)}")
        print(f"   Total characters: {sum(len(d.page_content) for d in documents)}")

    # Step 2: Split into chunks
    if progress_callback:
        progress_callback("Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            " ",
            "",
        ],  # Split by paragraph, then line, then space, then character
    )

    chunks = text_splitter.split_documents(documents)

    if print_debug:
        print("\n✅ Chunking complete!")
        print(f"   Total chunks: {len(chunks)}")
        print(
            f"   Average chunk size: {sum(len(chunk.page_content) for chunk in chunks) / len(chunks):.0f} characters"
        )

    return chunks


def hash_file_content(file_bytes: bytes) -> str:
    """Calculate MD5 hash of file content."""
    return hashlib.md5(file_bytes).hexdigest()


def detect_file_type(file_bytes: bytes) -> str:
    """Detect file type based on magic numbers."""
    if file_bytes.startswith(b"%PDF"):
        return "pdf"
    elif (
        file_bytes.startswith(b"PK\x03\x04")
        or file_bytes.startswith(b"PK\x05\x06")
        or file_bytes.startswith(b"PK\x07\x08")
    ):
        return "docx"
    else:
        raise ValueError(
            "Unsupported file format. Only PDF and Word (docx) are supported."
        )


def check_file_already_exists(file_hash: str) -> Optional[Dict[str, Any]]:
    """Check if a file with this hash was already uploaded globally."""
    from db.crud import get_source_by_hash

    return get_source_by_hash(file_hash)


def check_file_already_exists_in_notebook(
    file_hash: str, notebook_id: str
) -> Optional[Dict[str, Any]]:
    """Check if a file with this hash already exists in THIS specific notebook."""
    from db.crud import get_source_by_hash_and_notebook

    return get_source_by_hash_and_notebook(file_hash, notebook_id)


def chunk_and_process_file(
    file_path: str,
    file_type: str,
    filename: str,
    print_debug: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
    chunk_size: int = cfg.RAG_MAX_CHUNK_LEN,
    chunk_overlap: int = cfg.RAG_CHUNK_OVERLAP,
) -> Tuple[List[Document], int]:
    """Load File, chunk it, and add metadata."""
    if print_debug:
        logger.info(f"Loading and chunking {file_type.upper()}: {filename}")
    chunks = load_and_chunk_file(
        file_path,
        file_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        print_debug=print_debug,
        progress_callback=progress_callback,
    )

    # Add document name to metadata
    for chunk in chunks:
        chunk.metadata["document"] = filename  # type: ignore[index]

    return chunks, len(chunks)


def create_vectorstore_from_chunks(
    chunks: List[Document], embeddings: HuggingFaceEmbeddings, print_debug: bool = False
) -> FAISS:
    """Create FAISS vectorstore from document chunks."""
    if print_debug:
        logger.info(f"Creating vectorstore from {len(chunks)} chunks")
    return FAISS.from_documents(chunks, embeddings)


def merge_vectorstores(
    existing_vectorstore: Optional[FAISS],
    new_vectorstore: FAISS,
    print_debug: bool = False,
) -> FAISS:
    """Merge new vectorstore into existing one, or return new if none exists."""
    if existing_vectorstore is None:
        if print_debug:
            logger.info("No existing vectorstore, using new one")
        return new_vectorstore

    if print_debug:
        logger.info("Merging vectorstores")
    existing_vectorstore.merge_from(new_vectorstore)
    return existing_vectorstore


def save_source_to_database(
    notebook_id: str,
    filename: str,
    file_type: str,
    file_hash: str,
    summary: str,
    suggested_questions: List[str],
    vectorstore_path: str,
    source_id: Optional[str] = None,
    print_debug: bool = False,
) -> str:
    """Save source metadata and vectorstore path to database.

    Args:
        source_id: Pre-generated UUID (pass when you need it upfront for path calculation).
                   If None, a new UUID is generated internally.
    """
    from middlewares import db_middleware as db

    if print_debug:
        logger.info(f"Saving source to database: {filename}")

    if source_id is None:
        source_id = str(uuid.uuid4())

    saved_source_id = db.add_source(
        notebook_id=notebook_id,
        file_name=filename,
        file_type=file_type,
        file_path=vectorstore_path,
        file_hash=file_hash,
        summary=summary,
        suggested_questions=suggested_questions,
        source_id=source_id,
    )

    return saved_source_id


# ============================================================================
# CHAT PROCESSING UTILITIES
# ============================================================================


def process_user_query(
    query: str,
    rag_chain: Any,
    vectorstore: FAISS,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    print_debug: bool = False,
    notebook_id: Optional[str] = None,
) -> tuple[str, List[Dict[str, Any]], bool]:
    """
    Process a user query through the RAG chain with optional chat history context.

    Handles three scenarios:
    1. Greeting/general questions: Answer without searching documents
    2. Follow-up questions: Use chat history to rephrase for accurate retrieval
    3. New questions: Direct document search and answer

    Returns:
        (answer, sources_list, found_answer)
        - answer: The LLM's response (with [STATUS: ...] tag removed)
        - sources_list: Top sources for display
        - found_answer: Boolean indicating if LLM found useful context (defaults to True if not specified)
    """
    if print_debug:
        print("\n")
        logger.info(
            f"🔡\tProcessing query: {query[:60]}..."
            if len(query) > 60
            else f"🔡\tProcessing query: {query}"
        )

    # Step 1: Check if this is a greeting/general question
    is_greeting_query = is_greeting(query)
    if print_debug and is_greeting_query:
        logger.info(
            "👥💬\tDetected greeting/general question - using general knowledge"
        )

    settings = _load_notebook_settings(notebook_id)

    # Step 2: Prepare inputs for the RAG chain
    # Build chain input dict with proper structure for history-aware chain
    # Using a list for retrieved_docs so we can safely extract it back from the chain
    retrieved_docs: List[Document] = []
    chain_input_dict: Dict[str, Any] = {
        "input": query,
        "question": query,
        "__retrieved_docs__": retrieved_docs,
        "is_greeting": is_greeting_query,
    }

    if chat_history:
        formatted_history = format_chat_history_for_rephrase(
            chat_history, max_messages=int(settings["max_msg_history"])
        )
        chain_input_dict["chat_history"] = formatted_history
        if print_debug:
            logger.info("   " + "━" * 60)
            logger.info(
                f"   📜\tIncluding {len(formatted_history)} history messages for context"
            )
            for msg in formatted_history:
                role_icon = "👤" if msg.type == "human" else "🤖"
                content_str = str(msg.content)  # type: ignore
                content_preview = content_str.replace("\n", " ")
                content_preview = (
                    content_preview[:70] + "..."
                    if len(content_preview) > 70
                    else content_preview
                )
                logger.info(f"      {role_icon}\t{content_preview}")
            logger.info("   " + "━" * 60)

    # Step 3: Generate answer through RAG chain
    if print_debug:
        if is_greeting_query:
            logger.info("   ⏳\tInvoking RAG chain with general knowledge mode...")
        else:
            logger.info("   ⏳\tInvoking RAG chain with quality-filtered chunks...")

    try:
        # Invoke chain with proper input structure
        # The history-aware chain expects dict with "input" and optional "chat_history"
        answer = rag_chain.invoke(chain_input_dict)
    except Exception as e:
        logger.error(f"Error invoking RAG chain: {e}")
        if print_debug:
            logger.error(f"    Full error: {str(e)}")
        answer = f"[STATUS: DOC_MISSING]\nI encountered an error processing your query: {str(e)}"

    if print_debug:
        logger.info(f"   💬\tLLM response generated ({len(answer)} chars)")

        # Log the actual answer cleanly
        logger.info("   " + "━" * 60)
        logger.info("   🤖\tLLM Answer:")

        # Split by newlines so it aligns cleanly in the terminal
        for idx, line in enumerate(answer.split("\n")):
            if line.strip():
                # To prevent overwhelming logs, limit to first 10 lines max or 1000 chars
                if idx > 9 or len(line) > 150:
                    preview = line[:150] + "..." if len(line) > 150 else line
                    logger.info(f"      {preview}")
                else:
                    logger.info(f"      {line}")
        logger.info("   " + "━" * 60)

    # Step 4: Parse [STATUS: DOC_ANSWER/DOC_MISSING/GENERAL] tag from answer
    found_answer = True  # Default to True
    answer_clean = answer
    is_general_answer = is_greeting_query

    # Check for tags using robust regex
    if re.search(r"\[STATUS:\s*DOC_ANSWER\]", answer, flags=re.IGNORECASE):
        found_answer = True
    elif re.search(r"\[STATUS:\s*DOC_MISSING\]", answer, flags=re.IGNORECASE):
        found_answer = False
        is_general_answer = False
    elif re.search(r"\[STATUS:\s*GENERAL\]", answer, flags=re.IGNORECASE):
        found_answer = False
        is_general_answer = True

    # Strip out any tags and clean up
    answer_clean = re.sub(
        r"\[STATUS:\s*(?:DOC_ANSWER|DOC_MISSING|GENERAL)\]",
        "",
        answer,
        flags=re.IGNORECASE,
    ).strip()

    # Prevent 'Content cannot be empty' DB error if LLM only returned a tag in rare cases
    if not answer_clean:
        answer_clean = cfg.NOT_FOUND_ANSWER_FALL_BACK

    if print_debug:
        if found_answer and not is_general_answer:
            logger.info("   📍\tLLM found relevant context - sources will be displayed")
        elif is_general_answer:
            logger.info(
                "   💬\tLLM answered from general knowledge - no sources needed"
            )
        else:
            logger.warning(
                "   ⚠️\tLLM did not find relevant context - sources will be hidden"
            )

    # Step 5: Retrieve source documents for display (only if not a general question)
    sources: List[Dict[str, Any]] = []

    if not is_general_answer and print_debug:
        logger.info("   📎\tGathering exact retrieved sources for display...")

    if not is_general_answer:
        # Use the exact documents retrieved during the RAG chain invoke
        source_docs = chain_input_dict.get("__retrieved_docs__", [])

        # Fallback just in case they weren't saved
        if not source_docs:
            source_docs = retrieve_quality_chunks(
                vectorstore,
                query,
                k=settings["rag_retrieval_k"],
                min_results=settings["rag_retrieval_min_results"],
                score_threshold=settings["rag_retrieval_score_threshold"],
                print_debug=print_debug,
            )

        # Format sources for display
        for i, doc in enumerate(source_docs, 1):
            doc_metadata: Dict[str, Any] = doc.metadata  # type: ignore[assignment]
            source_entry: Dict[str, Any] = {
                "document": str(doc_metadata.get("document", "Unknown")),
                "page": str(doc_metadata.get("page", "N/A")),
                "content": (
                    doc.page_content[:200] + "..."
                    if len(doc.page_content) > 200
                    else doc.page_content
                ),
            }
            sources.append(source_entry)
            if print_debug:
                logger.info(
                    f"      • Source {i}: {source_entry['document']} (Page {source_entry['page']})"
                )

        if print_debug:
            logger.info("   " + "━" * 60)
            logger.info(f"   ✅\tProcessed {len(sources)} display sources for UI")

    return answer_clean, sources, found_answer


def save_query_and_answer_to_history(
    notebook_id: str,
    query: str,
    answer: str,
    sources: List[Dict[str, Any]],
    print_debug: bool = False,
) -> None:
    """Save user query and assistant answer to chat history database."""
    from middlewares import db_middleware as db

    if print_debug:
        logger.info(f"Saving query/answer to history for notebook {notebook_id}")

    # Save user message
    db.add_chat_message(
        notebook_id=notebook_id,
        role=cfg.USER_ROLE_NAME,
        content=query,
    )

    # Save assistant message with sources
    db.add_chat_message(
        notebook_id=notebook_id,
        role=cfg.ASSISTANT_ROLE_NAME,
        content=answer,
        sources=sources,
    )


def save_answer_as_note(
    notebook_id: str,
    query: str,
    answer: str,
    print_debug: bool = False,
) -> None:
    """Save an answer as a study note in the notebook."""
    from middlewares import db_middleware as db

    if print_debug:
        logger.info(f"Saving answer as note in notebook {notebook_id}")

    db.add_note(
        notebook_id=notebook_id,
        title=query,
        content=answer,
    )


# ============================================================================
# VECTORSTORE PERSISTENCE UTILITIES
# ============================================================================


def get_notebook_vectorstore_dir(notebook_id: str) -> Path:
    """Get the path to the vectorstores directory for a specific notebook."""
    base_dir = Path("data") / "vectorstores"
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


@st.cache_resource(show_spinner=False)
def try_load_embeddings() -> Optional[HuggingFaceEmbeddings]:
    """Try to load embeddings with GPU, fallback to CPU on OOM."""
    try:
        logger.info("🔳\tAttempting to load embeddings on GPU")
        return HuggingFaceEmbeddings(
            model_name=cfg.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cuda"},
        )
    except Exception as e:
        logger.warning(f"GPU loading failed: {e}. Falling back to CPU...")
        try:
            return HuggingFaceEmbeddings(
                model_name=cfg.EMBEDDING_MODEL_NAME,
                model_kwargs={"device": "cpu"},
            )
        except Exception as e2:
            logger.error(f"CPU loading also failed: {e2}")
            return None


def retrieve_quality_chunks(
    vectorstore: FAISS,
    query: str,
    k: int = cfg.RAG_RETRIEVAL_K,
    min_results: int = cfg.RAG_RETRIEVAL_MIN_RESULTS,
    score_threshold: float = cfg.RAG_RETRIEVAL_SCORE_THRESHOLD,
    print_debug: bool = False,
) -> List[Document]:
    """
    Retrieve chunks with intelligent quality-based filtering.

    Uses similarity score threshold filtering first, then falls back to top K
    results if threshold filtering doesn't return enough chunks.
    This prevents missing relevant context while maintaining quality.

    Args:
      vectorstore: FAISS vectorstore object
      query: User's question

    Returns:
      List of Document objects ranked by relevance
    """
    if print_debug:
        logger.info(f"🔎 Retrieving quality chunks from {k} total results")
        logger.info(
            f"   Quality threshold: {score_threshold}, Min fallback: {min_results}"
        )

    # Step 1: Get top K results with scores
    results_with_scores: List[Any] = vectorstore.similarity_search_with_score(  # pyright: ignore[reportUnknownMemberType]
        query, k=k
    )

    if print_debug:
        logger.debug(f"   Total search results: {len(results_with_scores)}")

    # Step 2: Filter by quality threshold (lower distance = better match)
    filtered_results: List[Document] = []
    passed_threshold = 0
    failed_threshold = 0

    for doc, score in results_with_scores:
        # Store similarity score in metadata for reference
        doc.metadata["similarity_score"] = score

        # Apply quality threshold: only include chunks with good similarity
        if score <= score_threshold:
            filtered_results.append(doc)
            doc.metadata["passed_quality_threshold"] = True
            passed_threshold += 1
        else:
            failed_threshold += 1

    if print_debug:
        logger.info(
            f"   ✅\tPassed threshold ({score_threshold}): {passed_threshold} chunks"
        )
        logger.info(f"   ❌\tFailed threshold: {failed_threshold} chunks")

    # Step 3: Fallback mechanism - if threshold filtered too much, use top min_results
    if len(filtered_results) < min_results:
        if print_debug:
            logger.warning(
                f"   ⚠️\tOnly {len(filtered_results)} chunks passed threshold. "
                f"Falling back to top {min_results} results..."
            )

        filtered_results = []
        for doc, score in results_with_scores[:min_results]:
            doc.metadata["similarity_score"] = score
            doc.metadata["passed_quality_threshold"] = False
            filtered_results.append(doc)

    if print_debug:
        logger.info(f"   ✅\tFinal selection: {len(filtered_results)} chunks")
        logger.info("   " + "━" * 60)

        for i, doc in enumerate(filtered_results, 1):
            doc_metadata: Dict[str, Any] = doc.metadata  # type: ignore[assignment]
            score = doc_metadata.get("similarity_score", "N/A")

            # Extract clean filename
            doc_name = doc_metadata.get(
                "source", doc_metadata.get("document", "Unknown")
            )
            if "/" in str(doc_name) or "\\" in str(doc_name):
                doc_name = Path(doc_name).name

            page_num = doc_metadata.get("page", "N/A")
            content_preview = doc.page_content[:85].replace("\n", " ").strip()
            score_str = f"{score:.4f}" if isinstance(score, float) else str(score)

            logger.info(
                f"   📄\t[{i}] L2 Distance: {score_str} | {doc_name} (Page {page_num})"
            )
            logger.info(f'       "{content_preview}..."')

        logger.info("   " + "━" * 60 + "\n")

    return filtered_results


def format_context_with_sources(
    docs: List[Document],
    max_chunk_length: int = cfg.RAG_MAX_CHUNK_LEN,
    print_debug: bool = False,
) -> str:
    """
    Format retrieved documents into a context string with source citations.
    Intelligently limits context to prevent GPU OOM during LLM inference.

    Args:
      docs: List of Document objects (with optional similarity_score metadata)

    Returns:
      Formatted context string with page numbers and citations
    """
    if not docs:
        if print_debug:
            logger.debug("📄 No relevant context found to format.")
        return "No relevant context found."

    context_parts: List[str] = []
    current_length = 0

    if print_debug:
        logger.debug(f"📄 Formatting {len(docs)} chunks into context...")

    for idx, doc in enumerate(docs, 1):
        metadata: dict[str, Any] = getattr(doc, "metadata", {})
        page_num = str(metadata.get("page", "Unknown"))
        # Clean up the text: remove extra whitespace
        text = " ".join(doc.page_content.split())

        # Limit individual chunk to configured max length to keep relevant info focused
        if len(text) > max_chunk_length:
            text = text[:max_chunk_length] + "..."

        part = f"[Source: Page {page_num}]\n{text}"
        part_length = len(part)

        # Stop adding chunks if we exceed max total context length
        if current_length + part_length > cfg.RAG_MAX_CTX_LEN:
            if print_debug:
                logger.debug(
                    f"   ⚠️  Max context length reached. Stopping at chunk {idx - 1}. "
                    f"(Current: {current_length}B, Limit: {cfg.RAG_MAX_CTX_LEN}B)"
                )
            break

        context_parts.append(part)
        current_length += part_length

        if print_debug:
            logger.debug(
                f"   📑 Chunk {idx}: Page {page_num} ({part_length}B, running total: {current_length}B)"
            )

    final_context = "\n\n".join(context_parts)
    if print_debug:
        logger.debug(f"   ✅ Final context size: {len(final_context)}B")

    return final_context


def reload_vectorstore_and_chain(
    notebook_id: str,
    selected_sources: Set[str],
    print_debug: bool = False,
) -> None:
    """Reload vectorstore and RAG chain based on currently selected sources."""
    st.session_state.vectorstore = load_persisted_vectorstore_filtered(
        notebook_id, selected_sources
    )

    if st.session_state.vectorstore is not None:
        st.session_state.rag_chain = create_history_aware_rag_chain(
            st.session_state.vectorstore,
            print_debug=print_debug,
            notebook_id=notebook_id,
        )
    else:
        st.session_state.rag_chain = None


def load_persisted_vectorstore_filtered(
    notebook_id: str, selected_source_ids: Set[str]
) -> Optional[FAISS]:
    """Load and merge only the selected sources' vectorstores.

    Args:
        notebook_id: The notebook ID
        selected_source_ids: Set of source IDs to load

    Returns:
        Merged FAISS vectorstore or None if no valid sources selected
    """
    from middlewares import db_middleware as db

    if not selected_source_ids:
        return None

    sources = db.get_sources_for_notebook(notebook_id)
    merged_vs = None

    embeddings = try_load_embeddings()
    if not embeddings:
        return None

    for source in sources:
        source_id = source["id"]
        # Only load if source is in selected set
        if source_id not in selected_source_ids:
            continue

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


def format_relative_time(dt_str: str) -> str:
    """Format a DB datetime string as a human-readable relative time."""
    try:
        dt = datetime.strptime(dt_str[:19], "%Y-%m-%d %H:%M:%S")
        now_utc = datetime.now(timezone.utc)

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        secs = int((now_utc - dt).total_seconds())
        if secs < 60:
            return f"{secs}s ago"
        if secs < 3600:
            return f"{secs // 60}m ago"
        if secs < 86400:
            return f"{secs // 3600}h ago"
        return f"{secs // 86400}d ago"
    except Exception:
        return dt_str[:10]


# ============================================================================
# ADVANCED RAG: CHAT HISTORY & GREETING DETECTION
# ============================================================================


def is_greeting(user_query: str) -> bool:
    """
    Detect if user query is a greeting or general question across multiple languages.

    Supports: English, Vietnamese, and Mandarin Chinese with automatic language detection.
    Falls back to English patterns if language detection fails.

    Args:
        user_query: The user's input text

    Returns:
        True if query is a greeting/general question, False if it needs document search
    """
    query_lower = clean_spaces(user_query.lower())

    try:
        # Detect language
        detected_lang: str = str(detect(query_lower))  # type: ignore[misc]

        # Map detected language codes to our supported languages
        # langdetect uses ISO 639-1 codes: 'en', 'vi', 'zh-cn', 'zh-tw', etc.
        if detected_lang.startswith("zh"):
            lang_code = "zh"  # Both Simplified and Traditional Chinese
        elif detected_lang in ("en", "vi"):
            lang_code = detected_lang
        else:
            # Unsupported language, fall back to English patterns
            lang_code = "en"

        # Get language-specific patterns
        patterns = cfg.GREETING_PATTERNS_BY_LANGUAGE.get(
            lang_code, cfg.DEFAULT_EN_GREETING_PATTERNS
        )

        for pattern in patterns:
            if re.match(pattern, query_lower):
                return True

        return False

    except Exception as e:
        # If language detection fails, fall back to English patterns
        logger.debug(
            f"Language detection failed: {e}. Using English patterns as fallback."
        )
        for pattern in cfg.DEFAULT_EN_GREETING_PATTERNS:
            if re.match(pattern, query_lower):
                return True
        return False


def format_chat_history_for_rephrase(
    chat_history: List[Dict[str, Any]], max_messages: int = cfg.MAX_MSG_HISTORY
) -> List[BaseMessage]:
    """
    Convert database chat history to LangChain message format for history-aware retriever.

    Takes the latest N messages from chat history and converts them to LangChain's
    HumanMessage/AIMessage format for use in history-aware retrieval chains.

    Args:
        chat_history: List of message dicts from database (role, content, etc.)
        max_messages: Maximum number of messages to include (use latest N messages)

    Returns:
        List of LangChain BaseMessage objects (HumanMessage or AIMessage)
    """
    messages: List[BaseMessage] = []

    # Use only the latest max_messages to avoid context overflow
    for msg in chat_history[-max_messages:]:
        if msg["role"] == cfg.USER_ROLE_NAME:
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == cfg.ASSISTANT_ROLE_NAME:
            messages.append(AIMessage(content=msg["content"]))

    return messages


def create_history_aware_rag_chain(
    vectorstore: FAISS,
    print_debug: bool = False,
    notebook_id: Optional[str] = None,
) -> Any:
    """
    Build a history-aware RAG chain that can handle follow-up questions.

    This chain:
    1. Takes chat history and current question
    2. Rephrases the question for standalone retrieval
    3. Retrieves context from vectorstore
    4. Generates answer with document context + general knowledge fallback

    Args:
        vectorstore: FAISS vectorstore object
        print_debug: Enable debug logging
        notebook_id: The ID of the notebook for loading settings

    Returns:
        Runnable chain that accepts {"input": question} and optional {"chat_history": [...]}
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_ollama import OllamaLLM
    from langchain_core.prompts import MessagesPlaceholder

    if print_debug:
        logger.info("\n\n🛠️\tBuilding History-Aware RAG Chain...")

    settings = _load_notebook_settings(notebook_id)

    # Step 1: Create quality-based retriever
    def quality_retriever(query: str) -> List[Document]:
        if print_debug:
            logger.info("   " + "━" * 60)
            logger.info("   🔍\tFinal Search Query (After History Rephrasing): ")
            logger.info(f"      '{query}'")
            logger.info("   " + "━" * 60)

        return retrieve_quality_chunks(
            vectorstore,
            query,
            k=int(settings["rag_retrieval_k"]),
            min_results=int(settings["rag_retrieval_min_results"]),
            score_threshold=float(settings["rag_retrieval_score_threshold"]),
            print_debug=print_debug,
        )

    # Step 2: Initialize LLM
    llm = OllamaLLM(
        model=str(settings["llm_model_name"]),
        base_url=cfg.OLLAMA_BASE_URL,
        temperature=float(settings["llm_temp"]),
        num_ctx=int(settings["llm_num_ctx"]),
    )

    # Step 3: Create history-aware retriever prompt with domain awareness
    # Optimized to better leverage chat history for question reformulation
    rephrase_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(  # type: ignore[misc]
        [
            (
                "system",
                cfg.REPHRASE_PROMPT,
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Step 4: Standalone Question Reformulator
    # Rephrase the question using chat history if available, else keep the original
    from langchain_core.runnables import RunnableBranch, RunnablePassthrough

    def check_has_history(x: Dict[str, Any]) -> bool:
        """Check if dictionary has chat history."""
        return bool(x.get("chat_history", []))

    def get_query_str(q: Any) -> str:
        if isinstance(q, dict):
            # For the reformulator, pull the raw input/question string
            q_dict: Dict[str, Any] = q  # pyright: ignore[reportUnknownVariableType]
            return str(q_dict.get("input", q_dict.get("question", "")))
        return str(q)

    def extract_rephrased_question(rephraser_output: Any) -> str:
        # StrOutputParser returns a string. But just to be sure we pull a clean string
        # we parse it out in case it's an AIMessage.
        if hasattr(rephraser_output, "content"):
            return str(rephraser_output.content).strip()
        return str(rephraser_output).strip()

    # Branch automatically executes the rephrase prompt if history is present
    question_rephraser: Any = RunnableBranch(  # pyright: ignore[reportUnknownVariableType]
        (
            check_has_history,
            rephrase_prompt
            | llm
            | StrOutputParser()
            | RunnableLambda(extract_rephrased_question),
        ),
        RunnableLambda(get_query_str),
    )

    # Step 5: Create enhanced prompt template with time and general knowledge permission
    custom_instructions = settings.get("personal_ctx", None)
    sys_prompt_raw: str = cfg.get_sys_prompt(
        custom_instructions=str(custom_instructions) if custom_instructions else None
    )

    # Extract the system portion by cleaning out the hardcoded user question block
    # so we can use a proper conversational Messages format
    clean_sys_prompt = re.sub(
        r"<user_question>.*?</user_question>\n*YOUR ANSWER:?",
        "",
        sys_prompt_raw,
        flags=re.DOTALL,
    )

    qa_prompt: Any = ChatPromptTemplate.from_messages(  # pyright: ignore[reportUnknownMemberType]
        [
            ("system", clean_sys_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )

    def retrieve_and_format_context(x: Dict[str, Any]) -> str:
        # CRITICAL OPTIMIZATION: If the query is detected strictly as a greeting,
        # simply route it cleanly around the FAISS retrieval mechanics.
        # This completely skips useless embedding retrieval but preserves LLM context formatting!
        rephrased_q = str(x.get("rephrased_question", ""))

        # Test BOTH the original query and the newly reformulated query!
        if x.get("is_greeting", False) or is_greeting(rephrased_q):
            return format_context_with_sources(docs=[], print_debug=print_debug)

        docs = quality_retriever(rephrased_q)

        if "__retrieved_docs__" in x:
            retrieved_docs: Any = x["__retrieved_docs__"]
            if isinstance(retrieved_docs, list):
                retrieved_docs.extend(docs)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]

        return format_context_with_sources(
            docs=docs,
            print_debug=print_debug,
        )

    def build_final_prompt_dict(x: Dict[str, Any]) -> Dict[str, Any]:
        """Map variables explicitly to the final qa_prompt structure."""
        current_question = get_query_str(x)
        rephrased_q = str(x.get("rephrased_question", ""))

        if print_debug and current_question != rephrased_q:
            logger.info("   " + "━" * 60)
            logger.info(f"   🧠\tContextual Question Formed: '{rephrased_q}'")
            logger.info("   " + "━" * 60)

        # Ensure we don't accidentally drop the is_greeting flag or other metadata
        return {
            "context": str(x.get("context", "")),
            "question": rephrased_q,
            "chat_history": x.get("chat_history", []),
        }

    # A single dynamic Rag Chain that uses history if provided, bypasses FAISS for greetings,
    # and natively passes the contextualized question directly to the final QA LLM
    rag_chain: Any = (
        RunnablePassthrough.assign(rephrased_question=question_rephraser)
        | RunnablePassthrough.assign(
            context=RunnableLambda(retrieve_and_format_context)
        )
        | RunnableLambda(build_final_prompt_dict)
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    if print_debug:
        logger.info("🚀\tAdvanced RAG Chain created!")
        logger.info(
            "🌊\tChain Flow: Question + History → Rephrase → Quality Retriever → Context Formatter → Enhanced Prompt → LLM → Answer"
        )

    return rag_chain


def get_installed_ollama_models() -> List[str]:
    """Fetch tags from local Ollama API (http://localhost:11434/api/tags)."""
    import requests

    try:
        response = requests.get(f"{cfg.OLLAMA_BASE_URL}/api/tags", timeout=2)
        if response.status_code == 200:
            return [m["name"] for m in response.json().get("models", [])]
        return []
    except Exception:
        return []
