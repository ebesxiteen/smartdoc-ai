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
    """
    Detect and gather system hardware capabilities including CPU, RAM, and GPU.

    This function probes the system for hardware information to enable dynamic
    UI warnings and RAG parameter optimization based on available resources.
    GPU detection attempts to use PyTorch/CUDA if available, with graceful fallback
    to CPU-only mode if GPU detection fails.

    Returns:
        Dict[str, Any]: Hardware capabilities dictionary containing:
            - os (str): Operating system name (e.g., "Linux", "Darwin", "Windows")
            - cpu_cores (int): Number of logical CPU cores
            - ram_gb (float): Total RAM in gigabytes
            - gpu_name (str): GPU model name or "None/Integrated" if not available
            - vram_gb (float): Total GPU VRAM in gigabytes (0.0 if no GPU)

    Example:
        >>> hw_info = get_system_hardware_info()
        >>> if hw_info["vram_gb"] > 0:
        ...     print(f"GPU detected: {hw_info['gpu_name']}")
    """
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
            props = getattr(torch.cuda, "get_device_properties")(0)
            total_memory_bytes = getattr(props, "total_memory", 0)
            total_vram_gb = round(int(total_memory_bytes) / (1024**3), 1)
    except Exception as e:
        debug_log("WARN", message=f"GPU detection failed: {e}")

    return {
        "os": os_name,
        "cpu_cores": cpu_cores,
        "ram_gb": total_ram_gb,
        "gpu_name": gpu_name,
        "vram_gb": total_vram_gb,
    }


def get_default_notebook_settings() -> Dict[str, Any]:
    """
    Get default notebook configuration settings for RAG pipeline initialization.

    Returns a dictionary of default settings covering retrieval parameters (K value,
    score thresholds, chunk sizes), LLM configuration (model name, temperature,
    context window), and hybrid search weights (semantic vs. keyword balance).

    Returns:
        Dict[str, Any]: Default settings dictionary with keys:
            - rag_retrieval_k (int): Number of top chunks to retrieve
            - rag_retrieval_min_results (int): Minimum guaranteed results
            - rag_retrieval_score_threshold (float): Similarity score threshold
            - rag_max_chunk_len (int): Maximum chunk size in characters
            - rag_chunk_overlap (int): Overlap between consecutive chunks
            - rag_max_ctx_len (int): Maximum context length for LLM
            - max_msg_history (int): Maximum chat history messages to retain
            - llm_model_name (str): Ollama model name
            - llm_num_ctx (int): LLM context window size
            - llm_temp (float): LLM temperature (0.0-2.0)
            - personal_ctx (None): Custom instructions placeholder
            - weight_semantic (float): Weight for semantic (0.0-1.0)
            - weight_bm25 (float): Weight for BM25 keyword search (0.0-1.0)

    Note:
        All values are loaded from core/configs.py. Modify settings there to change defaults.
    """
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
        "weight_semantic": cfg.WEIGHT_SEMANTIC,
        "weight_bm25": cfg.WEIGHT_BM25,
    }


def _load_notebook_settings(
    notebook_id: Optional[str],
) -> Dict[str, Any]:
    """
    Load notebook-specific settings from the database, with fallback to defaults.

    Retrieves persisted settings for a specific notebook from the database.
    If the notebook ID is None or notebook has no custom settings, returns
    default settings from get_default_notebook_settings().

    Args:
        notebook_id: The UUID of the notebook to load settings for.
                    If None, default settings are returned.

    Returns:
        Dict[str, Any]: Complete settings dictionary for the notebook.
                       If notebook has custom settings, those are merged with defaults.

    Note:
        This is an internal function used to ensure all RAG chains always have
        complete settings dictionaries, even if the notebook doesn't have
        explicitly saved settings.
    """
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
        "weight_semantic": settings.get("weight_semantic", cfg.WEIGHT_SEMANTIC),
        "weight_bm25": settings.get("weight_bm25", cfg.WEIGHT_BM25),
    }


# ============================================================================
# TEXT UTILITIES
# ============================================================================


def clean_spaces(text: str) -> str:
    """
    Normalize whitespace in text by removing leading/trailing spaces and
    collapsing multiple internal spaces into single spaces.

    This is useful for cleaning text extracted from PDFs or other sources
    that may have inconsistent whitespace formatting.

    Args:
        text: Input string to normalize.

    Returns:
        str: Text with normalized whitespace. Returns empty string if input is falsy.

    Example:
        >>> clean_spaces("  Hello    world  ")
        'Hello world'
    """
    if not text:
        return ""
    # .split() without arguments splits by any whitespace (space, \n, \t)
    # ' '.join(...) puts exactly one space between each word
    return " ".join(text.split())


# ============================================================================
# DEBUG LOGGING UTILITIES & CONSTANTS
# ============================================================================


def print_breaker() -> None:
    """
    Print a visual separator line in logs to demarcate logical sections.

    Outputs a 70-character wide line (━) to visually divide log sections,
    improving readability when scanning terminal output.
    """
    separator = "━" * 70
    logger.info(separator)


def debug_log(
    log_type: str = "INFO",
    emoji: Optional[str] = None,
    message: str = "",
    color: Optional[str] = None,
    show_metadata: bool = True,
) -> None:
    """
    Enhanced structured logging with automatic categorization and ANSI color support.

    This logger provides consistent, visually distinct log messages with automatic
    color mapping, caller metadata extraction, and support for structured log categories.

    Args:
        log_type: Log category or severity type. Can be:
                 - Standard type: "INFO", "WARNING", "ERROR", "SUCCESS"
                 - Structured category: Any key from LOG_CATEGORIES dict
                 Default: "INFO"
        emoji: Emoji to prefix the log message. If None and log_type is a key
               in LOG_CATEGORIES, the emoji from that category is used.
               Default: None
        message: The main log message content.
                Default: ""
        color: Optional ANSI color code override (e.g., "31" for red).
               If not provided, color is determined by log_type.
               Default: None
        show_metadata: Whether to include caller filename and line number in output.
                      Default: True

    Example:
        >>> debug_log("SUCCESS", "✅", "File loaded successfully")
        >>> debug_log("KEYWORD", message="Using BM25 keyword search")
        >>> debug_log("WARN", message="Threshold not met, using fallback")
    """
    import inspect
    import os

    # 1. Define Default Color Map
    # ANSI color codes: 31=Red, 33=Yellow, 36=Cyan, 32=Green
    COLOR_MAP = {
        "ERROR": "31",
        "WARNING": "33",
        "INFO": "36",
        "SUCCESS": "32",
    }

    # 2. Normalize type and determine emoji + color
    normalized_type = (log_type or "INFO").upper()

    # Check if it's a structured category
    if normalized_type in cfg.LOG_CATEGORIES:
        category = cfg.LOG_CATEGORIES[normalized_type]
        used_emoji = emoji or category["emoji"]
        used_type = category["type"]
        final_color = color if color else COLOR_MAP.get(used_type, "36")
    else:
        used_emoji = emoji
        used_type = normalized_type
        final_color = color if color else COLOR_MAP.get(normalized_type, "36")

    # 3. Capture caller metadata (if enabled)
    metadata_str = ""
    if show_metadata:
        frame = inspect.stack()[1]
        filename = os.path.basename(frame.filename)
        lineno = frame.lineno
        metadata_str = f"\033[{final_color}m[{filename}:{lineno}]\033[0m"

    # 4. Format the output
    # \u2009 is a thin space to prevent emoji overlap
    prefix = f"{used_emoji}\u2009" if used_emoji else ""
    if metadata_str:
        printed_message = f"{prefix}{metadata_str} {message}"
    else:
        printed_message = f"{prefix}{message}"

    # 5. Route to logger based on type
    if used_type == "ERROR":
        logger.error(printed_message)
    elif used_type == "WARNING":
        logger.warning(printed_message)
    elif used_type == "SUCCESS":
        logger.info(printed_message)
    else:
        logger.info(printed_message)


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
) -> List[Document]:
    """
    Load a file (PDF or DOCX) and split it into overlapping text chunks.

    This function extracts text from PDF or DOCX files and uses a recursive
    text splitter to create overlapping chunks suitable for embedding and RAG.
    Progress updates can be provided via callback for UI integration.

    Args:
        file_path: Absolute path to the file to load.
        file_type: File format - either "pdf" or "docx".
        chunk_size: Maximum characters per chunk. Default from configs.py.
        chunk_overlap: Character overlap between consecutive chunks. Default from configs.py.
        print_debug: If True, log detailed processing information. Default False.
        progress_callback: Optional callable to receive progress messages (e.g., for UI progress bar).

    Returns:
        List[Document]: LangChain Document objects with chunked text content and metadata.
                       Each document contains:
                       - page_content: Chunk text
                       - metadata: {"source": file_path, "page": page_number, "document": filename}

    Raises:
        ValueError: If file_type is neither "pdf" nor "docx".
        Exception: Various exceptions from PyMuPDF or python-docx if file parsing fails.

    Example:
        >>> chunks = load_and_chunk_file(
        ...     "document.pdf",
        ...     "pdf",
        ...     chunk_size=1000,
        ...     print_debug=True
        ... )
        >>> print(f"Created {len(chunks)} chunks")
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
        debug_log(
            "SUCCESS",
            message=f"Loaded {file_type.upper()}: {file_path}\n      {len(documents)} pages | {sum(len(d.page_content) for d in documents)} chars",
        )

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
        avg_chunk_size = (
            sum(len(chunk.page_content) for chunk in chunks) / len(chunks)
            if chunks
            else 0
        )
        debug_log(
            "COMPLETE",
            message=f"Chunking complete: {len(chunks)} chunks\n      Avg chunk size: {avg_chunk_size:.0f} chars",
        )

    return chunks


def hash_file_content(file_bytes: bytes) -> str:
    """
    Calculate MD5 hash of file content for duplicate detection.

    Args:
        file_bytes: Raw file content as bytes.

    Returns:
        str: Hex-encoded MD5 hash of the file content.

    Note:
        MD5 is used for duplicate detection only, not security. SHA256 could
        be used for enhanced security if needed in future.
    """
    return hashlib.md5(file_bytes).hexdigest()


def detect_file_type(file_bytes: bytes) -> str:
    """
    Detect file type based on magic number (file signature bytes).

    Args:
        file_bytes: First few bytes of the file content.

    Returns:
        str: Detected file type - either "pdf" or "docx".

    Raises:
        ValueError: If file type is neither PDF nor DOCX.
    """
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
    """
    Check if a file with the given hash was already uploaded to any notebook.

    Args:
        file_hash: MD5 hash of file content from hash_file_content().

    Returns:
        Dict[str, Any]: Source document info if found, None if not found.
    """
    from db.crud import get_source_by_hash

    return get_source_by_hash(file_hash)


def check_file_already_exists_in_notebook(
    file_hash: str, notebook_id: str
) -> Optional[Dict[str, Any]]:
    """
    Check if a file with the given hash was already uploaded to a specific notebook.

    Args:
        file_hash: MD5 hash of file content from hash_file_content().
        notebook_id: UUID of the notebook to check within.

    Returns:
        Dict[str, Any]: Source document info if found in this notebook, None if not found.
    """
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
    """
    Load a file, split into chunks, and enrich with metadata.

    Wraps load_and_chunk_file() to add the original filename to chunk metadata
    for source attribution in RAG responses.

    Args:
        file_path: Absolute path to the file.
        file_type: "pdf" or "docx".
        filename: Display name of the file (for metadata).
        print_debug: Enable debug logging.
        progress_callback: Optional progress update callback.
        chunk_size: Maximum chunk size in characters.
        chunk_overlap: Character overlap between chunks.

    Returns:
        Tuple[List[Document], int]: Tuple of:
            - List of Document objects with metadata
            - Count of created chunks
    """
    if print_debug:
        debug_log("INFO", "📄", f"Loading and chunking {file_type.upper()}: {filename}")
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
    """
    Create a FAISS vector database from document chunks.

    Args:
        chunks: List of Document objects with text content to embed and index.
        embeddings: HuggingFaceEmbeddings instance for vectorization.
        print_debug: Enable debug logging.

    Returns:
        FAISS: Vector store instance ready for similarity search.
    """
    if print_debug:
        debug_log("EMBED", message=f"Creating vectorstore from {len(chunks)} chunks...")
    return FAISS.from_documents(chunks, embeddings)


def merge_vectorstores(
    existing_vectorstore: Optional[FAISS],
    new_vectorstore: FAISS,
    print_debug: bool = False,
) -> FAISS:
    """
    Merge a new vectorstore into an existing one, or return the new one if none exists.

    Used when multiple sources are added to a notebook to maintain a consolidated
    vectorstore for efficient similarity search across all sources.

    Args:
        existing_vectorstore: Current FAISS vectorstore or None.
        new_vectorstore: New FAISS vectorstore to merge in.
        print_debug: Enable debug logging.

    Returns:
        FAISS: Merged vectorstore combining both inputs, or new_vectorstore if no existing one.
    """
    if existing_vectorstore is None:
        if print_debug:
            debug_log("TASK_START", message="No existing vectorstore - using new one")
        return new_vectorstore

    if print_debug:
        debug_log("MERGED", message="Vectorstores merged successfully")
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
    """
    Save source file metadata and vectorstore location to the database.

    Records a new source document in the notebook, linking it to its FAISS vectorstore.
    Enables tracking of document sources for citation and content refresh.

    Args:
        notebook_id: UUID of the notebook this source belongs to.
        filename: Original filename displayed to users.
        file_type: "pdf" or "docx".
        file_hash: MD5 hash of file content for duplicate detection.
        summary: AI-generated summary of the document.
        suggested_questions: List of auto-generated discussion questions.
        vectorstore_path: Path to the saved FAISS vectorstore directory.
        source_id: Optional pre-generated UUID. If None, one is created.
        print_debug: Enable debug logging.

    Returns:
        str: The source_id (UUID) assigned to this source.

    Note:
        The vectorstore_path is typically computed via get_source_vectorstore_dir().
    """
    from middlewares import db_middleware as db

    if print_debug:
        debug_log("SAVED", message=f"Saving source to database: {filename}")

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
    Process a user query through the RAG pipeline to generate an answer with source citations.

    Handles three query scenarios:
    1. Greetings/General knowledge: Answer without document retrieval
    2. Follow-up questions: Use chat history to rephrase query for better retrieval
    3. Standalone questions: Direct document retrieval and answering

    Args:
        query: The user's question.
        rag_chain: The instantiated RAG chain from create_history_aware_rag_chain().
        vectorstore: FAISS vectorstore for document retrieval.
        chat_history: Optional list of previous Q&A exchanges for context.
        print_debug: Enable detailed debug logging.
        notebook_id: UUID of the notebook for loading settings.

    Returns:
        Tuple[str, List[Dict[str, Any]], bool]:
            - answer (str): Clean LLM response (with status tags removed)
            - sources (List[Dict]): Citation sources with keys: document, page, content
            - found_answer (bool): Whether relevant context was found (affects UI display)

    Note:
        Answers are tagged with [STATUS: DOC_ANSWER], [STATUS: DOC_MISSING], or [STATUS: GENERAL].
        These tags are stripped before returning to maintain clean output.
    """
    if print_debug:
        query_preview = f"{query[:60]}..." if len(query) > 60 else query
        debug_log("QUERY", message=f"Processing: {query_preview}")

    # Step 1: Check if this is a greeting/general question
    is_greeting_query = is_greeting(query)
    if print_debug and is_greeting_query:
        debug_log("DEBUG", message="Detected greeting → using general knowledge")

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
            print_breaker()
            debug_log(
                "HISTORY", message=f"Chat history: {len(formatted_history)} messages"
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
                debug_log("ITEM", emoji=role_icon, message=content_preview)
            print_breaker()

    # Step 3: Generate answer through RAG chain
    if print_debug:
        if is_greeting_query:
            debug_log("CHAIN", message="Invoking RAG chain [General Knowledge Mode]...")
        else:
            debug_log(
                "CHAIN", message="Invoking RAG chain [Document Retrieval Mode]..."
            )

    try:
        # Invoke chain with proper input structure
        # The history-aware chain expects dict with "input" and optional "chat_history"
        answer = rag_chain.invoke(chain_input_dict)
    except Exception as e:
        debug_log("ERROR", message=f"RAG chain failed: {e}")
        answer = f"[STATUS: DOC_MISSING]\nI encountered an error processing your query: {str(e)}"

    if print_debug:
        debug_log("RESPONSE", message=f"LLM response generated ({len(answer)} chars)")

        # Log the actual answer cleanly
        print_breaker()
        debug_log("LLM_INIT", message="LLM Answer:")

        # Split by newlines so it aligns cleanly in the terminal
        for idx, line in enumerate(answer.split("\n")):
            if line.strip():
                # To prevent overwhelming logs, limit to first 10 lines max or 1000 chars
                if idx > 9 or len(line) > 150:
                    preview = line[:150] + "..." if len(line) > 150 else line
                    debug_log("INFO", None, f"{preview}")
                else:
                    debug_log("INFO", None, f"{line}")
        print_breaker()

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
            debug_log(
                "INFO", "📍", "LLM found relevant context - sources will be displayed"
            )
        elif is_general_answer:
            debug_log(
                "INFO", "💬", "LLM answered from general knowledge - no sources needed"
            )
        else:
            debug_log(
                "INFO",
                "⚠️",
                "LLM did not find relevant context - sources will be hidden",
            )

    # Step 5: Retrieve source documents for display (only if not a general question)
    sources: List[Dict[str, Any]] = []

    if not is_general_answer and print_debug:
        debug_log("INFO", "📎", "Gathering exact retrieved sources for display...")

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
                debug_log(
                    "INFO",
                    None,
                    f"• Source {i}: {source_entry['document']} (Page {source_entry['page']})",
                )

        if print_debug:
            print_breaker()
            debug_log("INFO", "📎", f"Processed {len(sources)} display sources for UI")

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
        debug_log(
            "INFO",
            "💾",
            f"Saving query and answer to history for notebook {notebook_id}",
        )

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
        debug_log("INFO", "📝", f"Saving answer as note in notebook {notebook_id}")

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
        debug_log("INFO", "🔳", "Attempting to load embeddings on GPU...")
        return HuggingFaceEmbeddings(
            model_name=cfg.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cuda"},
        )
    except Exception as e:
        debug_log("WARNING", "⚠️", f"GPU loading failed: {e}. Falling back to CPU...")
        try:
            return HuggingFaceEmbeddings(
                model_name=cfg.EMBEDDING_MODEL_NAME,
                model_kwargs={"device": "cpu"},
            )
        except Exception as e2:
            debug_log(
                "ERROR", "❌", f"CPU loading failed: {e2}. Embeddings cannot be loaded."
            )
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
    Retrieve document chunks with quality-based filtering and fallback guarantees.

    Uses L2 distance similarity scoring to filter chunks. If strict threshold filtering
    returns fewer than min_results, falls back to returning top-K results without
    threshold filtering to ensure the LLM always gets minimum context.

    Args:
        vectorstore: FAISS vectorstore containing indexed documents.
        query: User's question to retrieve relevant chunks for.
        k: Maximum number of chunks to retrieve.
        min_results: Minimum guaranteed result count (triggers fallback if not met).
        score_threshold: L2 distance threshold; lower scores = better matches.
                        Typical range: 0.0-30.0 for multilingual embeddings.
        print_debug: Enable detailed logging of retrieval process.

    Returns:
        List[Document]: Ordered list of relevant document chunks with metadata including
                       similarity_score and passed_quality_threshold flags.

    Note:
        L2 distance metric used by FAISS: lower distance = higher similarity.
        Values typically range 4.0-6.0 for moderate semantic relevance.
    """
    if print_debug:
        debug_log(
            "INFO",
            "🔎",
            f"Retrieving quality chunks from {k} total results with the threshold of {score_threshold} and min fallback of {min_results}",
        )

    # Step 1: Get top K results with scores
    results_with_scores: List[Any] = vectorstore.similarity_search_with_score(  # pyright: ignore[reportUnknownMemberType]
        query, k=k
    )

    if print_debug:
        debug_log("INFO", None, f"Total search results: {len(results_with_scores)}")

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
        debug_log(
            "INFO",
            "✅",
            f"Passed threshold ({score_threshold}): {passed_threshold} chunks",
        )
        debug_log("INFO", "❌", f"Failed threshold: {failed_threshold} chunks")

    # Step 3: Fallback mechanism - if threshold filtered too much, use top min_results
    if len(filtered_results) < min_results:
        if print_debug:
            debug_log(
                "WARNING",
                "⚠️",
                f"Only {len(filtered_results)} chunks passed threshold. Falling back to top {min_results} results...",
            )

        filtered_results = []
        for doc, score in results_with_scores[:min_results]:
            doc.metadata["similarity_score"] = score
            doc.metadata["passed_quality_threshold"] = False
            filtered_results.append(doc)

    if print_debug:
        debug_log(
            "INFO", "✅", f"Final retrieved chunks for display: {len(filtered_results)}"
        )
        print_breaker()

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

            debug_log(
                "INFO",
                "📄",
                f'[{i}] L2 Distance: {score_str} | {doc_name} (Page {page_num}): "{content_preview}"...',
            )

        print_breaker()

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
            debug_log("INFO", "📄", "No relevant context found to format.")
        return "No relevant context found."

    context_parts: List[str] = []
    current_length = 0

    if print_debug:
        debug_log(
            "INFO",
            "📄",
            f"Formatting {len(docs)} chunks into context with max total length of {cfg.RAG_MAX_CTX_LEN} bytes...",
        )

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
                debug_log(
                    "WARNING",
                    "⚠️",
                    f"Max context length reached. Stopping at chunk {idx - 1}. (Current: {current_length}B, Limit: {cfg.RAG_MAX_CTX_LEN}B)",
                )
            break

        context_parts.append(part)
        current_length += part_length

        if print_debug:
            debug_log(
                "INFO",
                "📑",
                f"Added chunk {idx}: Page {page_num} ({part_length}B, running total: {current_length}B)",
            )

    final_context = "\n\n".join(context_parts)
    if print_debug:
        debug_log(
            "INFO",
            "📄",
            f"Final formatted context prepared with {len(context_parts)} chunks, total length: {len(final_context)} bytes",
        )

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
                debug_log(
                    "WARNING",
                    "⚠️",
                    f"Failed to load vectorstore for source {source_id}: {e}",
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
        debug_log(
            "WARNING",
            "⚠️",
            f"Language detection failed: {e}. Using English patterns as fallback.",
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
    Build a production-ready history-aware RAG chain with hybrid search capability.

    This is the core RAG orchestration function. It constructs a sophisticated pipeline that:
    1. Uses chat history to rephrase questions for better retrieval
    2. Implements a "Gatekeeper" pattern to choose Pure Semantic, Pure Keyword, or Hybrid search
    3. Retrieves context with quality-based threshold filtering and fallback guarantees
    4. Formats context with source citations
    5. Generates answers using LLM with document context + general knowledge fallback

    Args:
        vectorstore: FAISS vectorstore containing indexed documents.
        print_debug: Enable detailed debug logging of chain execution.
        notebook_id: UUID of the notebook for loading custom RAG settings.

    Returns:
        Any: Runnable RAG chain that accepts {"input": question, "chat_history": [...]} (optional)
             and returns the final answer string.

    Pipeline Flow:
        Question + History → Rephrase → Quality Retriever → Context Formatter → Enhanced Prompt → LLM → Answer

    Hybrid Search Modes (controlled by weight_semantic and weight_bm25 settings):
        - Pure Semantic (weight_bm25 = 0.0): FAISS vector similarity only
        - Pure Keyword (weight_semantic = 0.0): BM25 lexical matching only
        - Hybrid (both > 0.0): EnsembleRetriever with RRF (Reciprocal Rank Fusion)

    Note:
        Uses Streamlit session_state to cache the expensive BM25 tokenizer.
        Automatically invalidates cache when vectorstore content or notebook changes.
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_ollama import OllamaLLM
    from langchain_core.prompts import MessagesPlaceholder

    if print_debug:
        print("\n")
        debug_log("INFO", "🛠️", "Building History-Aware RAG Chain...")

    settings = _load_notebook_settings(notebook_id)

    import streamlit as st
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from pydantic import Field
    from langchain_community.retrievers import BM25Retriever
    from langchain_classic.retrievers import EnsembleRetriever

    class CustomFAISSRetriever(BaseRetriever):
        vectorstore: Any = Field(description="FAISS vectorstore")
        settings_dict: Dict[str, Any] = Field(description="Settings dict")
        print_debug: bool = Field(default=False)

        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> List[Document]:
            return retrieve_quality_chunks(
                self.vectorstore,
                query,
                k=int(self.settings_dict["rag_retrieval_k"]),
                min_results=int(self.settings_dict["rag_retrieval_min_results"]),
                score_threshold=float(
                    self.settings_dict["rag_retrieval_score_threshold"]
                ),
                print_debug=self.print_debug,
            )

    # Step 1: Create hybrid/quality-based retriever implementations
    k_val = int(settings["rag_retrieval_k"])
    weight_semantic = float(settings.get("weight_semantic", cfg.WEIGHT_SEMANTIC))
    weight_bm25 = float(settings.get("weight_bm25", cfg.WEIGHT_BM25))
    min_results = int(settings["rag_retrieval_min_results"])

    # Provide FAISS Semantic Base Retriever
    faiss_retriever = CustomFAISSRetriever(
        vectorstore=vectorstore, settings_dict=settings, print_debug=print_debug
    )

    # Build or Load BM25 Retriever from Streamlit Cache (The "Cold Start" Strategy)
    current_doc_count = len(vectorstore.index_to_docstore_id)
    current_vs_id = id(vectorstore)

    rebuild_bm25 = (
        "bm25_retriever" not in st.session_state
        or st.session_state.get("bm25_doc_count") != current_doc_count
        or st.session_state.get("bm25_vs_id") != current_vs_id
        or st.session_state.get("bm25_notebook_id") != notebook_id
    )

    if rebuild_bm25:
        if print_debug:
            debug_log(
                "INFO", "⚙️", "Building/Rebuilding BM25Retriever from FAISS docstore..."
            )
        all_docs_ids = vectorstore.index_to_docstore_id.values()
        all_docs = [
            doc
            for doc_id in all_docs_ids
            if isinstance((doc := vectorstore.docstore.search(doc_id)), Document)
        ]
        if len(all_docs) > 0:
            bm25 = BM25Retriever.from_documents(all_docs)
            bm25.k = k_val
            st.session_state["bm25_retriever"] = bm25
        else:
            st.session_state["bm25_retriever"] = None

        st.session_state["bm25_doc_count"] = current_doc_count
        st.session_state["bm25_vs_id"] = current_vs_id
        st.session_state["bm25_notebook_id"] = notebook_id

    bm25_retriever = st.session_state.get("bm25_retriever")
    if bm25_retriever:
        bm25_retriever.k = k_val  # update k dynamically to user settings

    # The Gatekeeper Architectural Flow
    hybrid_retriever: Any = None
    if bm25_retriever is None or abs(weight_bm25) < 1e-5:
        if print_debug:
            debug_log(
                "INFO",
                "🧩",
                f"Initialized Pure Semantic Mode (Weight Semantic={weight_semantic:.2f} / Weight BM25=0.00)",
            )
        hybrid_retriever = faiss_retriever
    elif abs(weight_semantic) < 1e-5:
        if print_debug:
            debug_log(
                "INFO",
                "🧩",
                f"Initialized Pure Keyword Mode (Weight Semantic=0.00 / Weight BM25={weight_bm25:.2f})",
            )
        hybrid_retriever = bm25_retriever
    else:
        if print_debug:
            debug_log(
                "INFO",
                "🧩",
                f"Initialized Hybrid Ensemble Mode (Weight Semantic={weight_semantic:.2f} / Weight BM25={weight_bm25:.2f})",
            )
        hybrid_retriever = EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=[weight_semantic, weight_bm25],
            c=cfg.RRF_C,
        )

    def quality_retriever(query: str) -> List[Document]:
        if print_debug:
            print_breaker()
            debug_log("INFO", "🔍", f'Executing retrieval with query: "{query}"')
            if bm25_retriever is None or abs(weight_bm25) < 1e-5:
                debug_log(
                    "INFO",
                    "🧩",
                    f"Executing Pure Semantic Search (Weight Semantic: {weight_semantic:.2f} / Weight BM25: 0.00)",
                )
            elif abs(weight_semantic) < 1e-5:
                debug_log(
                    "INFO",
                    "🧩",
                    f"Executing Pure Keyword Search (Weight Semantic: 0.00 / Weight BM25: {weight_bm25:.2f})",
                )
            else:
                debug_log(
                    "INFO",
                    "🧩",
                    f"Executing Hybrid Ensemble Search (Weight Semantic: {weight_semantic:.2f} / Weight BM25: {weight_bm25:.2f})",
                )
            print_breaker()

        docs: List[Document] = hybrid_retriever.invoke(query)

        # Enforce exact top_k limit because EnsembleRetriever fuses lists and might return 2 * k elements
        if len(docs) > k_val:
            docs = docs[:k_val]

        # Fallback & Min Guarantee logic
        if len(docs) < min_results:
            if print_debug:
                debug_log(
                    "WARNING",
                    "⚠️",
                    f"Hybrid/Keyword Search returned {len(docs)} chunks. Triggering Fallback to Pure Semantic Search with relaxed threshold to guarantee at least {min_results} results.",
                )

            # Lower score threshold or run pure similarity
            fallback_docs = retrieve_quality_chunks(
                vectorstore,
                query,
                k=min_results,
                min_results=min_results,
                score_threshold=100.0,
                print_debug=print_debug,
            )
            return fallback_docs

        return docs

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
            debug_log("INFO", "🧠", f'Contextual Question Formed: "{rephrased_q}"')

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
        debug_log(
            "INFO",
            "🚀",
            "Advanced RAG Chain created: Question + History → Rephrase → Quality Retriever → Context Formatter → Enhanced Prompt → LLM → Answer",
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
