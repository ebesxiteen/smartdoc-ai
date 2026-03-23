"""
Centralized utility functions for RAG application.
Includes: PDF processing, chat handling, vectorstore management, and text cleaning.
"""

from datetime import datetime, timezone
import hashlib
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple
import logging

import streamlit as st
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

from core.configs import (
    APP_NAME,
    ASSISTANT_ROLE_NAME,
    DEFAULT_EN_GREETING_PATTERNS,
    EMBEDDING_MODEL_NAME,
    GREETING_PATTERNS_BY_LANGUAGE,
    LLM_BASE_URL,
    LLM_MODEL_NAME,
    LLM_NUM_CTX,
    LLM_PROMPT_TEMPLATE,
    LLM_TEMPERATURE,
    MAX_MSG_HISTORY,
    RAG_CHUNK_OVERLAP,
    RAG_MAX_CHUNK_LENGTH,
    RAG_MAX_CONTEXT_LENGTH,
    RAG_RETRIEVAL_K,
    RAG_RETRIEVAL_MIN_RESULTS,
    RAG_RETRIEVAL_SCORE_THRESHOLD,
    REPHRASE_PROMPT,
    USER_ROLE_NAME,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever
from pathlib import Path
import re
from langdetect import detect  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

logger = logging.getLogger(__name__)

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


def load_and_chunk_pdf(
    pdf_path: str,
    chunk_size: int = RAG_MAX_CHUNK_LENGTH,
    chunk_overlap: int = RAG_CHUNK_OVERLAP,
    print_debug: bool = False,
):
    """
    Load a PDF and chunk it into overlapping segments.

    Args:
      pdf_path: Path to the PDF file

    Returns:
      List of Document objects with chunked text
    """
    # Step 1: Load the PDF
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    if print_debug:
        print(f"✅ Loaded PDF: {pdf_path}")
        print(f"   Total pages: {len(documents)}")
        print(f"   Total characters: {sum(len(doc.page_content) for doc in documents)}")

    # Step 2: Split into chunks
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


def hash_pdf_file(file_bytes: bytes) -> str:
    """Calculate MD5 hash of PDF file content."""
    return hashlib.md5(file_bytes).hexdigest()


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


def chunk_and_process_pdf(
    pdf_path: str, filename: str, print_debug: bool = False
) -> Tuple[List[Document], int]:
    """Load PDF, chunk it, and add metadata."""
    if print_debug:
        logger.info(f"Loading and chunking PDF: {filename}")
    chunks = load_and_chunk_pdf(pdf_path)

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
) -> tuple[str, List[Dict[str, Any]], bool]:
    """
    Process a user query through the RAG chain with optional chat history context.

    Handles three scenarios:
    1. Greeting/general questions: Answer without searching documents
    2. Follow-up questions: Use chat history to rephrase for accurate retrieval
    3. New questions: Direct document search and answer

    Returns:
        (answer, sources_list, found_answer)
        - answer: The LLM's response (with [FOUND_ANSWER: ...] tag removed)
        - sources_list: Top sources for display
        - found_answer: Boolean indicating if LLM found useful context (defaults to True if not specified)
    """
    if print_debug:
        logger.info(
            f"🔡 Processing query: {query[:60]}..."
            if len(query) > 60
            else f"🔡 Processing query: {query}"
        )

    # Step 1: Check if this is a greeting/general question
    is_greeting_query = is_greeting(query)
    if print_debug and is_greeting_query:
        logger.info("👋 Detected greeting/general question - using general knowledge")

    # Step 2: Prepare inputs for the RAG chain
    # Build chain input dict with proper structure for history-aware chain
    chain_input_dict: Dict[str, Any] = {"input": query, "question": query}

    if chat_history and not is_greeting_query:
        formatted_history = format_chat_history_for_rephrase(
            chat_history, max_messages=MAX_MSG_HISTORY
        )
        chain_input_dict["chat_history"] = formatted_history  # type: ignore[typeddict-unknown-key]
        if print_debug:
            logger.info(
                f"📜 Including {len(formatted_history)} history messages for context"
            )

    # Step 3: Generate answer through RAG chain
    if print_debug:
        if is_greeting_query:
            logger.info("⏳ Invoking RAG chain with general knowledge mode...")
        else:
            logger.info("⏳ Invoking RAG chain with quality-filtered chunks...")

    try:
        # Invoke chain with proper input structure
        # The history-aware chain expects dict with "input" and optional "chat_history"
        answer = rag_chain.invoke(chain_input_dict)
    except Exception as e:
        logger.error(f"Error invoking RAG chain: {e}")
        if print_debug:
            logger.error(f"    Full error: {str(e)}")
        answer = f"[FOUND_ANSWER: false]\nI encountered an error processing your query: {str(e)}"

    if print_debug:
        logger.info(f"✅ LLM response generated ({len(answer)} chars)")

    # Step 4: Parse [FOUND_ANSWER: true/false/general] tag from answer
    found_answer = True  # Default to True
    answer_clean = answer
    is_general_answer = is_greeting_query

    # Check for [FOUND_ANSWER: true] tag
    if "[FOUND_ANSWER: true]" in answer:
        found_answer = True
        answer_clean = answer.replace("[FOUND_ANSWER: true]", "").strip()
    # Check for [FOUND_ANSWER: false] tag
    elif "[FOUND_ANSWER: false]" in answer:
        found_answer = False
        answer_clean = answer.replace("[FOUND_ANSWER: false]", "").strip()
        is_general_answer = True
    # Check for [FOUND_ANSWER: general] tag (new general knowledge marker)
    elif "[FOUND_ANSWER: general]" in answer:
        found_answer = False
        answer_clean = answer.replace("[FOUND_ANSWER: general]", "").strip()
        is_general_answer = True

    if print_debug:
        if found_answer:
            logger.info("✅ LLM found relevant context - sources will be displayed")
        elif is_general_answer:
            logger.info("💡 LLM answered from general knowledge - no sources needed")
        else:
            logger.warning(
                "⚠️  LLM did not find relevant context - sources will be hidden"
            )

    # Step 5: Retrieve source documents for display (only if not a general question)
    sources: List[Dict[str, Any]] = []

    if not is_general_answer and print_debug:
        logger.info("📎 Retrieving quality-filtered sources for display...")

    if not is_general_answer:
        source_docs = retrieve_quality_chunks(
            vectorstore,
            query,
            k=RAG_RETRIEVAL_K,
            min_results=RAG_RETRIEVAL_MIN_RESULTS,
            score_threshold=RAG_RETRIEVAL_SCORE_THRESHOLD,
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
                logger.debug(
                    f"   Source {i}: {source_entry['document']} (Page {source_entry['page']})"
                )

        if print_debug:
            logger.info(f"✅ Retrieved {len(sources)} display sources")

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
        role=USER_ROLE_NAME,
        content=query,
    )

    # Save assistant message with sources
    db.add_chat_message(
        notebook_id=notebook_id,
        role=ASSISTANT_ROLE_NAME,
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
        logger.info("Attempting to load embeddings on GPU")
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cuda"},
        )
    except Exception as e:
        logger.warning(f"GPU loading failed: {e}. Falling back to CPU...")
        try:
            return HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={"device": "cpu"},
            )
        except Exception as e2:
            logger.error(f"CPU loading also failed: {e2}")
            return None


def retrieve_quality_chunks(
    vectorstore: FAISS,
    query: str,
    k: int = RAG_RETRIEVAL_K,
    min_results: int = RAG_RETRIEVAL_MIN_RESULTS,
    score_threshold: float = RAG_RETRIEVAL_SCORE_THRESHOLD,
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
        doc.metadata["similarity_score"] = score  # type: ignore[index]

        # Apply quality threshold: only include chunks with good similarity
        if score <= score_threshold:
            filtered_results.append(doc)
            doc.metadata["passed_quality_threshold"] = True  # type: ignore[index]
            passed_threshold += 1
        else:
            failed_threshold += 1

    if print_debug:
        logger.info(
            f"   ✓ Passed threshold ({score_threshold}): {passed_threshold} chunks"
        )
        logger.info(f"   ✗ Failed threshold: {failed_threshold} chunks")

    # Step 3: Fallback mechanism - if threshold filtered too much, use top min_results
    if len(filtered_results) < min_results:
        if print_debug:
            logger.warning(
                f"   ⚠️  Only {len(filtered_results)} chunks passed threshold. "
                f"Falling back to top {min_results} results..."
            )

        filtered_results = []
        for doc, score in results_with_scores[:min_results]:
            doc.metadata["similarity_score"] = score  # type: ignore[index]
            doc.metadata["passed_quality_threshold"] = False  # type: ignore[index]
            filtered_results.append(doc)

    if print_debug:
        logger.info(f"   ✅ Final selection: {len(filtered_results)} chunks\n")

        for i, doc in enumerate(filtered_results, 1):
            doc_metadata: Dict[str, Any] = doc.metadata  # type: ignore[assignment]
            score = doc_metadata.get("similarity_score", "N/A")
            doc_name = doc_metadata.get("document", "Unknown")
            page_num = doc_metadata.get("page", "N/A")
            content_preview = doc.page_content[:80].replace("\n", " ")

            logger.debug(f"   📄 Chunk {i}: {doc_name} (Page {page_num})")
            score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
            logger.debug(f"      Distance: {score_str} | Content: {content_preview}...")

    return filtered_results


def format_context_with_sources(
    docs: List[Document],
    max_chunk_length: int = RAG_MAX_CHUNK_LENGTH,
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
        if current_length + part_length > RAG_MAX_CONTEXT_LENGTH:
            if print_debug:
                logger.debug(
                    f"   ⚠️  Max context length reached. Stopping at chunk {idx - 1}. "
                    f"(Current: {current_length}B, Limit: {RAG_MAX_CONTEXT_LENGTH}B)"
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


def create_rag_chain(vectorstore: FAISS, print_debug: bool = False) -> Any:
    """
    Build the complete RAG chain: Quality Retriever → Format Context → Prompt → LLM

    Uses intelligent quality-based chunk retrieval to maximize relevance while
    preventing CUDA OOM errors through configurable limits.

    Args:
        vectorstore: FAISS vectorstore object

    Returns:
        Runnable chain object
    """
    if print_debug:
        print("\n🔗 Building RAG Chain...")
        print(
            f"   Retrieval Config: k={RAG_RETRIEVAL_K}, min_results={RAG_RETRIEVAL_MIN_RESULTS}, threshold={RAG_RETRIEVAL_SCORE_THRESHOLD}"
        )
        print(
            f"   Context Config: max_length={RAG_MAX_CONTEXT_LENGTH}, max_chunk={RAG_MAX_CHUNK_LENGTH}"
        )
        print(
            f"   LLM Config: model={LLM_MODEL_NAME}, num_ctx={LLM_NUM_CTX}, temperature={LLM_TEMPERATURE}"
        )

    # Step 1: Create quality-based retriever (replaces basic vectorstore.as_retriever())
    def quality_retriever(query: str) -> List[Document]:
        return retrieve_quality_chunks(
            vectorstore,
            query,
            k=RAG_RETRIEVAL_K,
            min_results=RAG_RETRIEVAL_MIN_RESULTS,
            score_threshold=RAG_RETRIEVAL_SCORE_THRESHOLD,
            print_debug=print_debug,
        )

    # Step 2: Define the custom prompt template with source instructions
    prompt_template = LLM_PROMPT_TEMPLATE

    # Step 3: Initialize the LLM (Ollama running Qwen2.5)
    llm = OllamaLLM(
        model=LLM_MODEL_NAME,
        base_url=LLM_BASE_URL,
        temperature=LLM_TEMPERATURE,
        num_ctx=LLM_NUM_CTX,
    )

    # Step 4: Build the chain using LCEL
    # This chains: question → quality_retriever → format context → prompt → llm → output
    rag_chain: Any = (  # type: ignore
        {
            "context": RunnableLambda(
                lambda query: format_context_with_sources(
                    docs=quality_retriever(query),  # pyright: ignore[reportArgumentType]
                    print_debug=print_debug,
                )
            ),
            "question": RunnablePassthrough(),
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    if print_debug:
        print("   ✅ RAG Chain created!")
        print("\n   Chain Flow:")
        print(
            "   Question → Quality Retriever (Threshold + Fallback) → Context Formatter → Prompt → LLM (Qwen2.5) → Answer"
        )

    return rag_chain


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
        st.session_state.rag_chain = create_rag_chain(
            st.session_state.vectorstore, print_debug
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
        if detected_lang.startswith("zh"):  # type: ignore[union-attr]
            lang_code = "zh"  # Both Simplified and Traditional Chinese
        elif detected_lang in ("en", "vi"):
            lang_code = detected_lang
        else:
            # Unsupported language, fall back to English patterns
            lang_code = "en"

        # Get language-specific patterns
        patterns = GREETING_PATTERNS_BY_LANGUAGE.get(
            lang_code, DEFAULT_EN_GREETING_PATTERNS
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
        for pattern in DEFAULT_EN_GREETING_PATTERNS:
            if re.match(pattern, query_lower):
                return True
        return False


def format_chat_history_for_rephrase(
    chat_history: List[Dict[str, Any]], max_messages: int = MAX_MSG_HISTORY
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
        if msg["role"] == USER_ROLE_NAME:
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == ASSISTANT_ROLE_NAME:
            messages.append(AIMessage(content=msg["content"]))

    return messages


def condense_question_with_history(
    current_question: str,
    chat_history: List[BaseMessage],
    llm: OllamaLLM,
    print_debug: bool = False,
) -> str:
    """
    Rephrase a question using chat history to make it standalone.

    Converts conversational questions like "What about the second point?" into
    "What about the second point regarding photosynthesis?" by referencing prior context.

    Args:
        current_question: The current user question
        chat_history: Formatted chat history (LangChain messages)
        llm: OllamaLLM instance
        print_debug: Enable debug logging

    Returns:
        Rephrased standalone version of the question
    """
    if not chat_history:
        # No history, return original question
        return current_question

    if print_debug:
        logger.info(
            f"🔄 Condensing question with {len(chat_history)} history messages..."
        )

    # Prompt that instructs the LLM to rephrase the question
    rephrase_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(  # type: ignore[misc]
        [
            (
                "system",
                "Given a chat history and the latest user question which might reference context from previous messages, "
                "formulate a standalone question which can be understood without the chat history. "
                "Keep it concise and preserve all important details from the original question. "
                "Do NOT answer the question, just rephrase it to be standalone.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Build the rephrasing chain
    rephrase_chain: Any = rephrase_prompt | llm  # type: ignore[operator]

    # Invoke with chat history and current question
    condensed: str = rephrase_chain.invoke(  # type: ignore[attr-defined]
        {
            "chat_history": chat_history,
            "input": current_question,
        }
    )

    if print_debug:
        logger.info(f"✅ Original: {current_question}")
        logger.info(f"✅ Condensed: {condensed}")

    return condensed


def get_enhanced_system_prompt() -> str:
    """
    Get enhanced system prompt with current time and general knowledge permission.

    Returns a modified LLM prompt that:
    1. Includes current timestamp for time-aware questions
    2. Permits the LLM to answer general questions from its own knowledge
    3. Maintains strict grounding for document-based questions

    Returns:
        Enhanced system prompt string
    """
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    enhanced_prompt = f"""You are {APP_NAME} - a helpful research assistant.
Current Time: {current_time}

ANSWER GUIDELINES:
1. **For Document Questions**: Use the provided context to answer. Prioritize it over general knowledge.
   - Look carefully through ALL context segments — the answer may be implicit or spread across multiple sections.
   - If the context contains relevant information (even indirect), use it to form your best answer.
   - Do NOT invent specific facts, numbers, names, or data that are not present in the context.
   - Cite source page numbers naturally when referencing specific information.

2. **For General Questions**: If the user asks about greetings, your capabilities, the current time/date, or general knowledge:
   - Answer politely using your general knowledge (no document context needed).
   - Be helpful and conversational naturally without searching documents.

3. **Language & Tone**:
   - Answer in the same language as the question (Vietnamese, English, etc.).
   - Be concise but thorough.

4. **Unknown Answers**:
   - Only say you cannot find the answer in documents if the context has NO information related to the topic.
   - For general questions, you can use your knowledge to provide a helpful response.

IMPORTANT — End your response with exactly ONE of these tags (no text after it):
- [FOUND_ANSWER: true]  — the context contained useful information to answer the question
- [FOUND_ANSWER: false] — the context had no relevant information, but you answered from general knowledge
- [FOUND_ANSWER: general] — the question was a greeting/general question answered without document context

CONTEXT FROM DOCUMENTS:
{{context}}

USER QUESTION: {{question}}

YOUR ANSWER:"""

    return enhanced_prompt


def create_history_aware_rag_chain(
    vectorstore: FAISS,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    print_debug: bool = False,
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
        chat_history: Optional list of previous messages from database
        print_debug: Enable debug logging

    Returns:
        Runnable chain that accepts {"input": question} and optional {"chat_history": [...]}
    """
    if print_debug:
        logger.info("\n🔗 Building History-Aware RAG Chain...")

    # Step 1: Create quality-based retriever
    def quality_retriever(query: str) -> List[Document]:
        return retrieve_quality_chunks(
            vectorstore,
            query,
            k=RAG_RETRIEVAL_K,
            min_results=RAG_RETRIEVAL_MIN_RESULTS,
            score_threshold=RAG_RETRIEVAL_SCORE_THRESHOLD,
            print_debug=print_debug,
        )

    # Step 2: Initialize LLM
    llm = OllamaLLM(
        model=LLM_MODEL_NAME,
        base_url=LLM_BASE_URL,
        temperature=LLM_TEMPERATURE,
        num_ctx=LLM_NUM_CTX,
    )

    # Step 3: Create history-aware retriever prompt with domain awareness
    # Optimized to better leverage chat history for question reformulation
    rephrase_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(  # type: ignore[misc]
        [
            (
                "system",
                REPHRASE_PROMPT,
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Step 4: Create the history-aware retriever using LangChain
    history_aware_retriever: Any = create_history_aware_retriever(  # type: ignore[call-arg]
        llm, vectorstore.as_retriever(), rephrase_prompt
    )

    # Step 5: Create enhanced prompt template with time and general knowledge permission
    enhanced_system_prompt: str = get_enhanced_system_prompt()
    qa_prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(
        enhanced_system_prompt
    )  # type: ignore[misc]

    # Step 6: Build the complete chain
    # If chat history is provided, use the history-aware flow
    if chat_history:
        formatted_history: List[BaseMessage] = format_chat_history_for_rephrase(
            chat_history
        )

        rag_chain: Any = (  # type: ignore[assignment]
            {
                "chat_history": RunnableLambda(lambda x: formatted_history),
                "context": RunnableLambda(
                    lambda x: format_context_with_sources(
                        docs=history_aware_retriever.invoke(  # type: ignore[attr-defined]
                            {
                                "input": str(x.get("input", ""))  # type: ignore[union-attr]
                                if isinstance(x, dict)
                                else "",
                                "chat_history": formatted_history,
                            }
                        ),
                        print_debug=print_debug,
                    )
                ),
                "question": RunnablePassthrough(),
            }
            | qa_prompt
            | llm
            | StrOutputParser()
        )
    else:
        rag_chain: Any = (  # type: ignore[assignment]
            {
                "context": RunnableLambda(
                    lambda query: format_context_with_sources(  # type: ignore[misc]
                        docs=quality_retriever(str(query)),
                        print_debug=print_debug,
                    )
                ),
                "question": RunnablePassthrough(),
            }
            | qa_prompt
            | llm
            | StrOutputParser()
        )

    if print_debug:
        logger.info("   ✅ History-Aware RAG Chain created!")
        logger.info("\n   Chain Flow:")
        logger.info(
            "   Question + History → Rephrase → Quality Retriever → Context Formatter → Enhanced Prompt → LLM → Answer"
        )

    return rag_chain
