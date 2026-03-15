"""
Centralized utility functions for RAG application.
Includes: PDF processing, chat handling, vectorstore management, and text cleaning.
"""

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
    ASSISTANT_ROLE_NAME,
    EMBEDDING_MODEL_NAME,
    LLM_BASE_URL,
    LLM_MODEL_NAME,
    LLM_NUM_CTX,
    LLM_PROMPT_TEMPLATE,
    LLM_TEMPERATURE,
    RAG_CHUNK_OVERLAP,
    RAG_MAX_CHUNK_LENGTH,
    RAG_MAX_CONTEXT_LENGTH,
    RAG_RETRIEVAL_K,
    RAG_RETRIEVAL_MIN_RESULTS,
    RAG_RETRIEVAL_SCORE_THRESHOLD,
    USER_ROLE_NAME,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
from middlewares import db_middleware as db


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
    print_debug: bool = False,
) -> tuple[str, List[Dict[str, Any]], bool]:
    """
    Process a user query through the RAG chain and retrieve relevant sources.

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

    # Generate answer through RAG chain (quality retriever runs internally here)
    if print_debug:
        logger.info("⏳ Invoking RAG chain with quality-filtered chunks...")

    answer = rag_chain.invoke(query)

    if print_debug:
        logger.info(f"✅ LLM response generated ({len(answer)} chars)")

    # Parse [FOUND_ANSWER: true/false] tag from answer
    found_answer = True  # Default to True if tag not found
    answer_clean = answer

    # Check for [FOUND_ANSWER: true] tag
    if "[FOUND_ANSWER: true]" in answer:
        found_answer = True
        answer_clean = answer.replace("[FOUND_ANSWER: true]", "").strip()
    # Check for [FOUND_ANSWER: false] tag
    elif "[FOUND_ANSWER: false]" in answer:
        found_answer = False
        answer_clean = answer.replace("[FOUND_ANSWER: false]", "").strip()

    if print_debug:
        if found_answer:
            logger.info("✅ LLM found relevant context - sources will be displayed")
        else:
            logger.warning(
                "⚠️  LLM did not find relevant context - sources will be hidden"
            )

    # Retrieve source documents for display using SAME quality filter as the RAG chain
    # This ensures display citations match exactly what the LLM used to generate the answer
    if print_debug:
        logger.info(
            "📎 Retrieving quality-filtered sources for display (matching RAG chain)..."
        )

    source_docs = retrieve_quality_chunks(
        vectorstore,
        query,
        k=RAG_RETRIEVAL_K,
        min_results=RAG_RETRIEVAL_MIN_RESULTS,
        score_threshold=RAG_RETRIEVAL_SCORE_THRESHOLD,
        print_debug=print_debug,
    )

    # Format sources for display
    sources: List[Dict[str, Any]] = []
    for i, doc in enumerate(source_docs, 1):
        source_entry = {  # pyright: ignore[reportUnknownVariableType]
            "document": doc.metadata.get("document", "Unknown"),  # pyright: ignore[reportUnknownMemberType]
            "page": doc.metadata.get("page", "N/A"),  # pyright: ignore[reportUnknownMemberType]
            "content": (
                doc.page_content[:200] + "..."
                if len(doc.page_content) > 200
                else doc.page_content
            ),
        }
        sources.append(source_entry)  # pyright: ignore[reportUnknownArgumentType]
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
