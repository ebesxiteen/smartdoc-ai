"""
Self-RAG (Self-Retrieval Augmented Generation) pipeline for SmartDoc AI.

This module implements the complete Self-RAG orchestration pipeline:
  Step 0  — Intent routing (greeting detection, short-circuit for general queries)
  Step 1  — Search planning (decompose query into 1-3 independent sub-queries)
  Step 2  — Hybrid retrieval with surgical retry (semantic + BM25, cross-encoder re-rank)
  Step 3  — Candidate generation (diverse drafts at varied temperatures)
  Step 4  — Quality scoring (ISSUP / ISREL / ISUSE judging)
  Step 5a — Threshold validation (accept winner if all gates pass)
  Step 5b — Repair agent (re-plan and retry up to max_depth times)

Public API consumed by core/utils.py:
  - self_rag_query()              — main orchestrator
  - create_history_aware_rag_chain() — builds LangChain-compatible RAG chain wrapper
  - SelfRAGState                  — per-query state dataclass
"""

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, cast, Dict, List, Optional

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import core.configs as cfg

# Lazy imports from core.utils avoid circular-dependency issues:
# self_rag.py → core.utils (one-way).  core.utils only imports from self_rag.py
# inside function bodies (lazy), keeping the module-level graph acyclic.
from core.utils import (
    load_notebook_settings,
    debug_log,
    format_context_with_sources,
    generate_fallback_answer,
    is_greeting,
    rerank_with_cross_encoder,
    retrieve_quality_chunks,
)


# ============================================================================
# SELF-RAG STATE
# ============================================================================


def _empty_str_list() -> List[str]:
    return []


def _empty_doc_list() -> "List[Document]":
    return []


def _empty_float_dict() -> Dict[str, float]:
    return {}


@dataclass
class SelfRAGState:
    """
    Mutable state object tracking Self-RAG query execution across recursive hops.

    This state persists throughout the lifecycle of a single user query, tracking:
    - Current recursion depth for repair attempts (vertical hops)
    - Search queries attempted to prevent oscillation (horizontal retries)
    - Cumulative document chunks retrieved across all hops
    - Verbose decision trace for UI transparency
    - Original reformulated query for consistent retrieval

    Attributes:
        current_depth (int): Current recursion depth (0 = initial attempt, incremented per repair)
        search_history (List[str]): All previously generated sub-queries to prevent repeating failed searches
        retrieval_pool (List[Document]): Deduplicated cumulative chunks from all retrieval attempts across hops
        reasoning_trace (List[str]): Verbose log of decisions for debugging and UI transparency
        original_query (str): Contextually reformulated standalone query, reused for consistent retrieval
        confidence_metrics (Dict[str, float]): Scores from the last validation gate
    """

    current_depth: int = 0
    search_history: List[str] = field(default_factory=_empty_str_list)
    retrieval_pool: List[Document] = field(default_factory=_empty_doc_list)
    reasoning_trace: List[str] = field(default_factory=_empty_str_list)
    original_query: str = ""
    confidence_metrics: Dict[str, float] = field(default_factory=_empty_float_dict)


# ============================================================================
# STEP 0 — INTENT ROUTING
# ============================================================================


def reformulate_query_with_history(
    query: str,
    chat_history: List[Any],
    notebook_id: Optional[str] = None,
    print_debug: bool = False,
    settings: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Reformulate a follow-up question into a standalone question using chat history.

    Resolves pronouns and context references in the query to create a self-contained
    question that works well with vector search.

    Example:
        "What about him?" + history containing "Alice" -> "What about Alice?"

    Args:
        query: The original follow-up query.
        chat_history: List of chat messages for context.
        notebook_id: Notebook ID for loading custom settings (LLM model, etc.).
        print_debug: Whether to print debug logs.
        settings: Pre-loaded notebook settings dict. If provided, skips the DB
                  lookup, avoiding a redundant round-trip when the caller already
                  has settings loaded.

    Returns:
        str: Reformulated standalone query (or original if reformulation skipped).
    """
    if not chat_history:
        return query

    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_ollama import OllamaLLM

        settings = settings or load_notebook_settings(notebook_id)
        llm = OllamaLLM(
            model=settings["llm_model_name"],
            base_url=cfg.LLM_BASE_URL,
            temperature=0.1,  # Keep it deterministic
        )

        # Format chat history using notebook-configured max_msg_history
        _max_hist = max(1, int(settings["max_msg_history"]))
        history_str = "\n".join(
            [
                f"- {(msg.get('role', 'user') if hasattr(msg, 'get') else getattr(msg, 'type', 'user')).upper()}: "
                f"{(msg.get('content', '') if hasattr(msg, 'get') else getattr(msg, 'content', ''))}"
                for msg in chat_history[-_max_hist:]
            ]
        )

        prompt = ChatPromptTemplate.from_template(cfg.REFORMULATE_QUERY_PROMPT)
        chain: Any = cast(Any, prompt | llm)

        response = chain.invoke({"query": query, "chat_history": history_str})

        reformulated = str(response).strip()

        # Don't return empty strings or hallucinated long texts
        if not reformulated or len(reformulated) > len(query) + 200:
            return query

        if print_debug and reformulated != query:
            debug_log(
                "INFO",
                "🧠",
                f'Contextual Query Reformulation: "{query}"" -> "{reformulated}"',
            )

        return reformulated

    except Exception as e:
        if print_debug:
            debug_log(
                "WARNING",
                "🧠",
                f"Reformulation failed, using original. Error: {str(e)[:100]}",
            )
        return query


def _validate_greeting_with_llm(
    query: str,
    llm_chain: Any = None,
    print_debug: bool = False,
) -> bool:
    """
    Layer 2: LLM-based greeting detection (fallback when regex misses greetings).

    If regex detection missed a greeting, use LLM to validate intent.
    Example: "Hiya!" (informal) → Regex might miss, but LLM catches it.

    Args:
        query: User query to validate.
        llm_chain: Optional LLM chain for validation (initializes a new one if None).
        print_debug: Whether to print debug logs.

    Returns:
        bool: True if detected as greeting/chitchat, False if factual question.
    """
    try:
        # If no LLM chain provided, initialize a simple one
        if llm_chain is None:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_ollama import OllamaLLM

            llm = OllamaLLM(
                model=cfg.LLM_MODEL_NAME,
                base_url=cfg.LLM_BASE_URL,
                temperature=0.1,  # Low temp for deterministic classification
            )
            prompt = ChatPromptTemplate.from_template(cfg.LAYER2_LLM_ROUTER_PROMPT)
            llm_chain = cast(Any, prompt | llm)

        response = llm_chain.invoke({"query": query})
        classification = str(response).strip().upper()

        is_greeting_result = "GREETING" in classification

        if print_debug:
            debug_log(
                "INFO",
                "🎯",
                f'Layer 2 LLM Router | Query: "{query}" | Result: {classification}',
            )

        return is_greeting_result

    except Exception as e:
        # Graceful fallback to False (proceed with RAG) if LLM check fails
        if print_debug:
            debug_log("WARNING", "⚠️", f"Layer 2 LLM Router failed: {str(e)[:100]}")
        return False


def perform_intent_routing(
    query: str,
    notebook_id: Optional[str] = None,
    print_debug: bool = False,
) -> tuple[bool, str]:
    """
    Step 0: Two-layer intent routing to detect greetings and short-circuit RAG.

    Pipeline:
    1. Layer 1 (Regex): Ultra-fast regex matching against greeting patterns.
    2. Layer 2 (LLM): Fallback to LLM-based validation if regex misses.

    If a greeting is detected, skip document retrieval and return GENERAL status.

    Args:
        query: User query to route.
        notebook_id: Notebook ID for loading settings (unused in routing, kept for API consistency).
        print_debug: Whether to print debug logs.

    Returns:
        tuple: (is_greeting, routing_decision_description)
    """
    _ = notebook_id  # Reserved for future per-notebook greeting overrides

    # Layer 1: Regex-based greeting detection
    layer1_result = is_greeting(query)

    if layer1_result:
        if print_debug:
            debug_log(
                "INFO",
                "🎯",
                f"Step 0 Intent Routing | Layer 1 Regex Match | Query: '{query[:80]}'",
            )
        return True, "Layer 1 Regex: Greeting detected → Skip RAG → GENERAL status"

    # Layer 2: LLM-based fallback validation
    layer2_result = _validate_greeting_with_llm(
        query, llm_chain=None, print_debug=print_debug
    )

    if layer2_result:
        if print_debug:
            debug_log(
                "INFO",
                "🎯",
                f"Step 0 Intent Routing | Layer 2 LLM Match | Query: '{query[:80]}'",
            )
        return True, "Layer 2 LLM: Greeting detected → Skip RAG → GENERAL status"

    # No greeting detected — proceed with RAG pipeline
    if print_debug:
        debug_log(
            "INFO",
            "🎯",
            "Step 0 Intent Routing | FACTUAL Query | Proceeding to RAG pipeline",
        )
    return False, "No greeting detected → Proceed to RAG pipeline"


# ============================================================================
# STATE INITIALIZATION
# ============================================================================


def initialize_self_rag_state(
    query: str,
    notebook_id: Optional[str] = None,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    print_debug: bool = False,
) -> "SelfRAGState":
    """
    Initialize a fresh SelfRAGState for a new query execution.

    Steps:
    1. Reformulates the raw query using chat history for contextual resolution (if history available).
    2. Returns an initialized state ready for the Self-RAG pipeline.

    Note: Notebook settings are loaded in the main orchestrator (self_rag_query) for
    efficiency. This function only handles query reformulation and state setup.

    Args:
        query: Raw user query to process.
        notebook_id: Notebook ID to load custom settings (thresholds, etc.).
        chat_history: Optional chat history for contextual reformulation.
        print_debug: Whether to print debug logs.

    Returns:
        SelfRAGState: Initialized state with reformulated query and zeroed counters.

    Examples:
        >>> state = initialize_self_rag_state("What about it?", notebook_id="xyz",
        ...                                   chat_history=[{"role":"user","content":"Tell me about RAG"}])
        >>> print(state.original_query)  # "What about RAG?"
    """
    # Reformulate query using chat history for context-aware standalone question
    if chat_history:
        original_query = reformulate_query_with_history(
            query, chat_history, notebook_id, print_debug
        )
    else:
        original_query = query

    if print_debug:
        debug_log(
            "INFO",
            "🚀",
            f"Initialized Self-RAG State | Depth: 0 | Query: {original_query[:80]}...",
        )

    return SelfRAGState(
        current_depth=0,
        search_history=[],
        retrieval_pool=[],
        reasoning_trace=[],
        original_query=original_query,
    )


# ============================================================================
# STEP 1 — SEARCH PLANNING
# ============================================================================


def generate_search_plan(
    original_query: str,
    llm_chain: Any,
    current_depth: int = 0,
    search_history: Optional[List[str]] = None,
    print_debug: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> List[str]:
    """
    Step 1: Decompose the query into 1-3 independent sub-queries for retrieval.

    The planner breaks a complex question into independently retrievable sub-queries.
    At depth > 0 (repair hops), it includes failed search history so the repair
    agent can pivot to a different angle.

    Args:
        original_query: The reformulated standalone question.
        llm_chain: LLM chain for sub-query generation.
        current_depth: Current recursion depth (0 = first attempt).
        search_history: Previously attempted sub-queries (to avoid oscillation).
        print_debug: Whether to print debug logs.
        progress_callback: Optional callback for progress bar updates.

    Returns:
        List[str]: 1-3 independent sub-queries (max).
    """
    if search_history is None:
        search_history = []

    if progress_callback:
        progress_callback("🎯 Step 1: Planning search queries...")

    if search_history and current_depth > 0:
        # Include search history for the repair agent to pivot strategy
        prompt_input = (
            f"{original_query}\n\nPrevious failed searches: {'; '.join(search_history)}"
        )
    else:
        prompt_input = original_query

    try:
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_template(cfg.SEARCH_PLANNER_PROMPT)
        chain = prompt | llm_chain

        response = chain.invoke({"original_query": prompt_input})
        sub_queries_text = str(response).strip()

        # Parse sub-queries (one per line, no numbering)
        sub_queries = [sq.strip() for sq in sub_queries_text.split("\n") if sq.strip()]

        if print_debug:
            debug_log(
                "INFO",
                "🏗️",
                f"Step 1 Planner | Generated {len(sub_queries)} sub-queries | Depth: {current_depth}",
            )

        return sub_queries[:3]  # Max 3 sub-queries

    except Exception as e:
        if print_debug:
            debug_log("WARNING", "⚠️", f"Step 1 Planner failed: {str(e)[:100]}")
        # Fallback: return original query as single sub-query
        return [original_query]


# ============================================================================
# STEP 2 — HYBRID RETRIEVAL WITH SURGICAL RETRY
# ============================================================================


def retrieve_with_surgical_retry(
    sub_queries: List[str],
    vectorstore: FAISS,
    notebook_id: Optional[str] = None,
    state: Optional[SelfRAGState] = None,
    llm_chain: Any = None,
    print_debug: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> tuple[List[Document], SelfRAGState]:
    """
    Step 2: Hybrid retrieval with surgical horizontal retry for failing sub-queries.

    For each sub-query:
    1. Execute hybrid search (semantic + BM25).
    2. Apply threshold filtering.
    3. If 0 results, retry with LLM-rewritten query (up to max_retries).
    4. Merge and deduplicate across all sub-queries.
    5. Apply cross-encoder re-ranking.

    Args:
        sub_queries: List of independent sub-queries.
        vectorstore: FAISS vectorstore for retrieval.
        notebook_id: Notebook ID for loading settings.
        state: Current Self-RAG state to accumulate results.
        llm_chain: LLM chain for sub-query rewriting on retry.
        print_debug: Whether to print debug logs.
        progress_callback: Optional callback for progress bar updates.

    Returns:
        tuple: (retrieved_documents, updated_state)
    """
    if state is None:
        state = SelfRAGState()

    if progress_callback:
        progress_callback("📚 Step 2: Retrieving documents...")

    settings = load_notebook_settings(notebook_id)
    max_retries = settings["self_rag_max_retries_per_hop"]

    # Use dict keyed on content hash for O(1) deduplication
    all_docs: Dict[int, Document] = {}
    if state and state.retrieval_pool:
        for doc in state.retrieval_pool:
            all_docs[hash(doc.page_content[:100])] = doc
    retry_count = 0

    for i, sub_query in enumerate(sub_queries):
        state.search_history.append(sub_query)
        attempted_retries = 0
        current_sq = sub_query

        while attempted_retries <= max_retries:
            try:
                if print_debug:
                    debug_log(
                        "INFO",
                        "📖",
                        f'Step 2 Retrieval | Sub-Q {i + 1}/{len(sub_queries)}: "{current_sq}"',
                    )

                # Stage 1: Hybrid search (semantic + BM25)
                # Fetch more chunks initially for stage-2 cross-encoder re-ranking
                docs = retrieve_quality_chunks(
                    vectorstore,
                    current_sq,
                    k=settings["rag_rerank_top_n"],
                    score_threshold=settings["rag_retrieval_score_threshold"],
                    print_debug=print_debug,
                    weight_semantic=float(settings["weight_semantic"]),
                    weight_bm25=float(settings["weight_bm25"]),
                )

                if docs:
                    for doc in docs:
                        doc_hash = hash(doc.page_content[:100])
                        if doc_hash not in all_docs:
                            all_docs[doc_hash] = doc
                    break  # Success — move to next sub-query

                # Zero results — check retry budget
                if attempted_retries < max_retries:
                    attempted_retries += 1
                    retry_count += 1
                    state.reasoning_trace.append(
                        f"Sub-query {i + 1} returned 0 results, retrying "
                        f"(attempt {attempted_retries}/{max_retries})"
                    )

                    # LLM-based sub-query rewrite: semantically different angle
                    if llm_chain is not None:
                        try:
                            from langchain_core.prompts import (
                                ChatPromptTemplate as _CPT,
                            )

                            _rewrite_chain = (
                                _CPT.from_template(cfg.SUBQUERY_REWRITE_PROMPT)
                                | llm_chain
                            )
                            _success_ctx = (
                                " ".join(
                                    doc.page_content[:80]
                                    for doc in list(all_docs.values())[:2]
                                )
                                or "None"
                            )
                            _response = _rewrite_chain.invoke(
                                {
                                    "original_query": state.original_query
                                    if state
                                    else sub_query,
                                    "failed_subquery": current_sq,
                                    "success_context": _success_ctx,
                                }
                            )
                            _rewritten = str(_response).strip()
                            current_sq = (
                                _rewritten
                                if _rewritten and _rewritten != current_sq
                                else f"{sub_query} alternative approach"
                            )
                        except Exception:
                            current_sq = f"{sub_query} alternative approach"
                    else:
                        current_sq = f"{sub_query} alternative approach"

                    if print_debug:
                        debug_log(
                            "INFO",
                            "🔄",
                            f"Step 2 Retrieval | Sub-Q {i + 1} retry | "
                            f"Attempt {attempted_retries}/{max_retries}",
                        )
                else:
                    # Retries exhausted
                    state.reasoning_trace.append(
                        f"Sub-query {i + 1} failed after {max_retries} retries, "
                        f"moving forward with 0 results"
                    )
                    if print_debug:
                        debug_log(
                            "WARNING",
                            "⚠️",
                            f"Step 2 Retrieval | Sub-Q {i + 1} exhausted retries | Moving forward",
                        )
                    break  # Move to next sub-query

            except Exception as e:
                state.reasoning_trace.append(
                    f"Sub-query {i + 1} retrieval error: {str(e)[:50]}"
                )
                if print_debug:
                    debug_log("WARNING", "⚠️", f"Step 2 Retrieval error: {str(e)[:100]}")
                break  # Move to next sub-query

    # Merge and deduplicate
    state.retrieval_pool = list(all_docs.values())

    if print_debug:
        debug_log(
            "INFO",
            "📊",
            f"Step 2 Stage 1 Complete | Total unique pooled chunks: "
            f"{len(state.retrieval_pool)} | Retries: {retry_count}",
        )

    # Stage 2: Cross-Encoder re-ranking
    # Reduces the large `rag_rerank_top_n` pool down to `rag_final_context_k` most relevant items
    final_k = settings["rag_final_context_k"]
    state.retrieval_pool = rerank_with_cross_encoder(
        query=state.original_query,  # Use original query as the benchmark
        candidates=state.retrieval_pool,
        top_k=final_k,
        print_debug=print_debug,
    )

    if print_debug:
        debug_log(
            "INFO",
            "📊",
            f"Step 2 Stage 2 Complete | Reranked down to top "
            f"{len(state.retrieval_pool)} final chunks.",
        )

    return state.retrieval_pool, state


# ============================================================================
# STEP 3 — CANDIDATE GENERATION
# ============================================================================


def generate_candidate_answers(
    original_query: str,
    search_plan: List[str],
    context_docs: List[Document],
    llm_chain: Any,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    notebook_id: Optional[str] = None,
    max_candidates: int = 3,
    print_debug: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Step 3: Generate `max_candidates` diverse draft answers from retrieved context.

    Each candidate is generated with a slightly varied temperature to create diversity
    in perspective and writing style. Diversity hints steer the LLM toward different
    response strategies (concise, step-by-step, contextual, metrics-focused).

    Args:
        original_query: The reformulated standalone question.
        search_plan: Sub-queries used during retrieval (reserved for future use).
        context_docs: Top-K retrieved and re-ranked documents.
        llm_chain: LLM chain for generation (temperature varied per candidate).
        chat_history: Optional chat history for context injection.
        notebook_id: Notebook ID for loading settings.
        max_candidates: Number of diverse candidates to generate.
        print_debug: Whether to print debug logs.
        progress_callback: Optional callback for progress bar updates.

    Returns:
        List[Dict[str, Any]]: Generated candidate dicts with keys:
            - answer (str): Clean answer text (without [FOUND:] tag)
            - generator_found_answer (bool): True if LLM tagged [FOUND: YES]
    """
    _ = search_plan  # Reserved for future use

    settings = load_notebook_settings(notebook_id)

    if progress_callback:
        progress_callback("✍️ Step 3: Generating candidate answers...")

    context_str = format_context_with_sources(context_docs, print_debug=False)

    # Format chat history if available
    chat_history_str = ""
    max_history = int(settings["max_msg_history"])
    if chat_history and max_history > 0:
        chat_history_str = "\n".join(
            [
                f"- {(msg.get('role', 'user') if hasattr(msg, 'get') else getattr(msg, 'type', 'user')).upper()}: "
                f"{(msg.get('content', '') if hasattr(msg, 'get') else getattr(msg, 'content', ''))}"
                for msg in chat_history[-max_history:]
            ]
        )

    candidates: List[Dict[str, Any]] = []

    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_ollama import OllamaLLM

        current_time = datetime.now(timezone.utc).strftime("%A, %B %d, %Y at %H:%M UTC")
        personal_ctx = settings["personal_ctx"]

        system_prompt_str = cfg.get_self_rag_system_prompt(
            personal_ctx=personal_ctx,
            current_time=current_time,
        )

        # Diversity hints steer each candidate toward a different response strategy
        diversity_hints = [
            "Focus on a direct, concise explanation.",
            "Provide a detailed, step-by-step breakdown.",
            "Emphasize the overarching context and implications.",
            "Highlight key metrics or specific facts primarily.",
        ]

        for i in range(max_candidates):
            if progress_callback:
                progress_callback(f"✍️ Generating candidate {i + 1}/{max_candidates}...")

            hint = diversity_hints[i % len(diversity_hints)]

            user_template = f"""QUESTION: {{query}}

RETRIEVED CONTEXT:
{{context}}

CHAT HISTORY:
{{chat_history}}
INSTRUCTION: Based on the context above AND the CHAT HISTORY, provide a complete answer to the question. You MUST use the chat history as a valid factual source for personal/conversational queries if the document context doesn't contain the answer.
APPROACH: {hint}"""

            prompt: Any = cast(
                Any,
                ChatPromptTemplate.from_messages(  # pyright: ignore[reportUnknownMemberType]
                    [
                        ("system", system_prompt_str),
                        ("user", user_template),
                    ]
                ),
            )

            # Recreate LLM with slightly varied temperature for true candidate diversity
            base_temp = settings["llm_avg_temp"]
            var_temp = base_temp + (i * 0.05)

            try:
                temp_llm = OllamaLLM(
                    model=settings["llm_model_name"],
                    base_url=cfg.LLM_BASE_URL,
                    temperature=var_temp,
                )
                chain: Any = prompt | temp_llm
            except Exception:
                # Fallback to original chain if Ollama init fails
                chain = prompt | llm_chain

            response = chain.invoke(
                {
                    "query": original_query,
                    "context": context_str,
                    "chat_history": chat_history_str,
                }
            )

            answer = str(response).strip()

            # Capture the generator LLM's own [FOUND: YES/NO] tag before stripping it.
            # This tag is the source of truth for found_answer (whether citations are shown).
            found_match = re.search(r"\[FOUND:\s*(YES|NO)\]", answer, re.IGNORECASE)
            generator_found_answer = (
                found_match.group(1).upper() == "YES" if found_match else False
            )

            # Strip [FOUND:] tag and any legacy [STATUS:] tags from clean answer
            answer = re.sub(
                r"\[FOUND:\s*(?:YES|NO)\]", "", answer, flags=re.IGNORECASE
            ).strip()
            answer = re.sub(r"\[STATUS:[^\]]*\]", "", answer).strip()

            candidates.append(
                {"answer": answer, "generator_found_answer": generator_found_answer}
            )

            if print_debug:
                debug_log(
                    "INFO",
                    "✍️",
                    f"Step 3 Generator | Candidate {i + 1}/{max_candidates} | "
                    f"Length: {len(answer)} chars | FOUND: {'YES' if generator_found_answer else 'NO'}",
                )

        return candidates

    except Exception as e:
        if print_debug:
            debug_log("WARNING", "⚠️", f"Step 3 Generator failed: {str(e)[:100]}")
        return []


# ============================================================================
# STEP 4 — QUALITY SCORING
# ============================================================================


def score_candidates_with_judges(
    original_query: str,
    candidates: List[str],
    context_docs: List[Document],
    llm_chain: Any,
    print_debug: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
    notebook_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Step 4: Score all candidates on ISSUP (groundedness) and ISUSE (utility).

    Uses LLM to evaluate each candidate and produce structured JSON scores.
    ISREL is derived from cross-encoder scores stored in document metadata.

    Total score formula: 0.25 * ISREL + 0.50 * ISSUP + 0.25 * ISUSE

    Args:
        original_query: The reformulated question.
        candidates: List of generated candidate answers.
        context_docs: Retrieved documents for context verification.
        llm_chain: LLM chain for scoring.
        print_debug: Whether to print debug logs.
        progress_callback: Optional callback for progress bar updates.
        notebook_id: Notebook ID for loading quality thresholds.

    Returns:
        List[Dict]: Scored candidates with keys:
            answer, issup, isrel, isuse, total_score, reasoning, status
    """
    if progress_callback:
        progress_callback("📊 Step 4: Scoring candidates...")

    # Use top 5 context docs for judge (keeps prompt concise)
    context_str = format_context_with_sources(context_docs[:5], print_debug=False)

    scored_candidates: List[Dict[str, Any]] = []

    settings = load_notebook_settings(notebook_id)
    issup_threshold = settings["self_rag_threshold_issup"]
    isrel_threshold = settings["self_rag_threshold_isrel"]
    isuse_threshold = settings["self_rag_threshold_isuse"]

    try:
        import json

        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_template(cfg.QUALITY_JUDGE_PROMPT)
        chain = prompt | llm_chain

        for i, candidate in enumerate(candidates):
            if progress_callback:
                progress_callback(f"📊 Scoring candidate {i + 1}/{len(candidates)}...")

            response = chain.invoke(
                {
                    "query": original_query,
                    "answer": candidate,
                    "context": context_str,
                }
            )

            try:
                response_str = str(response).strip()
                # Extract JSON from response (handle markdown code blocks)
                json_match = re.search(r"\{.*\}", response_str, re.DOTALL)
                if json_match:
                    scores_dict = json.loads(json_match.group())
                else:
                    raise ValueError("No JSON found in response")

                issup = float(scores_dict.get("issup", 0.5))
                isuse = float(scores_dict.get("isuse", 0.5))

                # ISREL is inherited from cross-encoder scores (raw logits → sigmoid → [0,1])
                # Sigmoid ensures ISREL is commensurate with ISSUP/ISUSE for threshold checks.
                raw_ce_scores: List[float] = [
                    float(
                        cast(Dict[str, Any], getattr(doc, "metadata")).get(
                            "rerank_score", 0.0
                        )
                    )
                    for doc in context_docs
                ]
                normalized_ce_scores = [
                    1.0 / (1.0 + math.exp(-s)) for s in raw_ce_scores
                ]
                isrel = (
                    sum(normalized_ce_scores) / len(normalized_ce_scores)
                    if normalized_ce_scores
                    else 0.5
                )

                total_score = 0.25 * isrel + 0.50 * issup + 0.25 * isuse

                # Assign categorical status for tie-breaking and final routing
                if not context_docs:
                    # No retrieved context — answer from general knowledge / chat history
                    candidate_status = "DOC_GENERAL"
                elif (
                    issup >= issup_threshold
                    and isuse >= isuse_threshold
                    and isrel >= isrel_threshold
                ):
                    candidate_status = "DOC_ANSWER"
                else:
                    # Context retrieved but answer failed quality thresholds
                    candidate_status = "DOC_MISSING"

                scored_candidates.append(
                    {
                        "answer": candidate,
                        "issup": issup,
                        "isrel": isrel,
                        "isuse": isuse,
                        "total_score": total_score,
                        "reasoning": scores_dict.get("reasoning", ""),
                        "status": candidate_status,
                    }
                )

                if print_debug:
                    debug_log(
                        "INFO",
                        "📊",
                        f"Step 4 Judge | Candidate {i + 1} | Score: {total_score:.2f} "
                        f"(ISSUP:{issup:.2f}, ISUSE:{isuse:.2f})",
                    )

            except Exception:
                # Fallback: assign default scores
                scored_candidates.append(
                    {
                        "answer": candidate,
                        "issup": 0.5,
                        "isrel": 0.5,
                        "isuse": 0.5,
                        "total_score": 0.5,
                        "reasoning": "Scoring failed, using default",
                        "status": "DOC_MISSING",
                    }
                )
                if print_debug:
                    debug_log(
                        "WARNING",
                        "⚠️",
                        f"Step 4 Judge | JSON parse failed for candidate {i + 1}",
                    )

        # Print comprehensive scoring summary if debug enabled.
        # Avoid calling pick_winner here — it would trigger the tie-breaking log prematurely.
        # The real pick_winner call happens in the orchestrator (self_rag_query).
        if print_debug and scored_candidates:
            best_score = max(c.get("total_score", 0) for c in scored_candidates)
            winner_dict = next(
                (c for c in scored_candidates if c.get("total_score", 0) == best_score),
                scored_candidates[0],
            )
            debug_log("INFO", "📊", "═" * 70)
            debug_log("INFO", "📊", "STEP 4: CANDIDATE SCORING SUMMARY")
            debug_log("INFO", "📊", "─" * 70)
            for idx, candidate in enumerate(scored_candidates):
                winner_mark = " ← WINNER" if candidate is winner_dict else ""
                debug_log(
                    "INFO",
                    "📊",
                    f"Candidate {idx + 1}: Score {candidate['total_score']:.2f} | "
                    f"ISSUP: {candidate['issup']:.2f} | ISREL: {candidate['isrel']:.2f} | "
                    f"ISUSE: {candidate['isuse']:.2f}{winner_mark}",
                )
            debug_log("INFO", "📊", "═" * 70)

        return scored_candidates

    except Exception as e:
        if print_debug:
            debug_log("WARNING", "⚠️", f"Step 4 Judge failed: {str(e)[:100]}")
        # Return candidates with default scores
        return [
            {"answer": c, "issup": 0.5, "isrel": 0.5, "isuse": 0.5, "total_score": 0.5}
            for c in candidates
        ]


# ============================================================================
# STEP 5 — WINNER SELECTION & VALIDATION
# ============================================================================


def pick_winner(scored_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Select the best candidate using a deterministic tie-breaking hierarchy.

    Tie-Breaking Order:
    1. Highest total_score (weighted combination of ISSUP, ISREL, ISUSE).
    2. STATUS priority: DOC_ANSWER (3) > DOC_GENERAL (2) > DOC_MISSING (1).
    3. Higher ISSUP (groundedness wins if status is tied).
    4. First candidate in list (index fallback for stability).

    Args:
        scored_candidates: List of scored candidate dicts from score_candidates_with_judges().

    Returns:
        Dict[str, Any]: The winning candidate dict (or {} if list is empty).
    """
    if not scored_candidates:
        return {}

    STATUS_RANKS = {
        "DOC_ANSWER": 3,
        "DOC_GENERAL": 2,
        "DOC_MISSING": 1,
    }

    # Annotate with original index for stable tie-breaking
    for idx, c in enumerate(scored_candidates):
        c["_index"] = idx

    sorted_candidates = sorted(
        scored_candidates,
        key=lambda x: (
            x.get("total_score", 0),
            STATUS_RANKS.get(x.get("status", ""), 0),
            x.get("issup", 0.0),
            -x.get("_index", 0),
        ),
        reverse=True,
    )

    winner = sorted_candidates[0]

    # Log tie-breaking info when multiple candidates share the top score
    if len(scored_candidates) > 1:
        top_score = winner.get("total_score", 0)
        tied_count = sum(
            1 for c in scored_candidates if c.get("total_score", 0) == top_score
        )

        if tied_count > 1:
            runner_up = sorted_candidates[1]

            if STATUS_RANKS.get(winner.get("status", ""), 0) != STATUS_RANKS.get(
                runner_up.get("status", ""), 0
            ):
                tie_breaker = f"STATUS priority ({winner.get('status', 'UNKNOWN')})"
            elif winner.get("issup", 0.0) != runner_up.get("issup", 0.0):
                tie_breaker = (
                    f"Higher ISSUP "
                    f"({winner.get('issup', 0):.2f} vs {runner_up.get('issup', 0):.2f})"
                )
            else:
                tie_breaker = "Index fallback"

            debug_log(
                "INFO",
                "🎯",
                f"Tie-breaking: {tied_count} candidates tied on score "
                f"0.{int(top_score * 100)} → Winner by {tie_breaker}",
            )

    return winner


def validate_winner_against_thresholds(
    winner: Dict[str, Any],
    notebook_id: Optional[str] = None,
    state: Optional[SelfRAGState] = None,
    print_debug: bool = False,
) -> tuple[bool, str]:
    """
    Step 5a: Validate the winner against all three quality gates.

    Gates checked:
    - ISSUP >= self_rag_threshold_issup (groundedness)
    - ISREL >= self_rag_threshold_isrel (relevance)
    - ISUSE >= self_rag_threshold_isuse (utility)

    Args:
        winner: The highest-scoring candidate from pick_winner().
        notebook_id: Notebook ID for loading per-notebook thresholds.
        state: Self-RAG state for appending the decision to reasoning_trace.
        print_debug: Whether to print debug logs.

    Returns:
        tuple: (passed: bool, decision_reason: str describing all gate outcomes)
    """
    if state is None:
        state = SelfRAGState()

    settings = load_notebook_settings(notebook_id)

    issup_threshold = settings["self_rag_threshold_issup"]
    isrel_threshold = settings["self_rag_threshold_isrel"]
    isuse_threshold = settings["self_rag_threshold_isuse"]

    issup = winner.get("issup", 0.0)
    isrel = winner.get("isrel", 0.0)
    isuse = winner.get("isuse", 0.0)

    passed_issup = issup >= issup_threshold
    passed_isrel = isrel >= isrel_threshold
    passed_isuse = isuse >= isuse_threshold
    all_passed = passed_issup and passed_isrel and passed_isuse

    decision_reason = (
        f"ISSUP: {issup:.2f} vs {issup_threshold:.2f} {'✓' if passed_issup else '✗'} | "
        f"ISREL: {isrel:.2f} vs {isrel_threshold:.2f} {'✓' if passed_isrel else '✗'} | "
        f"ISUSE: {isuse:.2f} vs {isuse_threshold:.2f} {'✓' if passed_isuse else '✗'}"
    )

    if print_debug:
        debug_log("INFO", "═" * 70)
        debug_log("INFO", "STEP 5a: QUALITY GATE VALIDATION")
        debug_log("INFO", "─" * 70)
        debug_log(
            "INFO",
            f"ISSUP (Groundedness): {issup:.2f} >= {issup_threshold:.2f}  "
            f"{'✓ PASS' if passed_issup else '✗ FAIL'}",
        )
        debug_log(
            "INFO",
            f"ISREL (Relevance):    {isrel:.2f} >= {isrel_threshold:.2f}  "
            f"{'✓ PASS' if passed_isrel else '✗ FAIL'}",
        )
        debug_log(
            "INFO",
            f"ISUSE (Usefulness):   {isuse:.2f} >= {isuse_threshold:.2f}  "
            f"{'✓ PASS' if passed_isuse else '✗ FAIL'}",
        )
        debug_log("INFO", "─" * 70)
        if all_passed:
            debug_log("INFO", "✅ ALL GATES PASSED → Winner accepted!")
        else:
            debug_log(
                "INFO", "❌ GATE FAILED → Proceed to repair agent (if depth available)"
            )
        debug_log("INFO", "═" * 70)

    state.reasoning_trace.append(f"Step 5a Validation: {decision_reason}")
    state.confidence_metrics = {
        "issup": issup,
        "isrel": isrel,
        "isuse": isuse,
        "total_score": winner.get("total_score", 0.0),
    }

    return all_passed, decision_reason


# ============================================================================
# STEP 5b — REPAIR AGENT
# ============================================================================


def repair_failed_answer(
    winner: Dict[str, Any],
    state: SelfRAGState,
    original_query: str,
    llm_chain: Any,
    notebook_id: Optional[str] = None,
    print_debug: bool = False,
) -> List[str]:
    """
    Step 5b: Diagnose a failed winner and generate a new search strategy.

    When the winner fails quality thresholds:
    1. Identifies which thresholds failed.
    2. Diagnoses the root cause.
    3. Uses the repair agent prompt to pivot to a different retrieval angle.
    4. Returns new sub-queries for the next repair hop.

    Args:
        winner: The failed winning candidate.
        state: Current Self-RAG state (includes search_history to prevent oscillation).
        original_query: The reformulated standalone question.
        llm_chain: LLM chain for repair analysis.
        notebook_id: Notebook ID for loading quality thresholds.
        print_debug: Whether to print debug logs.

    Returns:
        List[str]: New sub-queries for the next repair hop (max 3).
    """
    settings = load_notebook_settings(notebook_id)
    issup_threshold = settings["self_rag_threshold_issup"]
    isrel_threshold = settings["self_rag_threshold_isrel"]
    isuse_threshold = settings["self_rag_threshold_isuse"]

    failures: List[str] = []
    if winner.get("issup", 0.0) < issup_threshold:
        failures.append(
            f"ISSUP (Groundedness): {winner.get('issup', 0):.2f} < {issup_threshold:.2f}"
        )
    if winner.get("isrel", 0.0) < isrel_threshold:
        failures.append(
            f"ISREL (Relevance): {winner.get('isrel', 0):.2f} < {isrel_threshold:.2f}"
        )
    if winner.get("isuse", 0.0) < isuse_threshold:
        failures.append(
            f"ISUSE (Utility): {winner.get('isuse', 0):.2f} < {isuse_threshold:.2f}"
        )

    failure_reason = " AND ".join(failures) if failures else "Unknown failure"
    state.reasoning_trace.append(f"Step 5b Repair: Failed because {failure_reason}")

    try:
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_template(cfg.REPAIR_AGENT_PROMPT)
        chain = prompt | llm_chain

        search_history_str = (
            "; ".join(state.search_history) if state.search_history else "None"
        )

        response = chain.invoke(
            {
                "original_query": original_query,
                "failed_answer": winner.get("answer", "")[:200],
                "failure_reason": failure_reason,
                "search_history": search_history_str,
            }
        )

        new_strategy_text = str(response).strip()

        # Parse new sub-queries (one per line, strip optional numbering)
        new_sub_queries = [
            sq.strip() for sq in new_strategy_text.split("\n") if sq.strip()
        ]
        new_sub_queries = [re.sub(r"^\d+\.\s*", "", sq) for sq in new_sub_queries]
        new_sub_queries = [sq for sq in new_sub_queries if sq][:3]  # Max 3

        if print_debug:
            debug_log(
                "INFO",
                "🔧",
                f"Step 5b Repair | Generated {len(new_sub_queries)} new sub-queries "
                f"| Depth: {state.current_depth + 1}",
            )

        state.reasoning_trace.append(
            f"Step 5b Repair: Generated {len(new_sub_queries)} new sub-queries: "
            f"{new_sub_queries}"
        )

        return new_sub_queries if new_sub_queries else [original_query]

    except Exception as e:
        if print_debug:
            debug_log("WARNING", "⚠️", f"Step 5b Repair failed: {str(e)[:100]}")
        return [original_query]


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================


def self_rag_query(
    query: str,
    vectorstore: FAISS,
    notebook_id: Optional[str] = None,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    llm_chain: Any = None,
    print_debug: bool = False,
) -> tuple[str, List[Dict[str, Any]], bool, List[str], Dict[str, float]]:
    """
    Main Self-RAG orchestrator — chains Steps 0-5b for a complete query pipeline.

    Execution flow:
    0. Intent routing (detect greetings, short-circuit if needed)
    1. Generate search plan (1-3 sub-queries)
    2. Retrieve with surgical retry (hybrid search + cross-encoder re-rank)
    3. Generate candidates (diverse answers at varied temperatures)
    4. Score candidates (ISSUP, ISREL, ISUSE)
    5a. Validate winner (pass all quality thresholds?)
    5b. Repair: if failed and depth < max_depth, diagnose and retry with new strategy.
    If validation passes or max_depth reached, return answer with sources.

    Args:
        query: User input query (raw, before reformulation).
        vectorstore: FAISS vectorstore for retrieval.
        notebook_id: Notebook ID for loading per-notebook settings and thresholds.
        chat_history: Optional chat history for contextual query reformulation.
        llm_chain: Pre-initialized LLM chain (will initialize if None).
        print_debug: Whether to print detailed debug logs.

    Returns:
        tuple:
            - answer (str): Final answer with [STATUS: ...] tag prefix
            - sources (List[Dict]): Top-5 cited sources (document, page, content)
            - found_answer (bool): True if answer is document-grounded (DOC_ANSWER)
            - reasoning_trace (List[str]): Full decision trace for UI transparency
            - confidence_metrics (Dict[str, float]): ISSUP/ISREL/ISUSE/total_score
    """
    # Initialize state (reformulates query using chat history if available)
    state = initialize_self_rag_state(query, notebook_id, chat_history, print_debug)

    # Load settings once before pipeline begins (LLM init + intent routing both need it)
    settings = load_notebook_settings(notebook_id)
    max_depth = settings["self_rag_max_depth"]
    max_candidates = settings["self_rag_candidates"]

    # Initialize LLM chain if not provided
    if llm_chain is None:
        try:
            from langchain_ollama import OllamaLLM

            llm = OllamaLLM(
                model=settings["llm_model_name"],
                base_url=cfg.LLM_BASE_URL,
                temperature=settings["llm_avg_temp"],
            )
            llm_chain = llm
        except Exception as e:
            if print_debug:
                debug_log("ERROR", "🔴", f"Failed to initialize LLM: {str(e)}")
            return (
                "Error: Could not initialize LLM. Please ensure Ollama is running.",
                [],
                False,
                state.reasoning_trace,
                state.confidence_metrics,
            )

    # Step 0: Intent routing (uses contextually reformulated query)
    is_greeting_query, routing_decision = perform_intent_routing(
        state.original_query, notebook_id, print_debug
    )
    state.reasoning_trace.append(routing_decision)

    if is_greeting_query:
        if print_debug:
            debug_log(
                "INFO", "🎯", "Greeting detected → generating dynamic general response"
            )
        greeting_answer = generate_fallback_answer(
            state.original_query, llm_chain, chat_history, print_debug
        )
        return (
            f"[STATUS: GENERAL]\n{greeting_answer}",
            [],
            False,
            state.reasoning_trace,
            state.confidence_metrics,
        )

    # Main loop: Steps 1-5b with potential recursive repair
    max_iterations = max_depth + 1
    winner: Dict[str, Any] = {}
    retrieved_docs: List[Document] = []
    next_sub_queries: Optional[List[str]] = None

    for iteration in range(max_iterations):
        state.current_depth = iteration

        if print_debug:
            debug_log(
                "INFO",
                "🌀",
                f"Self-RAG Iteration {iteration + 1}/{max_iterations} | Depth: {iteration}",
            )

        try:
            # Step 1: Generate (or reuse repair agent's) search plan
            if next_sub_queries:
                sub_queries = next_sub_queries
                next_sub_queries = None
            else:
                sub_queries = generate_search_plan(
                    state.original_query,
                    llm_chain,
                    current_depth=iteration,
                    search_history=state.search_history,
                    print_debug=print_debug,
                )

            # Oscillation guard: prevent identical search plans on consecutive hops
            current_plan_str = " | ".join(
                sorted([sq.lower().strip() for sq in sub_queries])
            )
            if current_plan_str in state.search_history:
                if print_debug:
                    debug_log(
                        "WARNING",
                        "⚠️",
                        "Oscillation Guard: Identical search plan detected twice. "
                        "Terminating loop to prevent infinite recursion.",
                    )
                state.reasoning_trace.append(
                    "Oscillation Guard: Repair agent generated identical search plan twice. Breaking cycle."
                )
                break

            state.search_history.append(current_plan_str)

            # Step 2: Retrieve with surgical retry
            retrieved_docs, state = retrieve_with_surgical_retry(
                sub_queries,
                vectorstore,
                notebook_id,
                state,
                llm_chain,
                print_debug,
            )

            if not retrieved_docs:
                if print_debug:
                    debug_log("WARNING", "⚠️", "No documents retrieved. Using fallback.")
                state.reasoning_trace.append(
                    "Step 2 Retrieval returned no documents - using fallback generation"
                )

            # Step 3: Generate candidates
            candidate_dicts: List[Dict[str, Any]] = generate_candidate_answers(
                state.original_query,
                sub_queries,
                retrieved_docs,
                llm_chain,
                chat_history,
                notebook_id,
                max_candidates,
                print_debug,
            )

            if not candidate_dicts:
                state.reasoning_trace.append("Step 3 Generator produced no candidates")
                candidate_dicts = [
                    {
                        "answer": "Unable to generate response. Please rephrase your question.",
                        "generator_found_answer": False,
                    }
                ]

            # Extract plain text candidates for the scorer
            candidate_texts = [cd["answer"] for cd in candidate_dicts]

            # Step 4: Score candidates
            scored_candidates = score_candidates_with_judges(
                state.original_query,
                candidate_texts,
                retrieved_docs,
                llm_chain,
                print_debug,
                None,  # progress_callback skipped at orchestrator level
                notebook_id=notebook_id,
            )

            # Merge generator_found_answer into scored candidates so pick_winner carries it forward
            for idx, scored in enumerate(scored_candidates):
                if idx < len(candidate_dicts):
                    scored["generator_found_answer"] = candidate_dicts[idx][
                        "generator_found_answer"
                    ]
                else:
                    scored["generator_found_answer"] = False

            # Pick winner
            winner = pick_winner(scored_candidates)

            # Step 5a: Validate winner against quality thresholds
            passed_validation, _ = validate_winner_against_thresholds(
                winner,
                notebook_id,
                state,
                print_debug,
            )

            if passed_validation:
                if print_debug:
                    debug_log("SUCCESS", "🎉", "Winner passed all quality thresholds!")
                break

            elif iteration >= max_depth:
                if print_debug:
                    debug_log(
                        "WARNING",
                        "⏹️",
                        f"Max depth ({max_depth}) reached. Returning best-so-far answer.",
                    )
                state.reasoning_trace.append(
                    f"Reached max depth ({max_depth}). Returning best-so-far with low confidence."
                )
                break

            else:
                # Step 5b: Repair and retry
                if print_debug:
                    debug_log(
                        "INFO",
                        "🔧",
                        f"Winner failed validation. Triggering repair "
                        f"(attempt {iteration + 1}/{max_depth})",
                    )

                next_sub_queries = repair_failed_answer(
                    winner,
                    state,
                    state.original_query,
                    llm_chain,
                    notebook_id,
                    print_debug,
                )

        except Exception as e:
            if print_debug:
                debug_log(
                    "ERROR", "🔴", f"Iteration {iteration + 1} error: {str(e)[:100]}"
                )
            state.reasoning_trace.append(
                f"Iteration {iteration + 1} error: {str(e)[:100]}"
            )
            if iteration >= max_depth:
                break

    # Handle empty winner (all iterations failed)
    if not winner:
        winner = {
            "answer": "Unable to generate a response. Please try again.",
            "issup": 0.0,
        }

    # Handle empty retrieval pool across all iterations
    if not state.retrieval_pool:
        if print_debug:
            debug_log(
                "WARNING",
                "⚠️",
                "Empty retrieval pool — generating LLM-based general knowledge response",
            )
        state.reasoning_trace.append(
            "No relevant documents found — generating LLM-based general knowledge response"
        )
        winner["answer"] = generate_fallback_answer(
            state.original_query, llm_chain, chat_history, print_debug
        )

    final_answer = winner.get("answer", "No answer generated.")

    # Determine if answer is document-grounded using the generator LLM's own assessment.
    # The generator included [FOUND: YES/NO] in its output; this was captured and stored
    # as generator_found_answer in each candidate dict, then merged into scored candidates.
    # Fall back to False if the key is missing (e.g. empty winner fallback).
    found_answer = bool(winner.get("generator_found_answer", False))

    # Format top-5 sources for citation display
    sources = [
        {
            "document": str(
                cast(Dict[str, Any], getattr(doc, "metadata")).get(
                    "document", "Unknown"
                )
            ),
            "page": str(
                cast(Dict[str, Any], getattr(doc, "metadata")).get("page", "N/A")
            ),
            "content": doc.page_content[:200] + "...",
        }
        for doc in retrieved_docs[:5]
    ]

    # Append final entry BEFORE printing summary so it appears in the trace output
    state.reasoning_trace.append(
        f"Self-RAG complete after {state.current_depth + 1} iteration(s). Found: {found_answer}"
    )

    if print_debug:
        debug_log(
            "SUCCESS",
            "✅",
            f"Self-RAG Complete | Depth: {state.current_depth} | "
            f"Found Answer: {found_answer} | Sources: {len(sources)}",
        )
        debug_log("INFO", "═" * 70)
        debug_log("INFO", "SELF-RAG EXECUTION SUMMARY")
        debug_log("INFO", "─" * 70)
        debug_log("INFO", f"Total Iterations: {state.current_depth + 1}")
        debug_log(
            "INFO",
            f"Answer Status: {'DOC_ANSWER' if found_answer else 'DOC_MISSING/GENERAL'}",
        )
        debug_log("INFO", f"Winner Confidence (ISSUP): {winner.get('issup', 0.0):.2f}")
        debug_log("INFO", f"Total Unique Docs Retrieved: {len(retrieved_docs)}")
        debug_log("INFO", f"Source Citations: {len(sources)}")
        debug_log("INFO", "─" * 70)
        debug_log("INFO", "Execution Trace (last 5 steps):")
        for idx, trace_entry in enumerate(state.reasoning_trace[-5:], 1):
            debug_log("INFO", f"  {idx}. {trace_entry[:80]}")
        debug_log("INFO", "═" * 70)

    status_tag = "[STATUS: DOC_ANSWER]" if found_answer else "[STATUS: DOC_MISSING]"

    return (
        f"{status_tag}\n{final_answer}",
        sources,
        found_answer,
        state.reasoning_trace,
        state.confidence_metrics,
    )


# ============================================================================
# RAG CHAIN FACTORY
# ============================================================================


def create_history_aware_rag_chain(
    vectorstore: FAISS,
    print_debug: bool = False,
    notebook_id: Optional[str] = None,
) -> Any:
    """
    Build a history-aware RAG chain backed by the Self-RAG orchestrator.

    Wraps self_rag_query() in a LangChain-compatible RunnableLambda so that callers
    only need to pass `{"input": question, "chat_history": [...]}` and receive the
    final answer string. Self-RAG metadata (sources, reasoning_trace, confidence_metrics,
    found_answer) is stored in st.session_state["self_rag_metadata"] for optional
    UI transparency display.

    Self-RAG Flow (Steps 0-5b):
    0. Intent Routing — detect greetings, short-circuit for general queries
    1. Search Planning — decompose query into 1-3 sub-queries
    2. Retrieval with Surgical Retry — hybrid search + cross-encoder re-rank
    3. Candidate Generation — 3 diverse drafts
    4. Quality Scoring — ISSUP / ISREL / ISUSE judging
    5a. Threshold Validation — accept winner if all gates pass
    5b. Recursive Repair — re-plan and retry up to max_depth times

    Args:
        vectorstore: FAISS vectorstore containing indexed document embeddings.
        print_debug: Enable detailed debug logging throughout chain execution.
        notebook_id: Notebook UUID for loading per-notebook RAG settings.

    Returns:
        RunnableLambda: Chain accepting {"input": str, "chat_history": list}
                        and returning the final answer string.
    """
    from langchain_core.runnables import RunnableLambda

    if print_debug:
        print("\n")
        debug_log("INFO", "🛠️", "Building Self-RAG Enhanced RAG Chain...")

    def self_rag_wrapper(x: Dict[str, Any]) -> str:
        """
        Adapt the LangChain input dict to the self_rag_query() call signature.

        Args:
            x: Dict with keys 'input' or 'question' (query) and optional 'chat_history'.

        Returns:
            str: Final answer from the Self-RAG orchestrator.
        """
        question = str(x.get("input", x.get("question", "")))
        chat_history: List[Dict[str, Any]] = x.get("chat_history", [])

        if print_debug:
            debug_log("INFO", "📝", f"Input Question: {question[:100]}...")

        answer, sources, found_answer, reasoning_trace, confidence_metrics = (
            self_rag_query(
                query=question,
                vectorstore=vectorstore,
                notebook_id=notebook_id,
                chat_history=chat_history,
                llm_chain=None,  # Let self_rag_query initialize its own LLM chain
                print_debug=print_debug,
            )
        )

        # Store metadata in session state for optional UI transparency display
        if "self_rag_metadata" not in st.session_state:
            st.session_state.self_rag_metadata = {}

        # Detect whether this was a greeting/general-knowledge short-circuit so
        # run_dual_rag() can skip Co-RAG (no document retrieval needed).
        is_greeting_query: bool = bool(
            re.search(r"\[STATUS:\s*GENERAL\]", answer, re.IGNORECASE)
        )

        st.session_state.self_rag_metadata = {
            "reasoning_trace": reasoning_trace,
            "sources": sources,
            "found_answer": found_answer,
            "last_query": question,
            "confidence_metrics": confidence_metrics,
            # True when Self-RAG short-circuited for a greeting/general query
            # (for UI display only — Co-RAG runs its own independent intent routing).
            "is_greeting": is_greeting_query,
        }

        if print_debug:
            debug_log(
                "INFO",
                "✅",
                f"Self-RAG Complete: {len(sources)} sources, found_answer={found_answer}",
            )
            # Reasoning trace is captured & logged downstream in process_user_query
            # to avoid printing it twice in the terminal.

        return answer

    rag_chain: Any = RunnableLambda(self_rag_wrapper)

    if print_debug:
        debug_log(
            "INFO",
            "🚀",
            "Self-RAG Chain created: Question + History → Intent Routing → "
            "Search Planning → Multi-Hop Retrieval → Quality Scoring → Answer",
        )

    return rag_chain
