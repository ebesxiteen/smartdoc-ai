"""
Co-RAG (Collaborative Retrieval-Augmented Generation) pipeline for SmartDoc AI.

This module implements the Co-RAG orchestration pipeline as a fully independent
engine from A to Z:
  Step 0a — Query Reformulation (using Co-RAG-isolated chat history)
  Step 0b — 2-Layer Intent Routing (independent greeting detection)
             Short-circuits with own greeting response if greeting detected
  Step 1  — Holistic single-shot retrieval (full context in one pass)
  Step 2  — Initial answer generation (Generator Mode A)
  Step 3  — Iterative Generator ↔ Reviewer collaboration loop
             - Reviewer diagnoses gaps, hallucinations, contradictions
             - Generator applies targeted redlines (Mode B)
             - Loop exits on [STATUS: VERIFIED] or exhausted retries
  Step 4  — Return final draft with critique history as reasoning trace

Public API consumed by core/rag.py:
  - co_rag_query()   — main orchestrator
  - CoRAGState       — per-query state dataclass

Design Principles:
  - Full pipeline independence: Co-RAG receives only the raw user query +
    its own co_rag_chat_history. It never reads Self-RAG’s session state,
    reformulated query, or greeting detection result.
  - Holistic retrieval: one broad pass instead of iterative sub-queries
  - Chat history isolation: Co-RAG builds its own context from co_rag_content
    (not self_rag_content) so each pipeline stays independent
  - Shared infrastructure only: FAISS/BM25 search, SHARED_RAG_STYLE_RULES,
    personal_ctx, and RAG configuration constants are shared.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Callable, cast, Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import core.configs as cfg
from core.utils import (
    debug_log,
    format_context_with_sources,
    generate_fallback_answer,
    load_notebook_settings,
    rerank_with_cross_encoder,
    retrieve_quality_chunks,
)


# ============================================================================
# CO-RAG STATE
# ============================================================================


def _empty_str_list() -> List[str]:
    return []


def _empty_dict_list() -> List[Dict[str, str]]:
    return []


def _empty_doc_list() -> List[Document]:
    return []


@dataclass
class CoRAGState:
    """
    Mutable state object tracking Co-RAG query execution across Generator↔Reviewer turns.

    Attributes:
        current_turn (int): Current collaboration turn (0 = after initial generation).
        critique_history (List[str]): All reviewer critiques so far (for context continuity).
        horizontal_trace (List[Dict[str, str]]): Verbose log of each generation/review step.
        original_query (str): Co-RAG’s own contextually reformulated standalone query
            (derived from the raw user query + co_rag_chat_history, independent of
            Self-RAG’s reformulated query).
        holistic_context_docs (List[Document]): Retrieved docs from single-shot retrieval.
        current_draft (str): The most recent answer draft from the Generator.
        co_rag_found_answer (bool): Whether relevant context was retrieved.
    """

    current_turn: int = 0
    critique_history: List[str] = field(default_factory=_empty_str_list)
    horizontal_trace: List[Dict[str, str]] = field(default_factory=_empty_dict_list)
    original_query: str = ""
    holistic_context_docs: List[Document] = field(default_factory=_empty_doc_list)
    context_str: str = (
        ""  # pre-computed once after retrieval; reused by generate/review
    )
    current_draft: str = ""
    co_rag_found_answer: bool = False


# ============================================================================
# STEP 1 — HOLISTIC RETRIEVAL
# ============================================================================


def _co_rag_retrieve(
    state: CoRAGState,
    vectorstore: FAISS,
    settings: Dict[str, Any],
    print_debug: bool = False,
) -> CoRAGState:
    """
    Perform a single broad holistic retrieval pass for Co-RAG with cross-encoder reranking.

    Unlike Self-RAG's multi-hop sub-query approach, Co-RAG retrieves the full
    relevant context in one pass using the reformulated query. A broad initial
    fetch (rag_rerank_top_n) is narrowed down via cross-encoder reranking to
    rag_final_context_k, matching Self-RAG's two-stage retrieval quality.

    Args:
        state: Current CoRAGState.
        vectorstore: FAISS vectorstore to search.
        settings: Pre-loaded notebook settings (avoids redundant DB round-trip).
        print_debug: Enable debug logging.

    Returns:
        Updated CoRAGState with holistic_context_docs and co_rag_found_answer set.
    """
    final_k = int(settings.get("rag_final_context_k", cfg.RAG_FINAL_CONTEXT_K))
    rerank_top_n = int(settings.get("rag_rerank_top_n", cfg.RAG_RERANK_TOP_N))
    min_results = int(
        settings.get("rag_retrieval_min_results", cfg.RAG_RETRIEVAL_MIN_RESULTS)
    )
    score_threshold = float(
        settings.get("rag_retrieval_score_threshold", cfg.RAG_RETRIEVAL_SCORE_THRESHOLD)
    )

    if print_debug:
        debug_log(
            "INFO",
            "🔍",
            f"Holistic Retrieval | Query: {state.original_query[:60]}...",
        )

    # Stage 1: broad initial fetch
    docs = retrieve_quality_chunks(
        vectorstore,
        state.original_query,
        k=rerank_top_n,
        min_results=min_results,
        score_threshold=score_threshold,
        print_debug=print_debug,
        weight_semantic=float(settings.get("weight_semantic", cfg.WEIGHT_SEMANTIC)),
        weight_bm25=float(settings.get("weight_bm25", cfg.WEIGHT_BM25)),
    )

    # Stage 2: cross-encoder reranking to final_k
    if len(docs) > final_k:
        docs = rerank_with_cross_encoder(
            state.original_query, docs, top_k=final_k, print_debug=print_debug
        )

    state.holistic_context_docs = docs
    state.co_rag_found_answer = len(docs) > 0

    action_msg = f"Holistic Retrieval: {len(docs)} chunk(s) retrieved"
    state.horizontal_trace.append({"step": "Retrieval", "action": action_msg})

    if print_debug:
        debug_log("INFO", "📦", action_msg)

    return state


# ============================================================================
# STEP 2 / 3A — GENERATOR (MODE A: INITIAL, MODE B: REFINE)
# ============================================================================


def _co_rag_generate(
    state: CoRAGState,
    llm_chain: Any,
    mode: str = "initial",
    last_critique: str = "",
    personal_ctx: str = "",
    chat_history_str: str = "",
    context_str: str = "",
    print_debug: bool = False,
) -> CoRAGState:
    """
    Generator step for Co-RAG — Mode A (initial) or Mode B (refine).

    Mode A produces the first draft grounded in the holistic context.
    Mode B applies the Reviewer's targeted critique to revise the previous draft.

    Args:
        state: Current CoRAGState.
        llm_chain: Initialized Ollama LLM instance.
        mode: "initial" for Mode A, "refine" for Mode B.
        last_critique: The Reviewer's critique from the previous turn (Mode B only).
        personal_ctx: Pre-formatted personal context block (empty string if none).
        chat_history_str: Pre-formatted Co-RAG-isolated chat history block (empty string if none).
        context_str: Pre-computed formatted context string (avoids redundant formatting).
        print_debug: Enable debug logging.

    Returns:
        Updated CoRAGState with current_draft populated.
    """
    from langchain_core.prompts import ChatPromptTemplate

    ctx = context_str or format_context_with_sources(state.holistic_context_docs)

    if mode == "initial":
        prompt_template = cfg.CO_RAG_GENERATOR_INITIAL_PROMPT
        input_vars: Dict[str, str] = {
            "context": ctx,
            "query": state.original_query,
            "personal_ctx_section": personal_ctx,
            "chat_history_section": chat_history_str,
        }
        step_label = "Draft (Initial)"
        trace_label = "Generator Mode A (Initial)"
    else:
        prompt_template = cfg.CO_RAG_GENERATOR_REFINE_PROMPT
        input_vars = {
            "context": ctx,
            "query": state.original_query,
            "draft": state.current_draft,
            "critique": last_critique,
            "personal_ctx_section": personal_ctx,
            "chat_history_section": chat_history_str,
        }
        step_label = f"Draft (Refine, Turn {state.current_turn})"
        trace_label = f"Generator Mode B (Refine, Turn {state.current_turn})"

    if print_debug:
        debug_log("INFO", "✍️", f"{trace_label}")

    try:
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain: Any = prompt | llm_chain
        raw_response = chain.invoke(input_vars)
        draft = str(raw_response).strip()

        if not draft:
            draft = cfg.NOT_FOUND_ANSWER_FALL_BACK

    except Exception as e:
        if print_debug:
            debug_log("WARNING", "⚠️", f"Generator error: {str(e)[:100]}")
        draft = state.current_draft or cfg.NOT_FOUND_ANSWER_FALL_BACK

    state.current_draft = draft
    action_msg = f"{trace_label}: draft generated ({len(draft)} chars)"
    state.horizontal_trace.append({"step": step_label, "action": action_msg})

    if print_debug:
        debug_log("INFO", "💬", action_msg)

    return state


# ============================================================================
# STEP 3B — REVIEWER
# ============================================================================

_REVIEWER_STATUS_RE = re.compile(
    r"\[STATUS:\s*(VERIFIED|PARTIAL_VERIFIED|CRITICAL_ERROR)\]",
    re.IGNORECASE,
)


def _co_rag_review(
    state: CoRAGState,
    llm_chain: Any,
    context_str: str = "",
    print_debug: bool = False,
) -> tuple[str, str]:
    """
    Reviewer step for Co-RAG — critiques the current draft and returns a status.

    The Reviewer has awareness of the full critique history so it can check
    whether previous issues were addressed by the Generator.

    Args:
        state: Current CoRAGState (reads current_draft, critique_history, original_query).
        llm_chain: Initialized Ollama LLM instance.
        context_str: Pre-computed formatted context string (avoids redundant formatting).
        print_debug: Enable debug logging.

    Returns:
        Tuple[str, str]:
            - status: "VERIFIED", "PARTIAL_VERIFIED", or "CRITICAL_ERROR"
            - critique_text: Full reviewer response including the status tag
    """
    from langchain_core.prompts import ChatPromptTemplate

    ctx = context_str or format_context_with_sources(state.holistic_context_docs)
    critique_history_str = (
        "\n\n---\n\n".join(
            [
                f"Turn {i + 1} Critique:\n{c}"
                for i, c in enumerate(state.critique_history)
            ]
        )
        if state.critique_history
        else "No prior critiques."
    )

    if print_debug:
        debug_log(
            "INFO",
            "🔬",
            f"Reviewer (Turn {state.current_turn}) — evaluating draft...",
        )

    try:
        prompt = ChatPromptTemplate.from_template(cfg.CO_RAG_REVIEWER_PROMPT)
        chain: Any = prompt | llm_chain
        raw_response = chain.invoke(
            {
                "context": ctx,
                "query": state.original_query,
                "draft": state.current_draft,
                "critique_history": critique_history_str,
            }
        )
        critique_text = str(raw_response).strip()
    except Exception as e:
        if print_debug:
            debug_log("WARNING", "⚠️", f"Reviewer error: {str(e)[:100]}")
        critique_text = (
            f"[STATUS: VERIFIED]\n(Reviewer failed, accepting draft: {str(e)[:80]})"
        )

    # Parse status from response
    match = _REVIEWER_STATUS_RE.search(critique_text)
    status = match.group(1).upper() if match else "PARTIAL_VERIFIED"

    # Append critique to history and trace
    state.critique_history.append(critique_text)
    action_msg = (
        f"Reviewer Turn {state.current_turn}: "
        f"STATUS={status} | critique ({len(critique_text)} chars)"
    )
    state.horizontal_trace.append(
        {"step": f"Review (Turn {state.current_turn})", "action": action_msg}
    )

    if print_debug:
        debug_log("INFO", "📋", action_msg)

    return status, critique_text


# ============================================================================
# MAIN CO-RAG ORCHESTRATOR
# ============================================================================


# Regex for parsing [FOUND: YES/NO] tag from Co-RAG generator output
_CO_RAG_FOUND_TAG_RE = re.compile(r"\[FOUND:\s*(YES|NO)\]", re.IGNORECASE)


def co_rag_query(
    query: str,
    vectorstore: FAISS,
    notebook_id: Optional[str] = None,
    llm_chain: Optional[Any] = None,
    co_rag_chat_history: Optional[List[Dict[str, Any]]] = None,
    print_debug: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> tuple[str, List[Dict[str, Any]], bool, List[Dict[str, str]]]:
    """
    Main Co-RAG orchestrator — fully independent pipeline from A to Z.

    Execution flow:
    0a. Query Reformulation: resolve follow-up references using co_rag_chat_history
    0b. 2-Layer Intent Routing: detect greetings (regex → LLM fallback)
        If greeting → generate own greeting response and return immediately
    1.  Holistic retrieval (single broad pass)
    2.  Generator Mode A (initial draft)
    3.  Loop up to co_rag_max_retries times:
        a. Reviewer critiques the draft → parses [STATUS: ...] tag
        b. If VERIFIED → exit loop
        c. Else → Generator Mode B (targeted refinement)
    4.  Parse [FOUND: YES/NO] from final draft → set co_rag_found_answer
    5.  Return final draft, sources, found_answer, horizontal_trace

    Args:
        query: Raw user query (not pre-reformulated). Co-RAG performs its own
            contextual reformulation using co_rag_chat_history independently.
        vectorstore: FAISS vectorstore for retrieval.
        notebook_id: Notebook ID for per-notebook settings.
        llm_chain: Pre-initialized OllamaLLM instance (initialized here if None).
        co_rag_chat_history: Isolated Co-RAG chat history (assistant turns use
            co_rag_content). Used for independent query reformulation, greeting
            generation, and context injection. Strictly limited to
            max_msg_history entries to prevent context overflow.
        print_debug: Enable detailed debug logging.
        progress_callback: Optional callable for UI progress updates.

    Returns:
        tuple:
            - content (str): Final answer from the Co-RAG pipeline (tags stripped)
            - sources (List[Dict]): Top-5 cited sources (document, page, content)
            - co_rag_found_answer (bool): True if LLM tagged [FOUND: YES] (or retrieval found docs)
            - horizontal_trace (List[Dict[str, str]]): Structured execution trace
              Format: [{"step": "...", "action": "..."}]
    """
    settings = load_notebook_settings(notebook_id)
    max_retries = int(settings.get("co_rag_max_retries", cfg.CO_RAG_MAX_RETRIES))
    max_msg_history = int(settings.get("max_msg_history", cfg.MAX_MSG_HISTORY))
    personal_ctx_raw: str = settings.get("personal_ctx") or ""

    if print_debug:
        debug_log("INFO", "═" * 70)
        debug_log("INFO", "⭐", "Co-RAG Pipeline START (independent)")
        debug_log(
            "INFO",
            "📝",
            f"Raw Query: {query[:80]}...",
        )
        debug_log("INFO", "⚙️", f"Max Retries: {max_retries}")

    # --- Build personal context section ---
    personal_ctx_section: str = (
        f"YOUR CONTEXT:\n{personal_ctx_raw}\n\n" if personal_ctx_raw else ""
    )

    # --- Build Co-RAG-isolated chat history section ---
    # Respects max_msg_history to prevent context overflow.
    # Note: Since a single turn now holds TWO LLM responses, the effective token
    # budget fills twice as fast — use at most half of max_msg_history turns.
    chat_history_section: str = ""
    if co_rag_chat_history:
        history_budget = max(1, max_msg_history // 2)
        trimmed = co_rag_chat_history[-history_budget:]
        lines: List[str] = []
        for msg in trimmed:
            role_label = (
                "User" if msg.get("role") == cfg.USER_ROLE_NAME else "Assistant"
            )
            lines.append(f"{role_label}: {msg.get('content', '').strip()}")
        if lines:
            chat_history_section = "RECENT CONVERSATION:\n" + "\n".join(lines) + "\n\n"
        if print_debug:
            debug_log(
                "INFO",
                "📚",
                f"Co-RAG history: {len(lines)} messages (budget: {history_budget})",
            )

    # Initialize LLM if not provided
    if llm_chain is None:
        try:
            from langchain_ollama import OllamaLLM

            llm_chain = OllamaLLM(
                model=settings.get("llm_model_name", cfg.LLM_MODEL_NAME),
                base_url=cfg.LLM_BASE_URL,
                temperature=settings.get("llm_avg_temp", cfg.LLM_AVG_TEMP),
            )
        except Exception as e:
            if print_debug:
                debug_log("ERROR", "🔴", f"LLM init failed: {str(e)[:100]}")
            fallback_trace: List[Dict[str, str]] = [
                {
                    "step": "Error",
                    "action": f"LLM initialization failed: {str(e)[:100]}",
                }
            ]
            return cfg.NOT_FOUND_ANSWER_FALL_BACK, [], False, fallback_trace

    # ── Step 0a: Co-RAG-independent query reformulation ───────────────────
    # Resolve follow-up references using co_rag_chat_history (NOT Self-RAG history).
    # This may produce a different reformulated query than Self-RAG if the two
    # pipelines have diverging conversation histories.
    if progress_callback:
        progress_callback("⭐ Co-RAG: Reformulating query...")

    from core.self_rag import reformulate_query_with_history  # lazy — no circular risk

    reformulated_query: str = reformulate_query_with_history(
        query,
        co_rag_chat_history or [],
        notebook_id,
        print_debug,
    )

    if print_debug:
        debug_log(
            "INFO",
            "🔄",
            f"Co-RAG reformulated query: {reformulated_query[:80]}...",
        )

    # ── Step 0b: Co-RAG-independent intent routing ────────────────────────
    # Run the same 2-layer greeting detection but on Co-RAG's own reformulated
    # query and using Co-RAG's chat history for greeting response generation.
    if progress_callback:
        progress_callback("⭐ Co-RAG: Intent routing...")

    from core.self_rag import perform_intent_routing  # lazy — no circular risk

    is_co_rag_greeting, routing_decision = perform_intent_routing(
        reformulated_query, notebook_id, print_debug
    )

    if is_co_rag_greeting:
        if print_debug:
            debug_log(
                "INFO",
                "🎯",
                "Co-RAG greeting detected → generating independent greeting response",
            )
        greeting_answer = generate_fallback_answer(
            reformulated_query, llm_chain, co_rag_chat_history, print_debug
        )
        greeting_trace: List[Dict[str, str]] = [
            {"step": "Intent Routing", "action": f"{routing_decision}"},
            {
                "step": "Greeting Bypass",
                "action": "Co-RAG generated independent greeting response. Holistic retrieval skipped.",
            },
        ]
        return greeting_answer, [], False, greeting_trace

    # Log routing decision for factual queries
    if print_debug:
        debug_log(
            "INFO",
            "🎯",
            f"Co-RAG intent routing: {routing_decision}",
        )

    # Initialize state with Co-RAG's own reformulated query
    state = CoRAGState(original_query=reformulated_query)
    state.horizontal_trace.append(
        {"step": "Intent Routing", "action": f"{routing_decision}"}
    )

    # --- Step 1: Holistic Retrieval ---
    if progress_callback:
        progress_callback("⭐ Co-RAG: Holistic retrieval...")

    state = _co_rag_retrieve(state, vectorstore, settings, print_debug)

    # Pre-compute context string once — reused by all Generator and Reviewer calls
    state.context_str = format_context_with_sources(state.holistic_context_docs)

    # Early exit if no context was retrieved
    if not state.co_rag_found_answer and not state.holistic_context_docs:
        fallback_answer = generate_fallback_answer(
            reformulated_query, llm_chain, co_rag_chat_history, print_debug
        )
        empty_trace: List[Dict[str, str]] = state.horizontal_trace + [
            {
                "step": "Empty Retrieval",
                "action": "No context retrieved. Returning fallback answer.",
            }
        ]
        return fallback_answer, [], False, empty_trace

    # --- Step 2: Initial Generation (Mode A) ---
    if progress_callback:
        progress_callback("⭐ Co-RAG: Generating initial draft...")

    state = _co_rag_generate(
        state,
        llm_chain,
        mode="initial",
        personal_ctx=personal_ctx_section,
        chat_history_str=chat_history_section,
        context_str=state.context_str,
        print_debug=print_debug,
    )

    # --- Step 3: Generator↔Reviewer Loop ---
    if max_retries == 0:
        # Skip review entirely — single-pass mode
        state.horizontal_trace.append(
            {
                "step": "Review",
                "action": "Review skipped (co_rag_max_retries=0). Returning initial draft.",
            }
        )
        if print_debug:
            debug_log("INFO", "⏭️", "Review loop skipped (max_retries=0)")
    else:
        for turn in range(max_retries):
            state.current_turn = turn + 1

            if progress_callback:
                progress_callback(
                    f"⭐ Co-RAG: Reviewing draft (Turn {state.current_turn}/{max_retries})..."
                )

            # Step 3a: Review
            status, critique_text = _co_rag_review(
                state, llm_chain, context_str=state.context_str, print_debug=print_debug
            )

            if status == "VERIFIED":
                state.horizontal_trace.append(
                    {
                        "step": "Verified",
                        "action": f"Turn {state.current_turn}: VERIFIED — answer accepted.",
                    }
                )
                if print_debug:
                    debug_log(
                        "SUCCESS",
                        "🎉",
                        f"VERIFIED at Turn {state.current_turn}",
                    )
                break

            # Step 3b: Refine if not last turn
            if turn < max_retries - 1:
                if progress_callback:
                    progress_callback(
                        f"⭐ Co-RAG: Refining draft (Turn {state.current_turn}/{max_retries})..."
                    )
                state = _co_rag_generate(
                    state,
                    llm_chain,
                    mode="refine",
                    last_critique=critique_text,
                    personal_ctx=personal_ctx_section,
                    chat_history_str=chat_history_section,
                    context_str=state.context_str,
                    print_debug=print_debug,
                )
            else:
                # Max retries exhausted without VERIFIED — circuit breaker
                fate_msg = (
                    f"[STATUS: FATE_ACCEPTED] Max retries ({max_retries}) reached without VERIFIED. "
                    f"Last status: {status}. Accepting current draft."
                )
                state.horizontal_trace.append(
                    {"step": "Circuit Breaker", "action": fate_msg}
                )
                if print_debug:
                    debug_log(
                        "WARNING",
                        "⏹️",
                        f"Max retries reached. Last status: {status}",
                    )

    # --- Step 4: Parse [FOUND: YES/NO] from final draft and strip tag ---
    raw_draft = state.current_draft or cfg.NOT_FOUND_ANSWER_FALL_BACK
    found_match = _CO_RAG_FOUND_TAG_RE.search(raw_draft)
    if found_match:
        co_rag_found_answer: bool = found_match.group(1).upper() == "YES"
        final_content = _CO_RAG_FOUND_TAG_RE.sub("", raw_draft).strip()
    else:
        co_rag_found_answer = state.co_rag_found_answer  # fallback to retrieval-based
        final_content = raw_draft

    if not final_content:
        final_content = cfg.NOT_FOUND_ANSWER_FALL_BACK

    # --- Step 5: Format sources ---
    sources: List[Dict[str, Any]] = [
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
        for doc in state.holistic_context_docs[:5]
    ]

    if print_debug:
        debug_log(
            "SUCCESS",
            "⭐",
            f"Co-RAG Complete | Found: {co_rag_found_answer} | "
            f"Turns: {state.current_turn} | Sources: {len(sources)} | "
            f"Answer: {len(final_content)} chars",
        )
        debug_log("INFO", "═" * 70)

    return final_content, sources, co_rag_found_answer, state.horizontal_trace
