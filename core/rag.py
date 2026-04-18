"""
Unified dual-pipeline RAG orchestrator for SmartDoc AI.

This module provides `run_dual_rag()`, the single entry-point that runs both
the Self-RAG and Co-RAG pipelines for every user query and returns a combined
result dictionary that app.py can render in a two-tab UI.

Pipeline execution order:
  1. Self-RAG (via `process_user_query()`)  — receives raw query + self_rag_history.
     Performs its own query reformulation, intent routing, multi-hop retrieval,
     and answer generation completely independently.
  2. Co-RAG  (via `co_rag_query()`)         — receives the same raw query +
     co_rag_history. Performs its own query reformulation, intent routing,
     holistic retrieval, and Generator↔Reviewer loop completely independently.

Neither engine reads the other's reformulated query, greeting detection result,
or session state. The only shared resources are the FAISS/BM25 search
infrastructure, SHARED_RAG_STYLE_RULES, personal_ctx, and RAG configuration
constants from core/configs.py.

Return dictionary (9 keys):
  self_rag_content          str
  self_rag_sources          List[Dict[str, Any]]
  self_rag_found_answer     bool
  self_rag_reasoning_trace  List[str]
  self_rag_confidence_score Optional[float]
  co_rag_content            str
  co_rag_sources            List[Dict[str, Any]]
  co_rag_found_answer       bool
  co_rag_reasoning_trace    List[Dict[str, str]]
"""

from typing import Any, Callable, Dict, List, Optional

from langchain_community.vectorstores import FAISS

import core.configs as cfg
from core.co_rag import co_rag_query
from core.utils import debug_log, process_user_query


def run_dual_rag(
    query: str,
    rag_chain: Any,
    vectorstore: FAISS,
    self_rag_chat_history: Optional[List[Dict[str, Any]]] = None,
    print_debug: bool = False,
    notebook_id: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    co_rag_chat_history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Run both Self-RAG and Co-RAG pipelines fully independently and return combined results.

    Each engine receives only the raw query and its own isolated chat history.
    Each performs its own query reformulation, intent routing, retrieval, and generation.
    No shared reformulated query, no shared session state, no cross-pipeline short-circuits.

    Args:
        query: Raw user input.
        rag_chain: Instantiated Self-RAG chain from create_history_aware_rag_chain().
        vectorstore: FAISS vectorstore for document retrieval.
        self_rag_chat_history: Chat history for Self-RAG context window
            (assistant turns use `self_rag_content`).
        print_debug: Enable verbose debug logging.
        notebook_id: Notebook ID for loading per-notebook settings.
        progress_callback: Optional callable for UI progress updates.
        co_rag_chat_history: Isolated Co-RAG chat history (assistant turns use
            `co_rag_content` to keep pipeline memories independent).

    Returns:
        Dict[str, Any] with 9 keys:
            self_rag_content, self_rag_sources, self_rag_found_answer,
            self_rag_reasoning_trace, self_rag_confidence_score,
            co_rag_content, co_rag_sources, co_rag_found_answer,
            co_rag_reasoning_trace (List[Dict[str, str]])
    """
    if print_debug:
        debug_log(
            "INFO",
            "🔀",
            "[rag.py] Starting Dual-RAG (Self-RAG + Co-RAG) — fully independent pipelines",
        )

    # ── Step A: Self-RAG ──────────────────────────────────────────────────
    # Runs independently: own reformulation, own intent routing, own retrieval.
    if progress_callback:
        progress_callback("⚡ Self-RAG: Processing query...")

    self_rag_content: str
    self_rag_sources: List[Dict[str, Any]]
    self_rag_found_answer: bool
    self_rag_reasoning_trace: List[str]
    self_rag_confidence_score: Optional[float]

    (
        self_rag_content,
        self_rag_sources,
        self_rag_found_answer,
        self_rag_reasoning_trace,
        self_rag_confidence_score,
    ) = process_user_query(
        query=query,
        rag_chain=rag_chain,
        vectorstore=vectorstore,
        chat_history=self_rag_chat_history,
        print_debug=print_debug,
        notebook_id=notebook_id,
    )

    if print_debug:
        debug_log(
            "INFO",
            "🔵",
            f"[rag.py] Self-RAG done | found={self_rag_found_answer} | "
            f"{len(self_rag_content)} chars",
        )

    # ── Step B: Co-RAG ────────────────────────────────────────────────────
    # Runs independently: own reformulation (from co_rag_chat_history),
    # own 2-layer intent routing, own holistic retrieval + Generator↔Reviewer.
    # Receives the raw query — no shared reformulated query from Self-RAG.
    if progress_callback:
        progress_callback("⭐ Co-RAG: Starting independent pipeline...")

    co_rag_content: str
    co_rag_sources: List[Dict[str, Any]]
    co_rag_found_answer: bool
    co_rag_reasoning_trace: List[Dict[str, str]]

    try:
        from langchain_ollama import OllamaLLM
        from core.utils import load_notebook_settings

        settings = load_notebook_settings(notebook_id)
        llm = OllamaLLM(
            model=settings.get("llm_model_name", cfg.LLM_MODEL_NAME),
            base_url=cfg.LLM_BASE_URL,
            temperature=settings.get("llm_avg_temp", cfg.LLM_AVG_TEMP),
            num_ctx=int(settings.get("llm_num_ctx", cfg.LLM_NUM_CTX)),
        )

        (
            co_rag_content,
            co_rag_sources,
            co_rag_found_answer,
            co_rag_reasoning_trace,
        ) = co_rag_query(
            query=query,
            vectorstore=vectorstore,
            notebook_id=notebook_id,
            llm_chain=llm,
            co_rag_chat_history=co_rag_chat_history,
            print_debug=print_debug,
            progress_callback=progress_callback,
        )

    except Exception as e:
        if print_debug:
            debug_log("ERROR", "🔴", f"[rag.py] Co-RAG pipeline error: {str(e)[:150]}")
        co_rag_content = cfg.NOT_FOUND_ANSWER_FALL_BACK
        co_rag_sources = []
        co_rag_found_answer = False
        co_rag_reasoning_trace = [
            {"step": "Error", "action": f"[rag.py] Co-RAG error: {str(e)[:150]}"}
        ]

    if print_debug:
        debug_log(
            "SUCCESS",
            "🟢",
            f"[rag.py] Co-RAG done | found={co_rag_found_answer} | "
            f"{len(co_rag_content)} chars",
        )
        debug_log("INFO", "🔀", "[rag.py] Dual-RAG complete")

    return {
        "self_rag_content": self_rag_content,
        "self_rag_sources": self_rag_sources,
        "self_rag_found_answer": self_rag_found_answer,
        "self_rag_reasoning_trace": self_rag_reasoning_trace,
        "self_rag_confidence_score": self_rag_confidence_score,
        "co_rag_content": co_rag_content,
        "co_rag_sources": co_rag_sources,
        "co_rag_found_answer": co_rag_found_answer,
        "co_rag_reasoning_trace": co_rag_reasoning_trace,
    }
