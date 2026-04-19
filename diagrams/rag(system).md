# Dual-Pipeline RAG System Diagram 🔀

```mermaid
flowchart TD
    In(["User Input: Raw Query"]) --> Entry["core/rag.py :: run_dual_rag()\nSingle Entry Point for Both Pipelines"]

    Entry --> SelfRAGIn
    Entry --> CoRAGIn

    subgraph SelfRAG ["⚙️  Self-RAG Pipeline  (core/self_rag.py)  — runs first"]
        SelfRAGIn(["Query + self_rag_content\nchat history"])
        SelfRAGIn --> SR_Reform["Contextual Reformulation\n(self_rag history)"]
        SR_Reform --> SR_Greet{"2-Layer Greeting\nDetection"}
        SR_Greet -- "Greeting" --> SR_GreetResp["LLM: Greeting Response"]
        SR_Greet -- "Factual" --> SR_Retrieve["Step 2: Multi-Hop Retrieval\n(Sub-queries + Surgical Retry)\nDedup + Cross-Encoder Reranking"]
        SR_Retrieve --> SR_Gen["Step 3: Candidate Generation\n(self_rag_candidates answers)"]
        SR_Gen --> SR_Score["Step 4: Score + Pick Winner\nISSUP + ISUSE (LLM judge)\nISREL (cross-encoder scores)\nconfidence = 0.25×ISREL + 0.5×ISSUP + 0.25×ISUSE"]
        SR_Score --> SR_Gate{"Quality Gate\nAll thresholds met?"}
        SR_Gate -- "✅ Pass or max_depth" --> SROut(["Self-RAG Result\nAnswer + Sources\nConfidence Score\nReasoning Trace"])
        SR_Gate -- "❌ Fail → Repair Agent" --> SR_Retrieve
        SR_GreetResp --> SROut
    end

    subgraph CoRAG ["⭐  Co-RAG Pipeline  (core/co_rag.py)  — runs second"]
        CoRAGIn(["Same Raw Query + co_rag_content\nchat history (independent)"])
        CoRAGIn --> CR_Reform["Contextual Reformulation\n(co_rag history — isolated)"]
        CR_Reform --> CR_Greet{"2-Layer Greeting\nDetection"}
        CR_Greet -- "Greeting" --> CR_GreetResp["LLM: Greeting Response"]
        CR_Greet -- "Factual" --> CR_Retrieve["Step 1: Holistic Single-Shot\nRetrieval + Cross-Encoder"]
        CR_Retrieve --> CR_Gen["Step 2: Generator Mode A\n(Initial Draft)"]
        CR_Gen --> CR_Review{"Step 3a: Reviewer\n[VERIFIED?]"}
        CR_Review -- "VERIFIED or max_retries" --> CROut(["Co-RAG Result\nAnswer + Sources\nHorizontal Trace"])
        CR_Review -- "PARTIAL/ERROR → Mode B\nRefine + re-review" --> CR_Gen
        CR_GreetResp --> CROut
    end

    SROut --> UI
    CROut --> UI

    UI["🖥️  Dual-Tab UI\nTab 1: Self-RAG Answer\n        (sources · confidence_score · reasoning_trace)\nTab 2: Co-RAG Answer\n        (sources · horizontal_trace)"]

    UI --> UserOut(["Displayed to User"])

    %% Styling
    style SelfRAG fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style CoRAG fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    style UI fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

---

## Description

- Entry: the user's raw query enters `core/rag.py :: run_dual_rag()` — the single entry point that orchestrates both pipelines sequentially.
- Both pipelines receive the same raw query but maintain completely independent state: Self-RAG uses `self_rag_content` chat history, and Co-RAG uses `co_rag_content` chat history. They never share history.

**Self-RAG pipeline runs FIRST** (see Self-RAG diagram for full details):

- The query + `self_rag_content` history go through independent contextual query reformulation.
- Then 2-Layer Greeting Detection using the Self-RAG reformulated query.
- If greeting detected → the LLM generates a greeting response (stored as the Self-RAG result).
- If FACTUAL → run the full Self-RAG pipeline (Steps 1–5b) and return the answer with confidence score and reasoning trace.

**Co-RAG pipeline runs SECOND** (see Co-RAG diagram for full details):

- The same raw query + `co_rag_content` history go through independent contextual query reformulation (completely isolated from Self-RAG's history and reformulated query).
- Then 2-Layer Greeting Detection using the Co-RAG reformulated query.
- If greeting detected → the LLM generates a greeting response (stored as the Co-RAG result).
- If FACTUAL → run the full Co-RAG pipeline (Steps 0a–3b) and return the answer with sources and horizontal trace.

**Display:**

- Both pipeline results are collected and displayed side by side in the Dual-Tab UI:
  - Tab 1: Self-RAG answer, showing its sources, confidence score, and reasoning trace.
  - Tab 2: Co-RAG answer, showing its sources and horizontal trace.
- Each pipeline is fully independent — a greeting detected in one pipeline does not affect or short-circuit the other.
