# Self-RAG Pipeline Diagram ⚙️

```mermaid
flowchart TD
    In(["User Query + self_rag Chat History"]) --> Reform["Contextual Query Reformulation\n(latest max_msg_history messages)"]

    Reform --> Greet{"2-Layer Greeting Detection"}

    Greet -- "Layer 1: Regex Match\nOR Layer 2: LLM → GREETING" --> GreetLLM["LLM: Generate Greeting Response\n(general knowledge + chat history)"]
    GreetLLM --> GreetOut(["Return Greeting to User ✓"])

    Greet -- "Layer 1: No Match\nLayer 2: FACTUAL" --> S1

    S1["Step 1: Generate Search Plan\nLLM splits reformulated query into sub-queries\n• Depth 0 → fresh plan from query\n• Depth > 0 → repair agent's new plan\n  (avoids previous failed search angles)"]

    S1 --> S2

    S2["Step 2: Multi-Hop Retrieval\n─────────────────────────────────────────\nFor EACH sub-query:\n  → Search Engine (Hybrid / Semantic / BM25)\n  → If 0 results: LLM rewrites sub-query\n     (up to self_rag_max_retries_per_hop)\n  → Accumulate into pool (dedup by hash)\n─────────────────────────────────────────\nAfter ALL sub-queries complete:\n  → Cross-Encoder Reranking on full pool\n  → Final rag_final_context_k chunks"]

    S2 --> S3

    S3["Step 3: Generate Candidates\nGenerate self_rag_candidates diverse answers\nVariable temperatures spread around llm_avg_temp\nInput: context chunks + reformulated query + chat history"]

    S3 --> S4

    S4["Step 4: Score and Pick Winner\n─────────────────────────────────────────\nFor each candidate:\n  LLM Judge  → ISSUP (groundedness)\n             → ISUSE (utility)\n  Cross-Enc. → ISREL (sigmoid of mean rerank_score)\n  Score = 0.25×ISREL + 0.50×ISSUP + 0.25×ISUSE\n─────────────────────────────────────────\nPick Winner (highest total score)\nTie-break L1: STATUS (DOC_ANSWER > GENERAL > MISSING)\nTie-break L2: Highest ISSUP\nTie-break L3: First index"]

    S4 --> S5

    S5{"Quality Gate\nISSUP ≥ threshold_issup?\nISREL ≥ threshold_isrel?\nISUSE ≥ threshold_isuse?"}

    S5 -- "✅ All gates pass" --> FinalOut
    S5 -- "❌ Any gate fails" --> DepthCheck

    DepthCheck{"self_rag_max_depth\nreached?"}
    DepthCheck -- "Yes → return best-so-far" --> FinalOut
    DepthCheck -- "No" --> S5b

    S5b["Step 5b: Repair Agent\n• Identify which gates failed\n• Diagnose root cause\n• LLM generates new sub-queries\n  (max 3, different search angles)\n• Oscillation guard: skip if identical plan"]

    S5b -- "depth + 1  ↩  Loop back to Step 1" --> S1

    FinalOut(["Return Final Answer\n+ Sources + Confidence Score\n+ Reasoning Trace"])

    %% Styling
    style S1 fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style S2 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style S3 fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style S4 fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    style S5 fill:#fff9c4,stroke:#f9a825,stroke-width:2px
    style S5b fill:#ffebee,stroke:#c62828,stroke-width:2px
```

---

## Description

- Entry: reformulated query + `self_rag_content` chat history.
- Run 2-Layer Greeting Detection (see Greeting Detection diagram):
  - If greeting detected in Layer 1 or Layer 2 → call the LLM to generate a greeting response using its general knowledge and the chat history → return to the user, end the Self-RAG pipeline.
  - If FACTUAL → proceed to the main Self-RAG loop.

The following steps run in a loop (up to `self_rag_max_depth + 1` iterations total):

**Step 1 – Search Plan Generation:**

- At depth 0: the LLM decomposes the reformulated query into multiple targeted sub-queries (each focused on a different search angle).
- At depth > 0 (after a repair): the repair agent's newly generated sub-queries replace the previous failed plan, targeting different search angles to avoid repeating the same failures.

**Step 2 – Multi-Hop Retrieval (per sub-query):**

- For each sub-query in the search plan:
  - Run the search engine (Hybrid / Semantic / BM25) to retrieve documents.
  - If the search returns 0 results → rewrite the sub-query using the LLM and retry, up to `self_rag_max_retries_per_hop` attempts.
  - If results are found (or retries exhausted) → add the documents to the shared pool.
  - Deduplication is applied incrementally as each sub-query's results are merged (by content hash).
- After ALL sub-queries have been processed:
  - Run Cross-Encoder Reranking on the full deduplicated pool.
  - Select the top `rag_final_context_k` chunks as the final retrieved context.

**Step 3 – Candidate Answer Generation:**

- Call the LLM `self_rag_candidates` times to generate diverse answer candidates.
- Each call uses a slightly different temperature (spread around `llm_avg_temp`) to encourage diversity.
- Input for each call: final retrieved context + reformulated query + chat history.

**Step 4 – Scoring and Winner Selection:**

- For each candidate answer:
  - LLM judge evaluates: ISSUP (groundedness — is the answer supported by the context?) and ISUSE (utility — is the answer useful and relevant to the question?).
  - Cross-Encoder scores from Step 2 are used to compute ISREL = mean(sigmoid(rerank_scores)) across all context documents.
  - Final confidence score = 0.25 × ISREL + 0.50 × ISSUP + 0.25 × ISUSE.
- Pick the winner (highest confidence score), using tie-breaking in order:
  - Layer 1: Prefer STATUS = DOC_ANSWER over DOC_GENERAL over DOC_MISSING.
  - Layer 2: Prefer the candidate with the higher ISSUP score.
  - Layer 3: Prefer the candidate with the lower index (first generated wins).

**Step 5a – Quality Gate:**

- Check if the winner meets all thresholds: ISSUP >= `self_rag_threshold_issup`, ISREL >= `self_rag_threshold_isrel`, ISUSE >= `self_rag_threshold_isuse`.
- If all gates pass → return the winner answer to the user with sources, confidence score, and reasoning trace. End the pipeline.
- If any gate fails:
  - If `self_rag_max_depth` is reached → return the best-so-far winner anyway. End the pipeline.
  - If `self_rag_max_depth` is not reached → proceed to Step 5b (Repair Agent).

**Step 5b – Repair Agent:**

- Diagnose which quality gates failed and why (e.g. low groundedness, low relevance).
- Generate a new search plan with different sub-queries (max 3) that target different angles, explicitly avoiding the previously failed strategies.
- Oscillation guard: if the new plan is identical to the previous one, skip the repair and return best-so-far.
- Increment depth by 1 and loop back to Step 1 with the new plan.
