# Search Engine Diagram 🔍

```mermaid
flowchart TD
    Start(["Input: Reformulated Query"]) --> Strategy{"Check Weights:\nweight_semantic vs weight_bm25"}

    %% Strategy Branching
    Strategy -- "weight_semantic > 0\nweight_bm25 = 0" --> PureSem["Pure Semantic Search\nk = rag_rerank_top_n"]
    Strategy -- "weight_semantic = 0\nweight_bm25 > 0" --> PureBM["Pure BM25 Search\nk = rag_rerank_top_n"]
    Strategy -- "Both > 0" --> Hybrid["Hybrid Search (RRF)\nsemantic_k = round(rag_rerank_top_n × weight_semantic)\nbm25_k = round(rag_rerank_top_n × weight_bm25)\ne.g. top_n=10, weights=0.7:0.3 → sem=7, bm25=3"]

    %% Semantic / Hybrid path
    PureSem --> SemFilter["Apply rag_retrieval_score_threshold\n(filter low-similarity semantic chunks)"]
    Hybrid --> SemFilter
    SemFilter --> SemCount{"Results <\nrag_retrieval_min_results?"}
    SemCount -- "Yes" --> SemFallback["Fallback: Return top rag_retrieval_min_results\nfrom semantic results\n(ignore threshold)"]
    SemCount -- "No" --> Pool
    SemFallback --> Pool

    %% BM25 path
    PureBM --> BMFilter["Apply rag_retrieval_score_threshold\n(cross-check BM25 docs with semantic scores)"]
    BMFilter --> BMCount{"Results <\nrag_retrieval_min_results?"}
    BMCount -- "Yes" --> BMFallback["RELAX Fallback:\nPure Semantic Search\n(ignore threshold)\nk = rag_retrieval_min_results"]
    BMCount -- "No" --> Pool
    BMFallback --> Pool

    %% Final check
    Pool["Candidate Chunks Pool"] --> FinalCheck{"Total Chunks >=\nrag_final_context_k?"}
    FinalCheck -- "Yes → send to reranking" --> ToRerank[/"Pass to Cross-Encoder Reranking"/]
    FinalCheck -- "No → skip reranking" --> DirectReturn[/"Return All Chunks\ndirectly to RAG Generator"/]

    %% Styling
    classDef semantic fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef bm25 fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef fallback fill:#ffebee,stroke:#c62828,stroke-width:2px;
    classDef pool fill:#f1f8e9,stroke:#33691e,stroke-width:2px;

    class PureSem,SemFilter semantic;
    class PureBM,BMFilter bm25;
    class SemFallback,BMFallback fallback;
    class Hybrid,Pool process;
```

---

## Description

- Input: reformulated query.
- Look at the `weight_semantic` and `weight_bm25` parameters to determine which type of search to perform:
  - If `weight_semantic` > 0 and `weight_bm25` = 0 → perform **pure semantic search**, fetching up to `rag_rerank_top_n` documents.
  - If `weight_semantic` = 0 and `weight_bm25` > 0 → perform **pure BM25 keyword search**, fetching up to `rag_rerank_top_n` documents.
  - If both `weight_semantic` > 0 and `weight_bm25` > 0 → perform **hybrid search** (Reciprocal Rank Fusion of semantic + BM25):
    - Semantic retriever fetches: `round(rag_rerank_top_n × weight_semantic)` documents (e.g. top_n=10, weight=0.7 → 7 docs).
    - BM25 retriever fetches: `round(rag_rerank_top_n × weight_bm25)` documents (e.g. top_n=10, weight=0.3 → 3 docs).
    - Results from both are fused and re-ranked by RRF scoring.

**After pure semantic search or hybrid search:**

- Filter the retrieved documents from the semantic component by `rag_retrieval_score_threshold` (L2 distance; lower = better).
- If the filtered results count < `rag_retrieval_min_results` → fallback: return the top `rag_retrieval_min_results` unfiltered documents from semantic search (ignore threshold).

**After pure BM25 search:**

- Cross-check BM25 results against semantic similarity scores so that `rag_retrieval_score_threshold` can meaningfully filter them (without this, all BM25 docs get score 0.0 and always pass).
- Apply the score threshold filter.
- If the filtered results count < `rag_retrieval_min_results` → RELAX fallback: execute pure semantic search (ignoring the threshold) to fetch the top `rag_retrieval_min_results` documents.
- If the initial BM25 search returns 0 documents → RELAX fallback is triggered immediately.

**After gathering results from any search mode:**

- Pool all candidate chunks.
- If total chunks >= `rag_final_context_k` → pass to Cross-Encoder Reranking for refinement (see Reranking diagram).
- If total chunks < `rag_final_context_k` → return all chunks directly to the RAG generator, skip reranking.
