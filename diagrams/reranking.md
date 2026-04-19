# Cross-Encoder Reranking Diagram ⚖️

```mermaid
flowchart TD
    Start(["Input: Candidate Chunks from Search Engine"]) --> HybridCheck{"Was Hybrid Search\nperformed?\n(weight_semantic > 0\nAND weight_bm25 > 0)"}

    HybridCheck -- "Yes" --> Dedup["Deduplication by Document ID\n(remove duplicates that appear in\nboth Semantic and BM25 results)"]
    HybridCheck -- "No" --> CountCheck

    Dedup --> CountCheck{"Total Chunks >=\nrag_final_context_k?"}

    CountCheck -- "No (too few chunks)" --> SkipRerank[/"Return All Chunks Directly\nto RAG Generator\n(skip reranking)"/]

    CountCheck -- "Yes" --> CrossEncode["Cross-Encoder Reranking\nScore each (query, chunk) pair\nraw logit stored in metadata\n['rerank_score']"]

    CrossEncode --> Sort["Sort Descending by rerank_score\nSelect Top rag_final_context_k chunks"]

    Sort --> FinalOut(["Output: Final rag_final_context_k Chunks\n→ RAG Generator"])

    SkipRerank --> FinalOut

    %% Styling
    style Dedup fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style CrossEncode fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style SkipRerank fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
```

---

## Description

- Input: candidate chunks from the search engine.
- If the hybrid search was performed (both `weight_semantic` > 0 and `weight_bm25` > 0) → deduplicate the candidate pool by document ID, because the semantic and BM25 retrievers can return overlapping documents.
- If the total deduplicated chunk count < `rag_final_context_k` → skip Cross-Encoder reranking entirely → return all chunks directly to the RAG generator.
- If the total chunk count >= `rag_final_context_k` → run Cross-Encoder reranking:
  - Score each (query, chunk) pair with the Cross-Encoder model.
  - Store the raw logit score in the document's metadata under `rerank_score`.
  - Sort all chunks descending by `rerank_score`.
  - Select the top `rag_final_context_k` chunks.
- Return the final chunks to the RAG generator.
