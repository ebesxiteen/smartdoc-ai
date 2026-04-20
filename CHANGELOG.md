# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Breaking Changes

- **Self-RAG Replaces Linear RAG Pipeline**: The existing single-pass linear RAG process (`generate_answer()` in `core/utils.py`) has been entirely replaced by a multi-hop Self-RAG orchestration pipeline (`core/self_rag.py`). The query path, prompt structure, and response metadata format have all changed. See the **Self-RAG Pipeline** entry under _Added_ for the full architecture.
- **Dual-Pipeline Architecture — `run_dual_rag()` Replaces Direct `process_user_query()` Calls**: All query execution in `app.py` now goes through `core/rag.py::run_dual_rag()`, which runs both Self-RAG and Co-RAG fully independently and returns a combined 9-key result dict. Direct calls to `process_user_query()` from `app.py` have been removed.
- **Database Schema Migration Required**: Existing databases must be deleted and re-initialized via `python db/setup.py`:
  - `chat_messages` table: The old `sources`, `found_answer`, `confidence_score`, and `reasoning_trace` columns have been replaced by pipeline-prefixed columns: `self_rag_content`, `self_rag_sources`, `self_rag_found_answer`, `self_rag_confidence_score`, `self_rag_reasoning_trace`, `co_rag_content`, `co_rag_sources`, `co_rag_found_answer`, `co_rag_reasoning_trace`.
  - `notebook_settings` table: `llm_temp` column renamed to `llm_avg_temp`; six new Self-RAG control columns added: `self_rag_max_depth`, `self_rag_candidates`, `self_rag_max_retries_per_hop`, `self_rag_threshold_issup`, `self_rag_threshold_isrel`, `self_rag_threshold_isuse`; one new Co-RAG column added: `co_rag_max_retries`.
- **`LLM_TEMPERATURE` Config Renamed to `LLM_AVG_TEMP`**: The `LLM_TEMPERATURE` constant in `core/configs.py` has been renamed to `LLM_AVG_TEMP` to reflect that Self-RAG derives a spread of sampling temperatures from this base value when generating diverse candidate answers.

### Added

- **Self-RAG Pipeline** (`core/self_rag.py`): New module implementing a complete 6-step Self-RAG orchestration pipeline that replaces the old linear generate-once approach:
  - **Step 0 — Intent Routing**: Two-layer greeting/factual classification. Layer 1 uses regex patterns; Layer 2 falls back to an LLM call (`LAYER2_LLM_ROUTER_PROMPT`) for ambiguous inputs, short-circuiting retrieval entirely for greetings and chitchat.
  - **Step 1 — Search Planning**: The LLM decomposes the user query into 1–3 independent sub-queries (`SEARCH_PLANNER_PROMPT`) to cover multiple aspects of complex questions. Follow-up questions are first rewritten into self-contained standalone queries (`REFORMULATE_QUERY_PROMPT`) to preserve context without polluting vector search.
  - **Step 2 — Hybrid Retrieval with Surgical Retry**: Runs hybrid semantic (FAISS) + keyword (BM25) search with cross-encoder re-ranking per sub-query. Sub-queries that return zero results are rewritten up to `SELF_RAG_MAX_RETRIES_PER_HOP` times via `SUBQUERY_REWRITE_PROMPT` before being skipped, preventing total retrieval failure from one bad sub-query.
  - **Step 3 — Candidate Generation**: Generates `SELF_RAG_CANDIDATES` diverse answer drafts at varied temperatures spread across `[0.1, LLM_AVG_TEMP × 1.5]` to increase the chance one candidate passes all quality gates.
  - **Step 4 — Quality Scoring**: An LLM judge scores each candidate on three axes via `QUALITY_JUDGE_PROMPT`: **ISSUP** (groundedness — is every claim supported by retrieved context?), **ISREL** (relevance — derived from cross-encoder scores stored in document metadata), and **ISUSE** (utility — does the answer fully satisfy the user's intent?).
  - **Step 5a — Threshold Validation**: Accepts the highest-scoring candidate if all three scores meet their configurable thresholds (`SELF_RAG_THRESHOLD_ISSUP`, `SELF_RAG_THRESHOLD_ISREL`, `SELF_RAG_THRESHOLD_ISUSE`).
  - **Step 5b — Repair Agent**: If no candidate passes, a repair agent (`REPAIR_AGENT_PROMPT`) diagnoses the failure reason (low groundedness, low relevance, or low utility) and generates a new retrieval strategy. The full pipeline retries up to `SELF_RAG_MAX_DEPTH` recursive hops. If max depth is reached, the best available candidate is returned as a graceful fallback.
- **Self-RAG State Tracking** (`SelfRAGState` dataclass): A mutable state object persists across all recursive hops, tracking current depth, search history (to prevent oscillating sub-queries), the cumulative retrieval pool, a verbose `reasoning_trace` for UI transparency, and `confidence_metrics` from the last quality gate.
- **Self-RAG Configuration Parameters** (`core/configs.py`): Six new tunable constants, each with min/max/step/help metadata for UI rendering:
  - `SELF_RAG_MAX_DEPTH` (default: 2) — maximum recursive repair hops before fallback.
  - `SELF_RAG_CANDIDATES` (default: 3) — number of diverse answer drafts generated per hop.
  - `SELF_RAG_MAX_RETRIES_PER_HOP` (default: 2) — sub-query rewrite attempts before skipping.
  - `SELF_RAG_THRESHOLD_ISSUP` (default: 0.70) — minimum groundedness score to accept an answer.
  - `SELF_RAG_THRESHOLD_ISREL` (default: 0.70) — minimum relevance score gate.
  - `SELF_RAG_THRESHOLD_ISUSE` (default: 0.70) — minimum utility score gate.
- **Self-RAG Notebook Settings**: All six Self-RAG parameters are configurable per-notebook via the existing Notebook Settings UI panel and persisted to the `notebook_settings` table with full input validation in `middlewares/db_middleware.py`.
- **Self-RAG Confidence & Trace Persistence**: `confidence_score` (composite quality score) and `reasoning_trace` (step-by-step decision log) from each Self-RAG run are stored in `chat_messages` and reconstructed on chat history load for UI display.
- **Re-ranking with Cross-Encoder**: Implemented the re-ranking of retrieved chunks using a Cross-Encoder model to improve the relevance of retrieved documents for answer generation. This involves many modifications from database schema (adding `rag_final_context_k` and `rag_rerank_top_n`) to the RAG pipeline logic in `app.py` and `core/utils.py`, as well as changes to the UI for configuring these parameters.
- **Co-RAG Pipeline** (`core/co_rag.py`): New module implementing a fully independent Collaborative RAG pipeline as a 4-step Generator↔Reviewer orchestration engine:
  - **Step 0a — Query Reformulation**: Rewrites follow-up questions into standalone queries using Co-RAG's own isolated chat history (independent of Self-RAG's reformulated query).
  - **Step 0b — 2-Layer Intent Routing**: Detects greetings with the same two-layer approach as Self-RAG (regex → LLM fallback) but using Co-RAG's own history, short-circuiting to a Co-RAG greeting response if matched.
  - **Step 1 — Holistic Single-Shot Retrieval**: Performs one broad retrieval pass using the reformulated query (FAISS semantic + BM25 keyword, then cross-encoder re-ranking), unlike Self-RAG's multi-hop sub-query decomposition.
  - **Step 2 — Initial Generation (Mode A)**: The Generator LLM produces a comprehensive first-draft answer grounded in the retrieved context.
  - **Step 3 — Iterative Generator↔Reviewer Loop**: The Reviewer LLM diagnoses gaps, hallucinations, and contradictions in the draft; the Generator applies targeted redlines (Mode B). The loop exits when the Reviewer issues `[STATUS: VERIFIED]` or `co_rag_max_retries` turns are exhausted.
  - **Step 4 — Return**: Final draft is returned with the full critique history stored as the Co-RAG reasoning trace.
- **Co-RAG State Tracking** (`CoRAGState` dataclass): Mutable state object tracking current collaboration turn, all reviewer critiques, a verbose `horizontal_trace` of each generation/review step, the reformulated query, retrieved docs, and current draft.
- **Dual-Pipeline Orchestrator** (`core/rag.py`): New `run_dual_rag()` function as the single app-level entry point. Runs Self-RAG (`process_user_query()`) and Co-RAG (`co_rag_query()`) fully independently in sequence, returning a combined dict with 9 keys: `self_rag_content`, `self_rag_sources`, `self_rag_found_answer`, `self_rag_reasoning_trace`, `self_rag_confidence_score`, `co_rag_content`, `co_rag_sources`, `co_rag_found_answer`, `co_rag_reasoning_trace`.
- **Dual-Pipeline Chat UI**: Each assistant message is now displayed in two tabs — **⚡ Self-RAG (Vertical)** and **⭐ Co-RAG (Horizontal)** — allowing side-by-side comparison of both pipelines' answers, sources, and reasoning traces for the same query.
- **Isolated Chat History Per Pipeline**: Co-RAG builds its context from `co_rag_content` turns; Self-RAG from `self_rag_content` turns. Neither pipeline reads the other's history, ensuring fully independent memory and reformulation.
- **Co-RAG Configuration Parameter** (`core/configs.py`): One new tunable constant with min/max/step/help metadata:
  - `CO_RAG_MAX_RETRIES` (default: 3) — maximum Generator↔Reviewer collaboration turns before accepting the current draft.
- **Co-RAG Notebook Settings**: `co_rag_max_retries` is configurable per-notebook via the Notebook Settings UI and persisted to the `notebook_settings` table with full input validation in `middlewares/db_middleware.py`.
- **Co-RAG Reasoning Trace Persistence**: The `horizontal_trace` (full Generator↔Reviewer step log) from each Co-RAG run is stored in `chat_messages.co_rag_reasoning_trace` and reconstructed on chat history load for UI display.
- **Shared RAG Style Rules** (`SHARED_RAG_STYLE_RULES` in `core/configs.py`): A unified set of style and grounding rules injected into both Self-RAG and Co-RAG prompts to enforce a consistent brand voice (language matching, no structural headers, grounding, conciseness, technical term preservation) across both pipelines.
- **Enhanced Settings Warnings for Self-RAG & Co-RAG**: The Notebook Settings validation panel now covers Self-RAG performance warnings (complex search, perfectionist trap, redundant retries, high branching, loose evidence gate), Co-RAG dual-pipeline fatigue warnings, and cross-pipeline latency alerts.
- **System Architecture Diagrams** (`diagrams/`): Added 7 Mermaid flowchart diagrams with written step-by-step descriptions covering every component of the dual-pipeline RAG system:
  - `contextual-reformulation.md` — Query reformulation flow using per-pipeline isolated chat history.
  - `greeting-detection.md` — 2-layer greeting detection: regex fast-path (Layer 1) + LLM intent classifier (Layer 2).
  - `search.md` — Search engine modes: pure semantic, pure BM25, and hybrid RRF with proportional `k` split by weight.
  - `reranking.md` — Cross-Encoder reranking with hybrid deduplication and conditional skip logic.
  - `self-rag.md` — Full Self-RAG pipeline (search plan → multi-hop retrieval with surgical retry → candidate generation → quality scoring → repair loop).
  - `co-rag.md` — Full Co-RAG pipeline (holistic retrieval → Generator Mode A → Generator↔Reviewer loop).
  - `rag(system).md` — End-to-end dual-pipeline system view: `run_dual_rag()` sequential execution and dual-tab UI display.
- **Markdown-First Ingestion Pipeline** (`core/utils.py`): Complete rewrite of the document ingestion stage. Three new helpers power the pipeline:
  - `_pdf_to_markdown()` — extracts per-page Markdown from PDFs using PyMuPDF's `page.get_text("dict", sort=True)`. Heading levels (`#`, `##`, `###`) are inferred by comparing each block's dominant font size against the per-page median (≥1.5×, ≥1.25×, ≥1.1×). Bold and italic spans are detected via font flags (bit 4 = bold, bit 1 = italic) and rendered as `**text**` / `*text*`.
  - `_docx_to_markdown()` — walks the DOCX body in document order using `qn("w:p")` / `qn("w:tbl")` lxml tag matching (no private `_element` attributes). Heading styles map to `#`–`######`; list paragraphs map to `- ` / `1. `; tables render as GitHub-Flavored Markdown pipe tables.
  - `_clean_markdown_text()` — normalizes the converted Markdown: removes lines containing only a digit (PDF page-number artifacts) and collapses three or more consecutive blank lines to two.
  - `load_and_chunk_file()` is rewritten to use `MarkdownHeaderTextSplitter` (`#`/`##`/`###`/`####`, `strip_headers=False`) as the primary splitter so each chunk is bounded by document structure and retains its heading context. Any section chunk that still exceeds `chunk_size` is further split by `RecursiveCharacterTextSplitter` (`["\n\n", "\n", " ", ""]`) as a fallback. `page_content` retains all Markdown syntax so downstream `st.markdown()` calls render rich formatting in source citations.
- **Ingestion Pipeline Diagram** (`diagrams/ingestion.md`): New Mermaid `flowchart TD` diagram covering the full ingestion flow — file-type detection by magic bytes, MD5 duplicate detection, PDF/DOCX→Markdown conversion with technical detail (font-size heading inference, qn() body walk), `_clean_markdown_text` normalization, `MarkdownHeaderTextSplitter` primary + `RecursiveCharacterTextSplitter` fallback chunking, metadata enrichment, HuggingFace embedding (GPU/CPU), per-source FAISS index creation and save, session vectorstore merge, SQLite persistence, and Self-RAG chain rebuild. Includes colour-coded `classDef` styling and a prose description section, consistent with the other diagrams.
- **Native Markdown Source Citation Rendering** (`app.py`): All five source-citation blocks replaced from `unsafe_allow_html` HTML-injection (`<div class='source-citation'>`) to native Streamlit components — `st.markdown()` for the header line and `st.container(border=True)` + `st.markdown(source["content"])` for the chunk body. Source content is now rendered as rich Markdown, preserving all heading, list, and table structure from the Markdown-first ingestion pipeline.

### Changed

- **`LLM_AVG_TEMP` Replaces `LLM_TEMPERATURE`**: The base LLM temperature constant has been renamed to `LLM_AVG_TEMP` to clarify its role as an average from which Self-RAG derives a spread of temperatures for candidate generation, rather than a single fixed inference temperature.
- **`retrieve_quality_chunks` Signature**: Removed the generic `settings: Dict` parameter; the function now accepts explicit `weight_semantic: float` and `weight_bm25: float` keyword arguments. All callers in `core/utils.py`, `core/self_rag.py`, and `core/co_rag.py` have been updated to pass these values explicitly.
- **`_reformulate_query_with_history` Renamed to `reformulate_query_with_history`**: The leading underscore has been removed to reflect that this function is part of the public API consumed by `core/co_rag.py`.
- **Settings Access — Fail-Loud on Missing Keys** (`app.py`, `core/self_rag.py`, `core/co_rag.py`, `core/rag.py`): All `settings.get("key", fallback_default)` calls replaced with `settings["key"]` direct access across all pipeline files (10 sites in `app.py`, 12 in `core/self_rag.py`, 6 in `core/co_rag.py`, 3 in `core/rag.py`). `load_notebook_settings()` now guarantees a fully-populated dict — every key from `get_default_notebook_settings()` is always present and `NULL` DB values are replaced with config defaults — so a missing key raises `KeyError` immediately rather than silently falling back to a stale default. This prevents silent misconfiguration when a new settings key is added to the schema.
- **`format_context_with_sources()` Markdown Preservation** (`core/utils.py`): Context formatting now preserves Markdown structure by collapsing only excess blank lines (`re.sub(r"\n{3,}", "\n\n", doc.page_content.strip())`) instead of the previous `" ".join(doc.page_content.split())` which flattened all whitespace. Heading hierarchy, list structure, and table formatting from the Markdown-first pipeline are now visible to the LLM in generated context.
- **Source Citation Snippet Length** (`core/self_rag.py`, `core/co_rag.py`): Increased source preview truncation from 200 to 400 characters to show more meaningful context in the UI for Markdown-formatted chunks.

### Fixed

- **Hybrid Search Proportional `k` Split** (`core/utils.py`): Fixed a critical bug where both the semantic and BM25 retrievers in hybrid mode each fetched `k * 2` documents regardless of their configured weights. The retrievers now split `rag_rerank_top_n` proportionally: `sem_k = max(1, round(k × weight_semantic))` and `bm25_k = max(1, round(k × weight_bm25))`. For example, with `rag_rerank_top_n = 10` and weights `0.7 : 0.3`, semantic now fetches 7 and BM25 fetches 3 instead of both fetching 20.
- **Debug Log Guard in `process_user_query`** (`core/utils.py`): Moved the `debug_log("Gathering sources…")` call inside the `if not is_general_answer:` guard so it no longer executes unconditionally on general-knowledge answers.
- **Debug Log Function Usage**: Corrected the `LOG_CATEGORIES` in `/core/utils.py` and optimized the usage of the `debug_log` function in `app.py` and `core/utils.py` to ensure there's no need to provide emojis as a parameter if the `log_type` is recognized. This streamlines the logging process and reduces redundancy in log statements.
- **Co-RAG Reviewer Permanent `PARTIAL_VERIFIED` Loop** (`core/configs.py`): Rewrote `CO_RAG_REVIEWER_PROMPT` to fix a behavioral defect where the Reviewer almost never issued `[STATUS: VERIFIED]`, causing every query to exhaust all `CO_RAG_MAX_RETRIES` turns without converging. Root causes: (1) the old prompt opened with "Identify specific gaps" which biased the Reviewer toward finding problems even in adequate drafts; (2) `VERIFIED` required the draft to "fully address… No further revision needed" — a bar too high for a 7B model; (3) no convergence rule existed to force termination after repeated partial fixes. The new prompt uses assessment-first framing, lowers the `VERIFIED` bar to "adequately answers with reasonable grounding in context", and adds mandatory convergence rules: on Turn 2 or later, if prior critiques have been substantially addressed and no new critical factual issues exist, the Reviewer MUST issue `[STATUS: VERIFIED]` rather than re-issuing `[STATUS: PARTIAL_VERIFIED]` for the same unresolved point.

## [1.1.0] - 2026-04-08

### Added

- **Hybrid Search**: Implemented a hybrid search approach combining semantic search (FAISS) and keyword search (BM25) with configurable `semantic_weight` and `bm25_weight` parameters to optimize retrieval accuracy.
- **Structured Debug Logging**: Enhanced internal debug logging to use a consistent, structured category-based format.
- **Comprehensive Documentation Suite**: Added comprehensive docstrings to all functions in `app.py` and `core/utils.py` and updated existing internal templates and logging guides.
- **Local Hardware Dashboard**: Display local machine hardware capabilities (RAM, VRAM, OS, CPU, GPU) in the `Source Hub` section.
- **Hardware-Aware Warnings**: Added intelligent warning messages based on local machine hardware limitations when configuring notebook settings (e.g., warning if trying to load a massive model on limited VRAM).
- **Word Document Support**: Upload and process `.docx` files alongside PDFs.
- **Answer Generation Resume**: Auto-resume answer generation if the process is interrupted (e.g., if the user accidentally refreshes the page or unselects all resources during generation).
- **Confirm Dialog For Deletions**: Added confirmation dialogs for all deletion actions (notebooks, sources, notes, chat history) to prevent accidental data loss.
- **Notebook Settings Feature**: Implemented a settings panel for each notebook allowing users to configure parameters like embedding model, FAISS K value, and summary generation strategy on a per-notebook basis.

### Changed

- **UI Theme Adaptation**: Refactored the app's CSS to remove hardcoded color values and instead utilize Streamlit's built-in theme variables, ensuring proper display in both light and dark modes without UI elements becoming invisible.
- **Saved Notes Section**: Moved the `Saved Notes` section into a collapsible expander within the sidebar to improve organization and user experience, allowing users to easily access their notes without taking up constant space in the main content area.
- **Toast Messages Styling**: Added visually appealing emojis to Streamlit toast messages for better user feedback and styling.
- **Personal Context Setting**: Renamed `notebook_setting.sys_prompt_override` to `personal_ctx` for better clarity in the codebase and UI.
- **Settings Save Optimization**: Re-engineered the 'Apply Settings' logic to selectively update system state, minimizing unnecessary RAG pipeline reloads.
- **RAG Pipeline Conversational Memory**: Rebuilt the LangChain pipeline to natively ingest `chat_message_history`, allowing the LLM to understand contextual follow-up questions while cleanly bypassing FAISS retrieval for standard greetings.
- **Configuration Refactoring**: Renamed global configuration variables in `configs.py` for better consistency and readability (e.g., `RAG_MAX_CONTEXT_LENGTH` to `RAG_MAX_CTX_LEN`) and globalized the number of first chunks used for summaries and suggested questions.
- **System & Generation Prompts Optimization**: Modified the base system prompt to enhance LLM performance and globalized the prompts for 'summary' and 'suggested questions'.
- **App & Database Performance**: Optimized startup and runtime performance by deferring heavy module loads using local imports, and optimized SQLite delete operations for lower latency.
- **Codebase Clean-up**: Removed redundant code in `utils.py` and `app.py` related to chat history management and prompt construction.
- **Suggested Questions UI**: Improved display of `source.suggested_questions` by randomly picking up to 3 questions across all available resources rather than just the first resource.
- **Summary Generation Strategy**: Modified the `source.summary` generation process to use a more efficient "Top-K Slicing" method instead of semantic search, improving summary relevance.
- **UI Processing Uploaded Sources Enhancement**: Use progress bar loading while processing uploaded sources to improve user experience and provide feedback on long-running operations.
- **Emojis For Displayed Messages**: Added visual appealing emojis to various user-facing messages to enhance the UI and make it more engaging (e.g., success, error, info states).

### Fixed

- **FAISS Score Threshold Range**: Fixed the `RAG_RETRIEVAL_SCORE_THRESHOLD` input constraints to properly restrict values between `0.0` and `Infinity` (valid bounds for FAISS Euclidean distance search).
- **Settings UI Reset UX**: Resolved an issue where Streamlit's slider and input components failed to auto-reset to DB values when the user clicked 'Reset to Default'.
- **Ghost Toasts Fix**: Fixed missing toast success messages when a user clicked 'Reset to Default' or 'Apply Settings' due to backend execution halts (`st.rerun()`).
- **Production Logging Noise**: Added missing `print_debug` condition checks to various debug logs in `app.py` to prevent stdout cluttering in production.
- **Input Text Truncation Handling**: Prevented silent truncation of user input before saving to the database by throwing an explicit error when limits are exceeded. Ensured LLM-generated fields like `source.summary` and `source.suggested_questions` are not restricted by truncation, preserving data integrity.
- **Middle Floating `Add Note` Button**: Repositioned the "Add Note" button to a fixed position at the bottom center of the section to improve visibility and accessibility, especially when there are no existing notes.
- **Unattended Suggested Questions Truncation Display**: Remove auto truncated display of suggested questions and always show full questions in the UI, ensuring users can see all generated suggestions without confusion.
- **Unnecessary Retrieval Process During Greetings**: Prevented the system from invoking the "Retrieval Process" to find relevant chunks when the user's input is detected as a greeting (e.g., "Hello", "Hi"), which caused unnecessary processing and delayed response times. Implemented a check to bypass retrieval and directly generate a greeting response in such cases.
- **LLM Empty Response Handling**: Added a fallback answer for cases when the LLM generates an empty response after tag removal, ensuring users receive a meaningful message instead of an empty response and preventing database insertion errors.
- **Robust Tag Removal**: Implemented robust regex-based tag removal to ensure that any tags like `[STATUS: DOC_ANSWER]` are completely removed from the LLM's response before being displayed to users, preventing confusion and improving the clarity of answers.

## [1.0.0] - 2026-03-25

### Project Status

- **Completion Rate**: 95.2% (79/83 development tasks completed)
- **Development Time**: Spring 2026 semester (Open Source Software Development course)
- **Institution**: Đại học Sài Gòn (Saigon University), Faculty of Information Technology

### Added

#### Core Features

- **Multi-Notebook System**: Create, manage, and switch between multiple document collections
- **PDF Document Ingestion**: Upload and process PDF files with automatic text extraction via PyMuPDF
- **Semantic Search**: FAISS-powered vector similarity search with configurable K and score thresholding
- **Grounded AI Responses**: LLM answers strictly constrained to uploaded documents using custom prompt templates
- **Source Citations**: Automatic tracking and display of which documents contributed to each answer
- **Vector Persistence**: FAISS indices stored locally for fast re-querying without re-embedding
- **Chat History**: Persistent conversation logs stored in SQLite with role-based messaging (user/assistant)
- **Study Notes**: Save important Q&A pairs as personal study guides

#### UI/UX Features

- **NotebookLM-Inspired Interface**: Clean, two-pane Streamlit layout matching Google's NotebookLM design
- **Source Hub (Sidebar)**: Dedicated document management panel with file upload, rename, and delete capabilities
- **Smart Document Selection**: "Select all sources" checkbox with visual feedback for active documents
- **Scrollable Chat Interface**: Auto-scrolling conversation history with fixed input box at bottom
- **Response Visualization**: Assistant responses in light gray cards, user messages in right-aligned bubbles
- **Source Details Panel**: Expandable sections showing exact chunks used to generate each answer
- **Document Metadata Display**: File names, upload dates, and optional summaries for each source

#### Language & Localization

- **Multi-language Query Support**: Automatically detect user's query language (English, Vietnamese, Chinese, etc.)
- **Language-Aware Responses**: Answer in the detected language while citing sources in original document language
- **Qwen2.5:7b LLM**: Superior multilingual reasoning, especially for Vietnamese
- **Greeting Pattern Detection**: Intelligent conversation initiation with language-specific greetings

#### Data Management

- **SQLite Backend**: Local database storing notebooks, sources, chat messages, and notes
- **File Hash Deduplication**: MD5-based duplicate detection to prevent uploading the same file twice per notebook
- **Foreign Key Constraints**: Referential integrity (CASCADE deletes prevent orphaned data)
- **Metadata Tracking**: Created/updated timestamps for all entities

#### Validation & Safety

- **Input Validation Middleware**: Sanitization of user input before database writes
- **Length Limits**: Configurable max lengths for notebook names, filenames, descriptions, etc.
- **XSS Prevention**: HTML escaping for all user-generated content in UI
- **CUDA Fallback**: Graceful GPU→CPU fallback when CUDA memory is exhausted

#### Developer Experience

- **Centralized Configuration**: All tunable parameters in `core/configs.py` (no hardcoding)
- **Modular Architecture**: Separation of concerns (core utils, DB operations, middleware, UI)
- **Rich Documentation**: Comprehensive code comments and architecture guides
- **Debug Logging**: Optional print statements via `print_debug` flags

### Fixed

#### Critical Bug Fixes

- **Session State Dictionary Collision**: Resolved `st.session_state.documents` key collision by switching from filename-based to UUID-based keying
- **File Hash Deduplication**: Implemented MD5-based checking with notebook-scoped uniqueness constraints to prevent duplicate uploads
- **Chat History Optimization**: Optimized LLM context window (increased from 1 message to 10 messages); fixed prompt truncation issues
- **CUDA Out-of-Memory Handling**: Added CPU fallback in `try_load_embeddings()` with success/failure logging
- **Suggested Questions Interactivity**: Implemented `pending_query` session state flag to trigger chat on question selection

#### UI/UX Fixes

- **Chat Message Layout**: Completely rewrote user message rendering with right-aligned flexbox HTML/CSS
- **File Upload Widget Auto-Clear**: Auto-increment `file_uploader_key` after batch processing to clear upload widget
- **Avatar Removal**: Removed user "U" and assistant 🤖 avatars for cleaner interface
- **Vector Store API Migration**: Migrated from deprecated `get_relevant_documents()` to `similarity_search()` method

#### Data Integrity Fixes

- **Vectorstore Isolation**: Each source gets its own FAISS index at `./data/vectorstores/{notebook_id}/{source_id}/`
- **Cascade Deletion**: Deleting a notebook now properly removes all associated sources, notes, and chat history
- **Metadata Consistency**: Source metadata (file_hash, file_path) always synchronized with actual FAISS indices

### Technical Details

#### Ingestion Pipeline

```txt
PDF Upload → PyMuPDF Extract → RecursiveCharacterTextSplitter
→ HuggingFace Embed (768-dim) → FAISS Store → SQLite Metadata
```

#### Query Pipeline

```txt
User Question → Embed (same model) → FAISS Similarity Search (top-K)
→ Score Threshold Filter → LangChain Prompt Augmentation
→ Ollama Qwen2.5 LLM → Streamed Response + Citations
```

### Performance Characteristics

- **Embedding Speed**: ~100-200 chunks/minute on NVIDIA GPU; ~10-20 chunks/minute on CPU
- **Search Latency**: <100ms for FAISS similarity search (in-memory index)
- **LLM Inference**: ~10-15 tokens/second on CPU; faster with GPU acceleration
- **Database**: SQLite queries complete in <10ms for typical notebook sizes (<100 sources)

### Known Limitations

- **Copy to Clipboard**: Streamlit's HTML sandboxing prevents native clipboard access
- **Red Delete Buttons**: Streamlit's button styling system doesn't support custom danger state
- **General Knowledge Fallback**: Fallback for off-document questions not yet implemented
- **Multi-Turn Conversation Context**: Conversation context awareness for follow-up questions pending implementation

### Dependencies Included

See [requirements.txt](./requirements.txt) for full dependency list. Key packages:

- **streamlit** ≥1.40 — UI framework
- **langchain** ≥0.2.0 — RAG orchestration
- **langchain-ollama** ≥0.2.0 — Ollama integration
- **langchain-huggingface** ≥0.1.0 — Embeddings
- **faiss-cpu** ≥1.8.0 — Vector search
- **pymupdf** ≥1.24.0 — PDF processing
- **sentence-transformers** ≥3.0.0 — Embedding models

### Testing & Validation

#### Acceptance Criteria Met

✅ Vectorstore isolation by source — Each document has independent FAISS index
✅ Citation accuracy — Retrieved chunks match LLM-cited sources
✅ Embedding caching — Repeated queries use cached FAISS indices
✅ Duplicate detection — Hash-based prevention of duplicate uploads
✅ Language detection — Queries properly identified as multilingual
✅ UI responsiveness — Sub-500ms response to user interactions

#### Code Quality Metrics

✅ Type hints on all public functions
✅ Input validation via middleware (no raw DB writes)
✅ Modular code structure (utils, db, middleware separation)
✅ Error handling with user-friendly messages
✅ Comprehensive inline documentation

### Breaking Changes

None (first release)

### Migration Guide

N/A (initial release with auto-schema creation)

### Deprecations

None

### Security Considerations

- ✅ No hardcoded API keys (all configs externalized)
- ✅ SQL injection prevention via parameterized queries and middleware
- ✅ XSS prevention via HTML escaping
- ✅ Local-only data (no external API calls beyond Ollama)
- ⚠️ Input length validation prevents denial-of-service attacks
- ⚠️ Note: Ollama server should be firewalled (no authentication by default)

### Future Roadmap (Post-1.0.0)

- [ ] **v1.1.0**: General knowledge fallback; conversation context awareness
- [ ] **v1.2.0**: Word/Excel document support; PDF OCR for scanned documents
- [ ] **v1.3.0**: Web deployment (Docker containerization)
- [ ] **v1.4.0**: User authentication; multi-user notebook sharing
- [ ] **v2.0.0**: Integration with cloud LLM providers (OpenAI fallback)

### Contributors

- SmartDoc AI Development Team (Đại học Sài Gòn, Spring 2026 OSSD Course)

### Links

- [Project Repository](https://github.com/dungtq2k5/smartdoc-ai)
- [Issue Tracker](https://github.com/dungtq2k5/smartdoc-ai/issues)

---

[Unreleased]: https://github.com/dungtq2k5/smartdoc-ai/compare/1.1.0...HEAD
[1.1.0]: https://github.com/dungtq2k5/smartdoc-ai/compare/1.0.0...1.1.0
[1.0.0]: https://github.com/dungtq2k5/smartdoc-ai/releases/tag/1.0.0