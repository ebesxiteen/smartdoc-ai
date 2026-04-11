# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- **Re-ranking with Cross-Encoder**: Implemented the re-ranking of retrieved chunks using a Cross-Encoder model to improve the relevance of retrieved documents for answer generation. This involves many modifications from database schema (adding `rag_final_context_k` and `rag_rerank_top_n`) to the RAG pipeline logic in `app.py` and `core/utils.py`, as well as changes to the UI for configuring these parameters.

### Fixed

- **Debug Log Function Usage**: Corrected the `LOG_CATEGORIES` in `/core/utils.py` and optimized the usage of the `debug_log` function in `app.py` and `core/utils.py` to ensure there's no need to provide emojis as a parameter if the `log_type` is recognized. This streamlines the logging process and reduces redundancy in log statements.

## [v1.1.0] - 2026-04-08

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

[Unreleased]: https://github.com/dungtq2k5/smartdoc-ai/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/dungtq2k5/smartdoc-ai/compare/v1.0.0...1.1.0
[1.0.0]: https://github.com/dungtq2k5/smartdoc-ai/releases/tag/v1.0.0