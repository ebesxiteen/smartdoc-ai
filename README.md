# SmartDoc AI - Local NotebookLM-Inspired Document Intelligence System

> A privacy-first, open-source alternative to Google's NotebookLM. Query your documents with AI-powered semantic search, source-grounded answers, and automatic citations.

![Status](https://img.shields.io/badge/status-v1.0.0-green) ![License](https://img.shields.io/badge/license-MIT-blue) ![Python](https://img.shields.io/badge/python-3.8%2B-blue) ![Course](https://img.shields.io/badge/course-OSSD%20Spring%202026-orange)

## 🌟 Features

- **⚙️ Notebook Settings**: Customize memory, retrieval count, score thresholds, Self-RAG quality gates, and prompts independently per notebook natively via UI.
- **📚 Document Hub**: Manage PDF and Word documents, track real-time hardware status (RAM/VRAM), and configure intelligent settings with hardware-aware warnings.
- **🧠 Self-RAG Pipeline**: Multi-hop retrieval with automatic quality scoring (groundedness, relevance, utility) and a repair agent that retries failed searches with a new strategy — delivering significantly more accurate and grounded answers than a single-pass RAG.
- **🤝 Co-RAG Pipeline**: Collaborative Generator↔Reviewer architecture that iteratively refines answers through peer critique — the Generator drafts a holistic answer, the Reviewer diagnoses gaps and hallucinations, and the loop repeats until the answer is verified or the turn limit is reached.
- **⚡ Dual-Pipeline Comparison**: Both Self-RAG and Co-RAG run independently on every query, each with its own chat memory and reformulated context. Results are displayed side-by-side in two tabs for direct comparison.
- **🔍 Hybrid Search**: Intelligently retrieve relevant document sections using combined semantic search (FAISS embeddings) and BM25 full-text search with configurable weighting.
- **🤖 Grounded AI Responses**: Get answers strictly based on your documents with automatic source citations
- **🌐 Multi-language Support**: Ask questions in Vietnamese, English, or other languages and receive answers in your preferred language
- **💾 Persistent Storage**: All documents, chat history, and notes are saved to a local SQLite database
- **📝 Study Notes**: Save important Q&A pairs as notes to build a personalized study guide
- **⚡ Privacy First**: Runs entirely locally with [Ollama](https://ollama.com/)-your documents never leave your machine
- **🎨 NotebookLM-Inspired UI**: Clean, intuitive Streamlit interface mirroring the NotebookLM experience

## 🔧 Technology Stack

| Component | Technology | Purpose |
| ----------- | ----------- | --------- |
| **UI Framework** | [Streamlit](https://streamlit.io/) | Interactive web interface |
| **Orchestration** | [LangChain](https://python.langchain.com/) | RAG pipeline & prompt management |
| **Vector Database** | [FAISS](https://github.com/facebookresearch/faiss) | Fast similarity search on CPU |
| **Embeddings** | [Sentence Transformers](https://www.sbert.net/) | Multi-language text vectorization (paraphrase-multilingual-mpnet-base-v2) |
| **LLM** | [Ollama](https://ollama.ai/) + [Qwen2.5:7b](https://huggingface.co/Qwen/Qwen2.5-7B) | Local inference with Vietnamese support |
| **Database** | [SQLite](https://sqlite.org/) | Metadata and chat history persistence |
| **Document Processing** | [PyMuPDF](https://pymupdf.readthedocs.io/) & [python-docx](https://python-docx.readthedocs.io/) | Extract text from PDFs and Word documents |

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **Ollama**: Running locally with `qwen2.5:7b` model pulled
- **System Memory**: 8GB RAM minimum (16GB+ recommended for optimal performance)
- **GPU** (optional): CUDA-capable GPU for faster embeddings (CPU fallback available)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/dungtq2k5/smartdoc-ai.git
cd smartdoc-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Install & Start Ollama

```bash
# Download Ollama from https://ollama.ai/
# Then pull the Qwen2.5 model (required)
ollama pull qwen2.5:7b

# Start Ollama server (runs in background)
ollama serve
```

### 3. Run the Application

```bash
# In a new terminal, ensure venv is activated
source venv/bin/activate

# Start Streamlit
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## 📖 Usage Guide

### Creating a Notebook

1. Click **"+ Create New Notebook"** on the dashboard
2. Enter a notebook name (e.g., "Machine Learning Research")
3. Optionally add a description

### Customizing Notebook Settings (⚙️ New)

1. Select a notebook and open the **Notebook Settings** panel in the sidebar
2. Adjust retrieved chunk count, conversation memory, temperature, or add special System Prompts for this specific notebook
3. Click **"Apply Settings"** to securely persist your configuration

### Uploading Documents

1. Select a notebook from the dashboard
2. Go to the **Source Hub** (left sidebar)
3. Click **"Upload Documents"** and select one or more PDF or DOCX files
4. Review and confirm uploads—documents are automatically processed

### Asking Questions

1. Select which documents to query using checkboxes
2. Type your question in the chat input
3. SmartDoc retrieves relevant sections and generates a grounded answer
4. Click on **retrieved sources** beneath each answer to verify citations

### Saving Notes

1. After getting a good answer, click **"Save as Note"** below the response
2. Edit the note title if needed
3. Saved notes are compiled in the **Notes Panel** (right sidebar)

### Deleting Content

- **Remove a document**: Click the ⋮ menu next to a source → "Delete"
- **Delete a notebook**: Click the ⋮ menu next to notebook name → "Delete"
- **Clear chat history**: Click the ⋮ menu next to "Chat" title → "Delete chat history"

## ⚙️ Configuration

All tunable parameters are centralized in [core/configs.py](./core/configs.py):

```python
# RAG Tuning (adjust for inference quality vs speed)
RAG_RETRIEVAL_K: int = 8                     # Top K chunks to retrieve
RAG_RETRIEVAL_SCORE_THRESHOLD: float = 1.0  # Euclidean distance (0.0 to 2.0); Lower = stricter filtering
RAG_MAX_CTX_LEN: int = RAG_RETRIEVAL_K * 1000 # Characters sent to LLM
WEIGHT_SEMANTIC: float = 0.5                 # Hybrid search: semantic vs keyword balance
WEIGHT_BM25: float = 0.5                     # BM25 weight (keyword match)

# LLM Setup
OLLAMA_BASE_URL = "http://localhost:11434"  # Ollama server
LLM_MODEL_NAME = "qwen2.5:7b"
LLM_AVG_TEMP = 0.7               # Base temperature; Self-RAG spreads candidates across [0.1, LLM_AVG_TEMP * 1.5]

# Self-RAG Quality Gates
SELF_RAG_MAX_DEPTH: int = 2          # Max recursive repair hops before fallback
SELF_RAG_CANDIDATES: int = 3         # Diverse answer drafts generated per hop
SELF_RAG_MAX_RETRIES_PER_HOP: int = 2 # Sub-query rewrite retries before skipping
SELF_RAG_THRESHOLD_ISSUP: float = 0.70 # Groundedness gate (answer supported by docs?)
SELF_RAG_THRESHOLD_ISREL: float = 0.70 # Relevance gate (chunks match the query?)
SELF_RAG_THRESHOLD_ISUSE: float = 0.70 # Utility gate (answer satisfies user intent?)

# Co-RAG Collaboration
CO_RAG_MAX_RETRIES: int = 3          # Max Generator↔Reviewer turns (0 = no review)

# ...
```

## 📁 Project Structure

```txt
smartdoc-ai/
├── app.py                    # Main Streamlit entry point
├── core/
│   ├── configs.py           # Centralized configuration parameters
│   ├── rag.py               # Dual-pipeline orchestrator (run_dual_rag)
│   ├── self_rag.py          # Self-RAG orchestration pipeline (6-step multi-hop)
│   ├── co_rag.py            # Co-RAG pipeline (Generator↔Reviewer collaboration)
│   └── utils.py             # RAG pipeline & utility functions
├── db/
│   ├── setup.py             # Database schema initialization
│   └── crud.py              # SQLite database operations
├── middlewares/
│   └── db_middleware.py     # Input validation & sanitization
├── data/
│   ├── smartdoc.db          # SQLite database (auto-created)
│   └── vectorstores/        # FAISS indices (organized by notebook/source)
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── LICENSE                 # MIT License
└── CHANGELOG.md            # Version history
```

## 🐛 Troubleshooting

### **Issue: "Ollama Offline" error in sidebar**

**Solution:**

- Verify Ollama is running: `ollama serve` in a separate terminal
- Check Ollama is at `http://localhost:11434`: `curl http://localhost:11434/api/tags`
- Restart Streamlit: `streamlit run app.py`

### **Issue: CUDA out of memory error**

**Solution:**

- The app automatically falls back to CPU—this is normal on limited GPUs
- Reduce the number of selected documents (use checkboxes)
- Lower `RAG_RETRIEVAL_K` in [core/configs.py](./core/configs.py)

### **Issue: Slow embedding or inference**

**Solution:**

- First embeddings are slower; subsequent queries cache results
- Qwen2.5:7b performs inference at ~10-15 tokens/sec on CPU (single-threaded)
- For faster speeds, use a GPU or upgrade to a larger GPU memory

### **Issue: Documents not appearing after upload**

**Solution:**

- Check file size (PDFs > 50MB may time out)
- Verify the PDF is text-based (not image scans)
- Try uploading a smaller section of the PDF first

## 🧠 How It Works

### Two-Pipeline Architecture

#### Pipeline 1: Ingestion (Document Upload)

1. Extract text from PDF using PyMuPDF or Word using python-docx
2. Split text into overlapping chunks (RecursiveCharacterTextSplitter)
3. Embed chunks using sentence-transformers
4. Store embeddings in FAISS; metadata in SQLite

#### Pipeline 2: Query (User Question — Dual Pipeline via `run_dual_rag()`)

Every user query runs both pipelines independently and returns a combined result displayed in two tabs.

**Self-RAG (Vertical — multi-hop):**

1. **Intent routing**: Classify query as greeting (skip retrieval) or factual (proceed). Layer 1 uses regex; Layer 2 uses an LLM call for ambiguous inputs.
2. **Query reformulation**: Rewrite follow-up questions into standalone queries using Self-RAG's own chat history.
3. **Search planning**: LLM decomposes the query into 1–3 independent sub-queries covering different aspects.
4. **Hybrid retrieval + retry**: Run FAISS semantic + BM25 keyword search per sub-query; cross-encoder re-ranks results. Failed sub-queries are rewritten and retried up to `SELF_RAG_MAX_RETRIES_PER_HOP` times.
5. **Candidate generation**: Generate `SELF_RAG_CANDIDATES` diverse answer drafts at varied temperatures.
6. **Quality scoring**: An LLM judge scores each draft on groundedness (ISSUP), relevance (ISREL), and utility (ISUSE).
7. **Threshold gate**: Accept the best-scoring candidate if all three scores meet configured thresholds.
8. **Repair & retry**: If no candidate passes, a repair agent diagnoses the failure, generates a new search strategy, and retries the full pipeline — up to `SELF_RAG_MAX_DEPTH` recursive hops.
9. Stream best answer with source citations; persist confidence score and reasoning trace.

**Co-RAG (Horizontal — iterative peer review):**

1. **Query reformulation**: Rewrite follow-up questions into standalone queries using Co-RAG's own isolated chat history.
2. **Intent routing**: Same two-layer greeting detection as Self-RAG, using Co-RAG's own history.
3. **Holistic retrieval**: Single broad FAISS + BM25 pass with cross-encoder re-ranking — no sub-query decomposition.
4. **Initial generation (Mode A)**: Generator LLM drafts a comprehensive answer grounded in retrieved context.
5. **Generator↔Reviewer loop**: Reviewer LLM diagnoses gaps, hallucinations, and contradictions; Generator applies targeted fixes (Mode B). Repeats until `[STATUS: VERIFIED]` or `CO_RAG_MAX_RETRIES` turns exhausted.
6. Stream final verified answer with sources and full critique trace.

### Why Local-First?

✅ **Privacy**: Your documents never leave your machine
✅ **Offline**: Works without internet connectivity
✅ **Cost-Free**: No API charges or subscriptions
✅ **Customizable**: Full control over embeddings, LLM, and parameters

## 📊 Project Status

**Version**: 1.2.0 (Dual-Pipeline Release)
**Completion**: 84.6% (99/117 development tasks completed)
**Last Updated**: April 18, 2026

## 📚 Documentation

- [CHANGELOG.md](./CHANGELOG.md) — Version history and release notes

## 🎓 Academic Context

This project was developed as part of the **Open Source Software Development (OSSD)** course at **Đại học Sài Gòn** (Saigon University), Spring 2026.

**Learning Objectives:**

- Build a full-stack AI application with modern Python frameworks
- Implement RAG principles for grounded AI responses
- Work with vector databases and embeddings
- Deploy privacy-first ML systems using local LLMs
- Practice collaborative open-source development

## 📝 License

This project is licensed under the **MIT License**—see [LICENSE](./LICENSE) for details.

You are free to use, modify, and distribute this software for educational, commercial, or personal purposes.

## 🍴 Forking

This project is designed as a learning implementation for the OSSD course and is **not actively maintained for external contributions**.

**Feel free to fork!** You can take this code and make it your own:

- **Extend it**: Add features like Excel document support, OCR for scanned PDFs, or cloud deployment
- **Customize it**: Modify the UI, change the LLM model, or integrate different vector databases
- **Learn from it**: Use this as a reference for building your own RAG systems
- **Share improvements**: If you make significant improvements, feel free to share them as a reference

This codebase is open under the [MIT License](https://en.wikipedia.org/wiki/MIT_License), so you have full freedom to use and modify it for any purpose.

## 🙏 Acknowledgments

- **Google NotebookLM** — Inspiration for the UI/UX design
- **LangChain** — RAG orchestration framework
- **Ollama** — Local LLM infrastructure
- **FAISS** — Efficient vector search
- **Streamlit** — Interactive web framework
- **Qwen Team** — Excellent multilingual LLM

## 📧 Issues & Questions

If you encounter bugs or have questions, please check the [Issues tab](https://github.com/dungtq2k5/smartdoc-ai/issues) first. Feel free to open a new issue if you find a bug.

---

**Happy documenting! 📚✨**
