import sqlite3
import core.configs as cfg
from core.utils import debug_log


def init_db(db_name: str = cfg.DB_ROOT_PATH, print_debug: bool = False) -> None:
    # Connect to the local SQLite file
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Enable Foreign Key support for this connection
    cursor.execute("PRAGMA foreign_keys = ON;")

    # 1. Create Notebooks Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS notebooks (
            id TEXT PRIMARY KEY NOT NULL,
            name VARCHAR(255) NOT NULL,
            description VARCHAR(255),
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 2. Create Sources Table
    # file_path is globally unique (vectorstore paths are unique)
    # file_hash is unique per notebook (same file can be in different notebooks)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sources (
            id TEXT PRIMARY KEY NOT NULL,
            notebook_id TEXT NOT NULL,
            file_name VARCHAR(255) NOT NULL,
            file_type VARCHAR(50) NOT NULL DEFAULT 'pdf',
            file_path VARCHAR(255) UNIQUE NOT NULL,
            file_hash VARCHAR(255) NOT NULL,
            summary TEXT,
            suggested_questions TEXT, -- Stored as JSON array
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (notebook_id) REFERENCES notebooks (id) ON DELETE CASCADE,
            UNIQUE(file_hash, notebook_id)
        )
    """)

    # 3. Create Notes (Saved Notes) Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id TEXT PRIMARY KEY NOT NULL,
            notebook_id TEXT NOT NULL,
            title VARCHAR(255) NOT NULL,
            content TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (notebook_id) REFERENCES notebooks (id) ON DELETE CASCADE
        )
    """)

    # 4. Create Chat Messages Table (NEW)
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id TEXT PRIMARY KEY NOT NULL,
            notebook_id TEXT NOT NULL,
            role VARCHAR(50) CHECK(role IN ('{cfg.USER_ROLE_NAME}', '{cfg.ASSISTANT_ROLE_NAME}')) NOT NULL,
            content TEXT NOT NULL,
            -- Self-RAG specific columns
            self_rag_content TEXT DEFAULT NULL, -- Self-RAG answer text
            self_rag_sources TEXT DEFAULT NULL, -- Self-RAG sources (JSON array)
            self_rag_found_answer INTEGER DEFAULT 1, -- Self-RAG: 1=True (found), 0=False (not found)
            self_rag_confidence_score REAL DEFAULT NULL, -- Self-RAG confidence score (0.0-1.0)
            self_rag_reasoning_trace TEXT DEFAULT NULL, -- Self-RAG reasoning trace (JSON array)
            -- Co-RAG specific columns (NULL for user messages and pre-Co-RAG assistant messages)
            co_rag_content TEXT DEFAULT NULL, -- Co-RAG answer text
            co_rag_sources TEXT DEFAULT NULL, -- Co-RAG sources (JSON array)
            co_rag_found_answer INTEGER DEFAULT NULL, -- Co-RAG found_answer flag
            co_rag_reasoning_trace TEXT DEFAULT NULL, -- Co-RAG horizontal trace (JSON array)
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (notebook_id) REFERENCES notebooks (id) ON DELETE CASCADE
        )
    """)

    # 5. Create Notebook Settings Table
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS notebook_settings (
            id TEXT PRIMARY KEY NOT NULL,
            notebook_id TEXT NOT NULL UNIQUE,
            rag_final_context_k INTEGER DEFAULT {cfg.RAG_FINAL_CONTEXT_K},
            rag_rerank_top_n INTEGER DEFAULT {cfg.RAG_RERANK_TOP_N},
            rag_retrieval_min_results INTEGER DEFAULT {cfg.RAG_RETRIEVAL_MIN_RESULTS},
            rag_retrieval_score_threshold REAL DEFAULT {cfg.RAG_RETRIEVAL_SCORE_THRESHOLD},
            rag_max_chunk_len INTEGER DEFAULT {cfg.RAG_MAX_CHUNK_LEN},
            rag_chunk_overlap INTEGER DEFAULT {cfg.RAG_CHUNK_OVERLAP},
            rag_max_ctx_len INTEGER DEFAULT {cfg.RAG_MAX_CTX_LEN},
            max_msg_history INTEGER DEFAULT {cfg.MAX_MSG_HISTORY},
            llm_model_name VARCHAR(255) DEFAULT '{cfg.LLM_MODEL_NAME}',
            llm_num_ctx INTEGER DEFAULT {cfg.LLM_NUM_CTX},
            llm_avg_temp REAL DEFAULT {cfg.LLM_AVG_TEMP},
            personal_ctx TEXT,
            weight_semantic REAL DEFAULT {cfg.WEIGHT_SEMANTIC},
            weight_bm25 REAL DEFAULT {cfg.WEIGHT_BM25},
            self_rag_max_depth INTEGER DEFAULT {cfg.SELF_RAG_MAX_DEPTH},
            self_rag_candidates INTEGER DEFAULT {cfg.SELF_RAG_CANDIDATES},
            self_rag_max_retries_per_hop INTEGER DEFAULT {cfg.SELF_RAG_MAX_RETRIES_PER_HOP},
            self_rag_threshold_issup REAL DEFAULT {cfg.SELF_RAG_THRESHOLD_ISSUP},
            self_rag_threshold_isrel REAL DEFAULT {cfg.SELF_RAG_THRESHOLD_ISREL},
            self_rag_threshold_isuse REAL DEFAULT {cfg.SELF_RAG_THRESHOLD_ISUSE},
            co_rag_max_retries INTEGER DEFAULT {cfg.CO_RAG_MAX_RETRIES},
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (notebook_id) REFERENCES notebooks (id) ON DELETE CASCADE
        )
        """)

    conn.commit()
    conn.close()
    if print_debug:
        debug_log(
            "INFO", "🗄️", f'Database "{db_name}" initialized with NotebookLM schema.'
        )
