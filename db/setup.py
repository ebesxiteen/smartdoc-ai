import sqlite3
from core.configs import DB_ROOT_PATH, USER_ROLE_NAME, ASSISTANT_ROLE_NAME


def init_db(db_name: str = DB_ROOT_PATH, print_debug: bool = False) -> None:
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
            role VARCHAR(50) CHECK(role IN ('{USER_ROLE_NAME}', '{ASSISTANT_ROLE_NAME}')) NOT NULL,
            content TEXT NOT NULL,
            sources TEXT, -- Stored as JSON string
            found_answer INTEGER DEFAULT 1, -- 1=True (found), 0=False (not found)
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (notebook_id) REFERENCES notebooks (id) ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()
    if print_debug:
        print(f"✅ Database '{db_name}' initialized with NotebookLM schema.")
