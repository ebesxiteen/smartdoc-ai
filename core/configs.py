from langchain_core.prompts import PromptTemplate

APP_NAME = "SmartDoc AI"
APP_FULLNAME = "SmartDoc - Your AI Research Assistant"

DB_NAME = "smartdoc"
DB_ROOT_PATH = f"./data/{DB_NAME}.db"

USER_ROLE_NAME: str = "user"
ASSISTANT_ROLE_NAME: str = "assistant"
NOTEBOOK_DEFAULT_NAME: str = "Untitled Notebook"
SOURCE_DEFAULT_NAME: str = "Untitled Document"
NOTE_DEFAULT_TITLE: str = "Untitled Note"

MAX_NOTEBOOK_NAME_LEN = 100
MAX_DESCRIPTION_LEN = 300
MAX_FILENAME_LEN = 200
MAX_NOTE_TITLE_LEN = 150
MAX_SOURCE_SUMMARY_LEN = 1000
MAX_SUGGESTED_QUESTION_LEN = 300

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# ============================================================================
# RAG RETRIEVAL CONFIGURATIONS - Optimize these for your use case
# ============================================================================
# Maximum number of chunks to retrieve (higher = more context but may cause CUDA OOM)
RAG_RETRIEVAL_K = 8

# Minimum results to guarantee even if quality threshold filters out too much
RAG_RETRIEVAL_MIN_RESULTS = 1

# Similarity score threshold for quality filtering (FAISS distance: lower = better match)
# Typical range: 5.0-15.0. Lower = stricter filtering (fewer chunks but higher quality)
RAG_RETRIEVAL_SCORE_THRESHOLD = 15.0

# Max length per chunk in characters (prevents extremely long single chunks)
RAG_MAX_CHUNK_LENGTH = 1000

# 20% of chunk length overlap between chunks to preserve context
RAG_CHUNK_OVERLAP = round(.2 * RAG_MAX_CHUNK_LENGTH)

# Max total context length in characters to send to LLM (prevents CUDA OOM)
RAG_MAX_CONTEXT_LENGTH = RAG_RETRIEVAL_K * RAG_MAX_CHUNK_LENGTH

# ============================================================================
# LLM CONFIGURATIONS
# ============================================================================

LLM_PROMPT_TEMPLATE = PromptTemplate(
    template="""You are a helpful research assistant. Answer the user's question using the provided document context.

ANSWER GUIDELINES:
1. Answer using information from the provided context. Prioritize it over general knowledge.
2. Look carefully through ALL context segments — the answer may be implicit, partial, or spread across multiple sections.
3. If the context contains relevant information (even indirect or partial), use it to form your best answer.
4. Do NOT invent specific facts, numbers, names, or data that are not present in the context.
5. Cite source page numbers naturally when referencing specific information.
6. Be concise but thorough. Answer in the same language as the question (Vietnamese, English, etc.).
7. Only say you cannot find the answer if the context has NO information related to the topic at all.

IMPORTANT — End your response with exactly ONE of these tags (no text after it):
- [FOUND_ANSWER: true]  — the context contained useful information to answer the question
- [FOUND_ANSWER: false] — the context had absolutely no information related to this topic

CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION: {question}

YOUR ANSWER:""",
    input_variables=["context", "question"],
)
LLM_MODEL_NAME = "qwen2.5:7b"
LLM_BASE_URL = "http://localhost:11434"

# Low temperature for factual, grounded answers, higher may be more creative but less accurate
LLM_TEMPERATURE = 0.7

# Context window size
LLM_NUM_CTX = 4096

# Set to True to enable detailed debug prints during RAG chain creation and query processing
PRINT_DEBUG = True
