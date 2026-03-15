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
RAG_RETRIEVAL_K = 5

# Minimum results to guarantee even if quality threshold filters out too much
RAG_RETRIEVAL_MIN_RESULTS = 1

# Similarity score threshold for quality filtering (FAISS distance: lower = better match)
# Typical range: 5.0-15.0. Lower = stricter filtering (fewer chunks but higher quality)
RAG_RETRIEVAL_SCORE_THRESHOLD = 15.0

# Max total context length in characters to send to LLM (prevents CUDA OOM)
RAG_MAX_CONTEXT_LENGTH = 5000

# Max length per chunk in characters (prevents extremely long single chunks)
RAG_MAX_CHUNK_LENGTH = 1000

# 20% of chunk length overlap between chunks to preserve context
RAG_CHUNK_OVERLAP = round(.2 * RAG_MAX_CHUNK_LENGTH)

# ============================================================================
# LLM CONFIGURATIONS
# ============================================================================

LLM_PROMPT_TEMPLATE = PromptTemplate(
    template="""You are a helpful research assistant. Your role is to answer questions based ONLY on the provided document content.

CRITICAL CONSTRAINTS:
1. You MUST answer using ONLY the information from the provided context.
2. Do NOT use your pre-trained knowledge. Do NOT hallucinate or invent information.
3. If the answer is not found in the context, explicitly state: "The provided sources do not contain information about this."
4. When citing information, naturally reference the source page number in your response.
5. Be concise but comprehensive in your answers.
6. Answer in the same language as the user's question (detect Vietnamese, English, etc.).

IMPORTANT - At the end of your response, ALWAYS include ONE of these tags:
- [FOUND_ANSWER: true] — if you found useful information in the context to answer the question
- [FOUND_ANSWER: false] — if you had to rely on general knowledge or couldn't find relevant info in the context

Do NOT put anything after the [FOUND_ANSWER: true/false] tag.

CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION: {question}

YOUR ANSWER (grounded only in the provided context):""",
    input_variables=["context", "question"],
)
LLM_MODEL_NAME = "qwen2.5:7b"
LLM_BASE_URL = "http://localhost:11434"
LLM_TEMPERATURE = 0.7  # Slightly creative but grounded
LLM_NUM_CTX = 2048  # Context window size — lower = less VRAM (default 32768 causes OOM)

PRINT_DEBUG = True  # Set to True to enable detailed debug prints during RAG chain creation and query processing
