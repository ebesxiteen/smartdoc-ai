from typing import Dict, List, Optional
from datetime import datetime, timezone

APP_NAME: str = "SmartDoc AI"
APP_FULLNAME: str = "SmartDoc - Your AI Research Assistant"

DB_NAME: str = "smartdoc"
DB_ROOT_PATH: str = f"./data/{DB_NAME}.db"

USER_ROLE_NAME: str = "user"
ASSISTANT_ROLE_NAME: str = "assistant"
NOTEBOOK_DEFAULT_NAME: str = "Untitled Notebook"
SOURCE_DEFAULT_NAME: str = "Untitled Document"
NOTE_DEFAULT_TITLE: str = "Untitled Note"

MAX_NOTEBOOK_NAME_LEN: int = 50
MAX_DESCRIPTION_LEN: int = 200
MAX_FILENAME_LEN: int = 100
MAX_NOTE_TITLE_LEN: int = 80

EMBEDDING_MODEL_NAME: str = (
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# Maximum number of past messages (user + assistant) to keep in conversation history for context (higher = more context but may cause CUDA OOM)
MAX_MSG_HISTORY: int = 10

# Number of chunks to use for summary generation
TOP_K_CHUNKS_FOR_SUMMARY: int = 3

# Number of chunks to use for suggested questions generation
TOP_K_CHUNKS_FOR_QUESTIONS: int = 3

# Use this answer as a fallback when LLM generate an empty response after tag removal (in random rare cases), also to prevent DB insertion error and ensure user gets a response even if LLM fails to generate text
NOT_FOUND_ANSWER_FALL_BACK: str = "Sorry, I couldn't find any relevant information in the documents to answer your question. Please try rephrasing or ask about a different topic."

# For hybrid search: weight for combining semantic similarity and BM25 keyword matching scores (range 0.0 to 1.0 as total, where 0.5 means equal weighting)
WEIGHT_SEMANTIC: float = 0.5
WEIGHT_BM25: float = 0.5

# The "magic number" for RRF (Reciprocal Rank Fusion) re-ranking algorithm, which helps to balance the influence of rank positions in the final combined score.
# Higher values makes the ranking less "aggressive" and more tolerant to lower-ranked results, while lower values makes it more "aggressive" and focused on top-ranked results.
RRF_C: int = 60

# ============================================================================
# MULTI-LANGUAGE GREETING PATTERNS
# ============================================================================
# Language-specific greeting patterns for automatic language detection
# Supports: English (en), Vietnamese (vi), Mandarin Chinese (zh)

# Fallback patterns (English only) - used when language detection fails
# This is a safety net to ensure greetings are still detected even without langdetect
DEFAULT_EN_GREETING_PATTERNS: list[str] = [
    # ==== GREETINGS ====
    r"^(hi|hello|hey|greetings|welcome)",
    r"^(good morning|good afternoon|good evening|good day)",
    # ==== INTRODUCTIONS & IDENTITY ====
    r"^who are you",
    r"^what.*your name",
    r"^what are you",
    r"^are you.*ai|assistant|bot",
    r"^can you introduce yourself",
    r"^tell me.*about yourself",
    r"^introduce yourself",
    r"^do you know my",
    r"^do you remember",
    r"^can you remember",
    # ==== CAPABILITIES & PURPOSE ====
    r"^what can you do",
    r"^what.*your purpose",
    r"^what.*your capabilities",
    r"^how can you help",
    r"^can you help.*with",
    r"^are you able to",
    r"^do you know about",  # General knowledge indicator
    # ==== TIME & DATE AWARENESS ====
    r"^what (is|'s) the (current )?(time|date|day)",
    r"^what time is it",
    r"^today'?s date",
    r"^what day is it",
    # ==== CONVERSATION & WELL-BEING ====
    r"^how are you",
    r"^how are things",
    r"^how's it going",
    r"^how is everything",
    r"^nice to meet you",
    # ==== CLOSINGS & GRATITUDE ====
    r"^(bye|goodbye|see you|take care|farewell|see ya)",
    r"^(thank you|thank|thanks|thanx|thx)",
    r"^(ok|okay|alright|got it|understood|copy)",
    r"^(perfect|great|cool|awesome|excellent|wonderful)",
    r"^have a (great|good|nice|wonderful)",
    r"^that ('s|is|was) (great|helpful|perfect|awesome)",
    # ==== USAGE & ONBOARDING ====
    r"^how (do|should) i (use|interact with|work with|get started)",
    r"^how do i get started",
    r"^what should i ask",
    r"^what can i ask you",
    r"^can i ask you about",
    r"^how does (this work|it work)",
    # ==== GENERAL KNOWLEDGE REQUESTS (Non-Document) ====
    r"^tell me about (?!.*this document)",  # "tell me about X" (NOT in documents)
    r"^explain (what |how |why )?(?!.*from)",  # "explain X" (standalone)
    r"^what('s| is) (the )?definition of",
    r"^what is the meaning of",
    r"^(define|definition of)",
    # ==== SMALL TALK ====
    r"^(lol|haha|ha|hehe)",
    r"^(yep|yup|yeah|nope|nah)",
    # ==== POLITENESS WITH GENERAL TOPIC ====
    r"^please tell me about(?!.*in|from)",
    r"^could you explain(?!.*from|in)",
]

GREETING_PATTERNS_BY_LANGUAGE: Dict[str, List[str]] = {
    "en": DEFAULT_EN_GREETING_PATTERNS,
    "vi": [
        # ==== GREETINGS (VIETNAMESE) ====
        r"^(xin chào|chào|chào bạn)",
        r"^(chào buổi sáng|chào buổi chiều|chào buổi tối)",
        # ==== INTRODUCTIONS & IDENTITY ====
        r"^(bạn là ai|bạn tên gì|tên của bạn)",
        r"^(bạn có phải|giới thiệu|hãy giới thiệu)",
        r"^(bạn có biết tên|bạn còn nhớ)",
        # ==== CAPABILITIES & PURPOSE ====
        r"^(bạn có thể|bạn sẽ làm gì|mục đích của bạn)",
        r"^(bạn hỗ trợ|bạn giúp tôi)",
        # ==== TIME & DATE AWARENESS ====
        r"^(mấy giờ rồi|hôm nay mấy giờ|ngày mấy)",
        r"^(hôm nay|ngày hôm nay)",
        # ==== CONVERSATION & WELL-BEING ====
        r"^(bạn khỏe|bạn thế nào|mọi thứ thế nào)",
        r"^(vui được gặp|rất vui gặp)",
        # ==== CLOSINGS & GRATITUDE ====
        r"^(tạm biệt|tạm biệt bạn|hẹn gặp lại)",
        r"^(cảm ơn|cám ơn|cảm ơn bạn)",
        r"^(được|ok|oké|hiểu rồi)",
        r"^(tuyệt vời|rất tốt|tốt lắm)",
        # ==== USAGE & ONBOARDING ====
        r"^(làm sao để|cách sử dụng|hướng dẫn)",
        r"^(tôi nên hỏi gì|tôi có thể hỏi)",
        r"^(cách nó hoạt động|nó hoạt động)",
        # ==== GENERAL KNOWLEDGE ====
        r"^(kể cho tôi|hãy kể|nói cho tôi)",
        r"^(giải thích|hãy giải thích)",
        r"^(định nghĩa|ý nghĩa|khái niệm)",
        # ==== SMALL TALK ====
        r"^(haha|hihi|hehe|lol)",
        r"^(được|ok|ừ|không|đúng)",
    ],
    "zh": [
        # ==== GREETINGS (MANDARIN CHINESE) ====
        r"^(你好|你好吗|你好啊|嗨|喂|早上好|下午好|晚上好)",
        r"^(最近怎么样|你好|您好|大家好)",
        # ==== INTRODUCTIONS & IDENTITY ====
        r"^(你是谁|你叫什么|你的名字|你是)",
        r"^(自我介绍|请介绍)",
        # ==== CAPABILITIES & PURPOSE ====
        r"^(你能做什么|你可以做|你的目的|你的作用)",
        r"^(你能帮|你可以帮|你支持)",
        # ==== TIME & DATE AWARENESS ====
        r"^(现在几点|几点|今天几|当前时间)",
        r"^(现在时间|表示时刻)",
        # ==== CONVERSATION & WELL-BEING ====
        r"^(你怎么样|最近怎么样)",
        r"^(很高兴认识|认识你很|很高兴|见面|见到)",
        # ==== CLOSINGS & GRATITUDE ====
        r"^(再见|拜拜|谢谢|感谢|谢了)",
        r"^(好的|好|好吧|没问题|明白)",
        r"^(很好|很棒|不错|优秀|完美)",
        # ==== USAGE & ONBOARDING ====
        r"^(怎样使用|如何使用|怎么用|使用方法)",
        r"^(我应该问|我能问|我该怎样)",
        r"^(怎么工作|它如何|如何工作)",
        # ==== GENERAL KNOWLEDGE ====
        r"^(告诉我|请告诉|请说)",
        r"^(解释|请解释|说明)",
        r"^(定义|含义|意思|概念|是什么)",
        # ==== SMALL TALK ====
        r"^(哈哈|嘿|呃|嗯|对|不对|是的)",
        r"^(好|不好|行|可以)",
    ],
}

# Prompt template for rephrasing follow-up questions into standalone questions for better retrieval. This is crucial for handling conversational queries that rely on previous context.
REPHRASE_PROMPT: str = """You are a question reformulation specialist for a multi-lingual document retrieval system.

TASK: Analyze the chat history and transform the latest user question into a standalone question optimized for vector database search.

RULES:
1. **Preserve Intent & Language**: Keep all key terms. **CRITICAL**: DO NOT TRANSLATE. The reformulated question MUST be in the exact same language as the ORIGINAL latest user question. If the user asks in English, the output MUST be in English.
2. **Replace References**: Replace pronouns ("it", "they", "this", "đó") and ambiguous references with explicit terms from the chat history.
3. **Handle Standalone Queries**: If the latest question makes perfect sense on its own and does not rely on previous context, return it EXACTLY as it is. Do not add unnecessary history.
4. **Keep Concise**: Output ONLY the rephrased question. No introductory phrases, no explanations, no formatting tags.
5. **Domain-Specific**: Use specific terminology mentioned previously rather than vague descriptions.

EXAMPLES:
- History: User: What is photosynthesis? -> AI: [explanation]
  Original: How does it work?
  Output: How does the photosynthesis process work?

- History: User: AI có thể dùng Neural Networks hoặc Decision Trees. -> AI: [explanation]
  Original: Vậy phương pháp thứ hai có ưu điểm gì?
  Output: Decision Trees có ưu điểm gì?

- History: User: Hello, my name is John -> AI: Nice to meet you John.
  Original: Do you know my name?
  Output: Does the AI know the user's name is John?

- History: User: Xin chào -> AI: Chào bạn, tôi có thể giúp gì?
  Original: RAG là gì?
  Output: RAG là gì?"""

# ============================================================================
# DOCUMENT PROCESSING PROMPTS
# ============================================================================

SUMMARY_PROMPT: str = """You are a highly capable document analyst.

TASK: Analyze the provided document text and write a brief 2-3 sentence overview of its main topics.

RULES:
1. **Be Concise**: Limit your answer to exactly 2-3 sentences.
2. **Language Matching**: You MUST write the summary in the EXACT SAME LANGUAGE as the provided text (e.g., if the text is in Vietnamese, write in Vietnamese).
3. **No Fluff**: DO NOT include introductory conversational phrases (e.g., "Here is a summary", "This document is about"). Just output the summary directly.

<document_text>
{text}
</document_text>

SUMMARY:"""

SUGGESTED_QUESTIONS_PROMPT: str = """You are an expert educational assistant building a study guide.

TASK: Analyze the provided document text and generate exactly 3 specific, interesting, and diverse questions that a reader might ask to test their knowledge of this text.

RULES:
1. **Format**: Format EXACTLY as a numbered list (1., 2., 3.). Do NOT use bolding or markdown for the numbers.
2. **Accuracy**: Questions MUST be directly answerable using only the provided text.
3. **Language Matching**: You MUST write the questions in the EXACT SAME LANGUAGE as the provided text.
4. **No Fluff**: DO NOT include conversational filler, introductory text, or concluding remarks. Output ONLY the 3 numbered questions.

<document_text>
{text}
</document_text>

QUESTIONS:"""

# ============================================================================
# RAG RETRIEVAL CONFIGURATIONS - Optimize these for your use case
# ============================================================================
# Maximum number of chunks to retrieve (higher = more context but may cause CUDA OOM)
RAG_RETRIEVAL_K: int = 8

# Minimum results to guarantee even if quality threshold filters out too much
RAG_RETRIEVAL_MIN_RESULTS: int = 1

# Euclidean Distance (L2): similarity score threshold for quality filtering (FAISS distance: lower = better match)
# 5.0-10.0 = Strict filtering (only highest quality matches)
# 10.0-15.0 = Moderate filtering (balanced quality/quantity)
# 15.0-Infinity = Loose filtering (include more results even if less relevant)
RAG_RETRIEVAL_SCORE_THRESHOLD: float = 15.0

# Max length per chunk in characters (prevents extremely long single chunks)
RAG_MAX_CHUNK_LEN: int = 1000

# Typically 20% of chunk length overlap between chunks to preserve context
RAG_CHUNK_OVERLAP: int = round(0.2 * RAG_MAX_CHUNK_LEN)

# Max total context length in characters to send to LLM (prevents CUDA OOM)
RAG_MAX_CTX_LEN: int = RAG_RETRIEVAL_K * RAG_MAX_CHUNK_LEN

# ============================================================================
# LLM CONFIGURATIONS
# ============================================================================


def get_sys_prompt(custom_instructions: Optional[str] = None) -> str:
    """
    Get enhanced system prompt with current time, general knowledge permission,
    and strict XML boundaries for RAG processing.
    """
    # Using a slightly cleaner time format for readability
    current_time = datetime.now(timezone.utc).strftime("%A, %B %d, %Y at %H:%M UTC")

    base_role = (
        f"You are {APP_NAME}, a highly capable and professional research assistant."
    )
    override_block = (
        f"\nADDITIONAL USER INSTRUCTIONS:\n{custom_instructions}\n"
        if custom_instructions
        else ""
    )

    enhanced_prompt = f"""{base_role}
Current Time: {current_time}
{override_block}
ROLE & BEHAVIOR:
You answer user questions based primarily on the provided `<context>`. If the user asks general questions, greetings, or about your capabilities, you may use your internal knowledge.

STRICT GUIDELINES:
1. **Document-Based Q&A**:
   - If the question asks for facts, summaries, or specific data, search the `<context>` thoroughly.
   - Do NOT hallucinate or invent facts, numbers, or names not present in the context.
   - Cite source page numbers naturally if they are available in the context.

2. **General Knowledge & Greetings**:
   - If the user says "Hello," asks for the time, or asks a broad question clearly outside the scope of the documents, answer naturally using your general knowledge.

3. **Language & Formatting**:
   - Respond in the EXACT language of the user's question.
   - **CRITICAL**: Do NOT translate technical keywords (e.g., "RAG", "LLM", "API", "Database"). Keep them in their original form.
   - Use Markdown (bullet points, bold text) to structure your answer cleanly.

TAGGING REQUIREMENT (CRITICAL):
You MUST end your final response with exactly ONE of the following tags on a new line. Do not write anything after the tag:
- [STATUS: DOC_ANSWER] — You successfully answered the question using information found in the context.
- [STATUS: DOC_MISSING] — The user asked about a specific topic, but the context lacked the information. (You must politely explain that the requested information is not available in the documents).
- [STATUS: GENERAL] — The user asked a greeting or general question, and you answered from your internal knowledge.

<context>
{{context}}
</context>

<user_question>
{{question}}
</user_question>

YOUR ANSWER:"""

    return enhanced_prompt


LLM_MODEL_NAME: str = "qwen2.5:7b"
OLLAMA_BASE_URL: str = "http://localhost:11434"

# Low temperature for factual, grounded answers, higher may be more creative but less accurate
LLM_TEMPERATURE: float = 0.7

# Context window size
LLM_NUM_CTX: int = 4096

# Set to True to enable detailed debug prints during RAG chain creation and query processing
PRINT_DEBUG: bool = True

# ============================================================================
# USER SETTINGS CONFIGURATIONS
# ============================================================================

# Retrieval k
RAG_RETRIEVAL_K_MIN: int = 1
RAG_RETRIEVAL_K_MAX: int = 20
RAG_RETRIEVAL_K_STEP: int = 1
RAG_RETRIEVAL_K_HELP_MSG: str = "Number of document chunks to retrieve for context. Higher values provide more information but may cause slower responses or GPU memory issues."

# Retrieval minimum results
RAG_RETRIEVAL_MIN_RESULTS_MIN = 0
RAG_RETRIEVAL_MIN_RESULTS_MAX = 10
RAG_RETRIEVAL_MIN_RESULTS_STEP = 1
RAG_RETRIEVAL_MIN_RESULTS_HELP_MSG = "Minimum number of retrieved chunks to use as context, even if they don't meet the quality threshold. This ensures the AI always has some information to work with, but may reduce answer quality if set too high."

# Retrieval score threshold
RAG_RETRIEVAL_SCORE_THRESHOLD_MIN: float = 0.0
RAG_RETRIEVAL_SCORE_THRESHOLD_MAX: float = 30.0
RAG_RETRIEVAL_SCORE_THRESHOLD_STEP: float = 0.5
RAG_RETRIEVAL_SCORE_THRESHOLD_HELP_MSG: str = "Similarity score threshold for filtering retrieved chunks (lower = stricter). Adjust based on your documents and embedding model for best results."

# Max chunk length
RAG_MAX_CHUNK_LEN_MIN: int = 500
RAG_MAX_CHUNK_LEN_MAX: int = 5000
RAG_MAX_CHUNK_LEN_STEP: int = 100
RAG_MAX_CHUNK_LEN_HELP_MSG: str = "Maximum length of each document chunk in characters. Longer chunks provide more context but may reduce retrieval precision. Adjust based on your document structure."

# Chunk overlap
RAG_CHUNK_OVERLAP_MIN: int = 0
RAG_CHUNK_OVERLAP_MAX: int = 1000
RAG_CHUNK_OVERLAP_STEP: int = 50
RAG_CHUNK_OVERLAP_HELP_MSG: str = "Number of overlapping characters between consecutive chunks. Overlap helps preserve context across chunks but increases total tokens sent to the LLM. Typically set to around 20% of max chunk length."

# Max context length
RAG_MAX_CTX_LEN_MIN: int = 1000
RAG_MAX_CTX_LEN_MAX: int = 50000
RAG_MAX_CTX_LEN_STEP: int = 1000
RAG_MAX_CTX_LEN_HELP_MSG: str = "Maximum total length of all retrieved chunks combined to send to the LLM. Higher values allow more context but may cause slower responses or GPU memory issues. Adjust based on your documents and LLM capabilities."

# Max message history
MAX_MSG_HISTORY_MIN: int = 0
MAX_MSG_HISTORY_MAX: int = 50
MAX_MSG_HISTORY_STEP: int = 1
MAX_MSG_HISTORY_HELP_MSG: str = "Maximum number of past messages (user + assistant) to keep in conversation history for context. Higher values provide more context but may cause slower responses or GPU memory issues."

# LLM Model Name
LLM_MODEL_NAME_HELP_MSG: str = "The 🧠 of your assistant. Larger models (e.g., 14b+) are smarter but require more VRAM. Smaller models (7b) are faster and require less VRAM."

# LLM Number of Context Tokens
LLM_NUM_CTX_MIN: int = 1024
LLM_NUM_CTX_MAX: int = 32000
LLM_NUM_CTX_STEP: int = 512
LLM_NUM_CTX_HELP_MSG: str = "The AI's total short-term memory (1k tokens ≈ 750 words). Higher values allow more sources and longer history but increase GPU RAM usage."

# LLM Temperature
LLM_TEMPERATURE_MIN: float = 0.0
LLM_TEMPERATURE_MAX: float = 1.0
LLM_TEMPERATURE_STEP: float = 0.05
LLM_TEMPERATURE_HELP_MSG: str = "Controls creativity. Range 0.1 - 0.3 is best for factual research and citations. 0.7-0.9 is better for brainstorming and creative summaries."

# System Prompt Override
PERSONAL_CTX_HELP_MSG: str = "Custom instructions or your personal background to guide the AI's behavior and personality."
PERSONAL_CTX_PLACEHOLDER: str = "E.g., I am a high school biology teacher..."

WEIGHT_SEMANTIC_MIN: float = 0.0
WEIGHT_SEMANTIC_MAX: float = 1.0
WEIGHT_SEMANTIC_STEP: float = 0.05
WEIGHT_SEMANTIC_HELP_MSG: str = "Balance between Semantic Vector Search (set value) and Keyword BM25 Search (remaining value). BM25 is better for exact technical terms while Semantic is better for concepts."

# ============================================================================
# DEBUG LOGGING CONFIGURATIONS
# ============================================================================
# Structured logging emoji categories - Each emoji uniquely identifies a log type
LOG_CATEGORIES = {
    # Task Initialization & Start
    "TASK_START": {
        "emoji": "🚀",
        "type": "INFO",
        "description": "Starting a major task or operation",
    },
    "LOAD_START": {
        "emoji": "📂",
        "type": "INFO",
        "description": "Starting to load/read files",
    },
    "PROCESS_START": {
        "emoji": "⚙️",
        "type": "INFO",
        "description": "Starting data processing step",
    },
    # Operations & Completion
    "SUCCESS": {
        "emoji": "✅",
        "type": "SUCCESS",
        "description": "Operation completed successfully",
    },
    "COMPLETE": {
        "emoji": "✔️",
        "type": "SUCCESS",
        "description": "Task fully completed",
    },
    "MERGED": {
        "emoji": "🔀",
        "type": "SUCCESS",
        "description": "Data merger/combination operation",
    },
    "SAVED": {
        "emoji": "💾",
        "type": "SUCCESS",
        "description": "Data saved to storage/database",
    },
    # Data & Search Operations
    "QUERY": {"emoji": "🔍", "type": "INFO", "description": "Query/search operation"},
    "RETRIEVE": {
        "emoji": "📦",
        "type": "INFO",
        "description": "Retrieving data from storage",
    },
    "CHUNK": {"emoji": "📄", "type": "INFO", "description": "Chunking/splitting data"},
    "EMBED": {
        "emoji": "🧬",
        "type": "INFO",
        "description": "Embedding/vectorization operation",
    },
    "SEARCH": {
        "emoji": "🔎",
        "type": "INFO",
        "description": "Performing search in vectorstore",
    },
    # AI/LLM Operations
    "LLM_INIT": {
        "emoji": "🤖",
        "type": "INFO",
        "description": "Initializing or calling LLM",
    },
    "CHAIN": {
        "emoji": "⛓️",
        "type": "INFO",
        "description": "Building or invoking RAG chain",
    },
    "HISTORY": {
        "emoji": "📜",
        "type": "INFO",
        "description": "Processing chat history",
    },
    "RESPONSE": {"emoji": "💬", "type": "INFO", "description": "LLM response received"},
    # Retrieval Strategy
    "RETRIEVER_INIT": {
        "emoji": "🎯",
        "type": "INFO",
        "description": "Initializing retriever strategy",
    },
    "SEMANTIC": {
        "emoji": "🧠",
        "type": "INFO",
        "description": "Semantic/vector retrieval mode",
    },
    "KEYWORD": {
        "emoji": "🔑",
        "type": "INFO",
        "description": "Keyword/BM25 retrieval mode",
    },
    "HYBRID": {
        "emoji": "🔀",
        "type": "INFO",
        "description": "Hybrid retrieval mode (semantic + keyword)",
    },
    "FALLBACK": {
        "emoji": "🔄",
        "type": "INFO",
        "description": "Fallback retrieval strategy activated",
    },
    # Warnings & Cautions
    "WARN": {
        "emoji": "⚠️",
        "type": "WARNING",
        "description": "Warning - potential issue",
    },
    "SKIP": {"emoji": "⊘", "type": "WARNING", "description": "Skipping operation/step"},
    "THRESHOLD": {
        "emoji": "📊",
        "type": "WARNING",
        "description": "Threshold-related alert",
    },
    "FALLBACK_ACTIVE": {
        "emoji": "🛡️",
        "type": "WARNING",
        "description": "Fallback mechanism activated",
    },
    # Errors & Failures
    "ERROR": {"emoji": "❌", "type": "ERROR", "description": "Error occurred"},
    "FAIL": {"emoji": "💥", "type": "ERROR", "description": "Operation failed"},
    "MISSING": {
        "emoji": "🚫",
        "type": "ERROR",
        "description": "Missing required data/resource",
    },
    # Debug & Inspection
    "DEBUG": {"emoji": "🔬", "type": "INFO", "description": "Debug information"},
    "STATS": {"emoji": "📈", "type": "INFO", "description": "Statistics/metrics"},
    "CACHE": {"emoji": "💾", "type": "INFO", "description": "Cache operation"},
    # Misc/Utility
    "SECTION": {"emoji": "📍", "type": "INFO", "description": "Section marker"},
    "ITEM": {"emoji": "•", "type": "INFO", "description": "List item"},
}
