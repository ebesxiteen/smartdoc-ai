from typing import Dict, List
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

MAX_NOTEBOOK_NAME_LEN = 50
MAX_DESCRIPTION_LEN = 200
MAX_FILENAME_LEN = 100
MAX_NOTE_TITLE_LEN = 80

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Maximum number of past messages (user + assistant) to keep in conversation history for context (higher = more context but may cause CUDA OOM)
MAX_MSG_HISTORY = 10

# ============================================================================
# MULTI-LANGUAGE GREETING PATTERNS
# ============================================================================
# Language-specific greeting patterns for automatic language detection
# Supports: English (en), Vietnamese (vi), Mandarin Chinese (zh)

# Fallback patterns (English only) - used when language detection fails
# This is a safety net to ensure greetings are still detected even without langdetect
DEFAULT_EN_GREETING_PATTERNS = [
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
REPHRASE_PROMPT = """You are a question reformulation specialist for document retrieval systems.

TASK: Transform a follow-up user question into a standalone question that can be
searched in a document vector database.

RULES:
1. **Preserve Intent**: Keep all key terms, numbers, and constraints from the original question
2. **Replace References**: Replace pronouns and ambiguous references with explicit terms
   from previous exchanges
   - "it" → actual topic name
   - "that approach" → specific approach mentioned
   - "the other one" → actual item name
3. **Add Context**: Include relevant context from chat history that clarifies the question
4. **Keep Concise**: Use 1-2 sentences maximum, optimized for vector database search
5. **Domain-Specific**: Use terminology that would appear in documents (e.g., "machine learning"
   instead of "that thing you mentioned")
6. **No Answering**: Only rephrase the question; do NOT provide answers or explanations

EXAMPLES:
- Original: "How does it work?"
  History: Previous question was about photosynthesis
  Output: "How does the photosynthesis process work?"

- Original: "What about the other approach?"
  History: Discussed neural networks and decision trees
  Output: "What are the advantages and disadvantages of the decision tree approach compared to neural networks?"

Format your response as just the rephrased question, no other text."""

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
RAG_CHUNK_OVERLAP = round(0.2 * RAG_MAX_CHUNK_LENGTH)

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
6. **Language Detection & Accuracy**:
   - Detect the language of the USER QUESTION and respond in that same language.
   - If the question is in Vietnamese, respond in natural, professional Vietnamese.
   - **CRITICAL**: Do NOT translate technical keywords, proper names, or industry-standard terms (e.g., "RAG", "LLM", "FAISS", "Machine Learning", "Database", "API"). Keep them in their original form.
7. Only say you cannot find the answer if the context has NO information related to the topic at all.
8. **Tone**: Be concise but thorough. Ensure the sentence structure is natural to the detected language.

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
