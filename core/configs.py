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

CROSS_ENCODER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

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
    r"^(what'?s up|wassup|wass up)",  # Informal greetings
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
    r"^explain (what |how |why )",  # "explain what/how/why X" (must have keyword)
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

TASK: Analyze the provided document text and write a concise 3-5 sentence overview of its main topics.

RULES:
1. **Be Concise**: Limit your answer to 3-5 sentences. Longer documents with multiple major themes warrant more sentences.
2. **Language Matching**: You MUST write the summary in the EXACT SAME LANGUAGE as the provided text (e.g., if the text is in Vietnamese, write in Vietnamese).
3. **No Fluff**: DO NOT include introductory conversational phrases (e.g., "Here is a summary", "This document is about"). Just output the summary directly.
4. **Key Terms**: Include the document's key technical, domain-specific, or subject-matter terms so the reader immediately understands the field.

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
5. **Question Variety**: Vary the question types — include at least one factual recall question (who/what/when/where), one conceptual understanding question (why/how does it work), and one applied or analytical question (compare, evaluate, or apply a concept).

<document_text>
{text}
</document_text>

QUESTIONS:"""

# ============================================================================
# SELF-RAG PROMPTS - Advanced multi-hop retrieval with quality scoring
# ============================================================================

# Layer 2 LLM-based greeting validation (fallback if regex misses greetings)
LAYER2_LLM_ROUTER_PROMPT: str = """You are an intent classifier for a document retrieval system.

Given a user query, determine if it's a greeting/chitchat or a factual question requiring document search.

QUERY: {query}

Respond with ONLY one word: "GREETING" or "FACTUAL"

GREETING examples (return GREETING):
- "Hello!", "Hi there!", "Hey!", "Hiya!", "Greetings"
- "How are you?", "How are you doing?", "What's up?"
- "What's your name?", "Who are you?", "Are you an AI?"
- "Good morning!", "Good evening!", "Chào bạn!", "你好!"
- "Thanks!", "Thank you!", "Goodbye!", "See you!"
- Mixed: "Hi! What's your name and can you help me?" → GREETING (greeting intent dominates)

FACTUAL examples (return FACTUAL):
- "What is machine learning?"
- "How does photosynthesis work?"
- "Explain the main conclusions of this paper."
- "What were the results in section 3?"
- Any question requiring specific factual information from documents

Response:"""

# Search plan generation: Break down complex query into 1-3 independent sub-queries
SEARCH_PLANNER_PROMPT: str = """You are a search strategy expert for document retrieval.

Analyze the user's original query and break it into 1-3 independent sub-queries that together cover all aspects of the original question.

ORIGINAL QUERY: {original_query}

Rules:
1. Each sub-query must be independent and retrievable via vector search.
2. Sub-queries should cover different angles/aspects of the original query if complex.
3. Avoid redundancy — don't repeat the same query or surface the same aspect twice.
4. Keep each sub-query under 15 words.
5. **Language**: Generate sub-queries in the EXACT same language as the original query. Do NOT translate. If the query is in Vietnamese, output Vietnamese sub-queries. If in English, output English sub-queries.
6. If the original query references previous context (e.g., "What about it?", "Tell me more"), incorporate that context explicitly into the sub-queries so each one is self-contained.
7. Output ONLY the sub-queries, one per line, no numbering, no bullet points, no explanations.

SUB-QUERIES:"""

# Repair agent: Diagnose failed answers and generate new strategy
REPAIR_AGENT_PROMPT: str = """You are a diagnostic repair agent for a multi-hop retrieval system.

An AI answer failed quality checks. Analyze the failure and suggest a DIFFERENT search strategy to find better information.

ORIGINAL QUERY: {original_query}
FAILED ANSWER: {failed_answer}
FAILURE REASON: {failure_reason}
PREVIOUS SEARCH ATTEMPTS: {search_history}

Diagnosis and strategy:
- If ISSUP failed (low groundedness): Search for more specific source passages; try quoting exact terms from the domain.
- If ISREL failed (low relevance): Broaden the search angle; try synonyms or related concepts.
- If ISUSE failed (low utility): Reframe the sub-query to better match the user's intent.
- Always avoid repeating keywords or angles from PREVIOUS SEARCH ATTEMPTS.

Generate 1-3 NEW search sub-queries using a different angle or related concepts.

**Language**: Generate sub-queries in the EXACT same language as the original query.
**Format**: Output ONLY the new sub-queries, one per line, no numbering, no bullet points, no explanations.

NEW SEARCH STRATEGY:"""

# Sub-query rewrite prompt: LLM-based rewrite for failed sub-queries during horizontal retry
SUBQUERY_REWRITE_PROMPT: str = """You are a search query optimization specialist.
A sub-query returned zero results from the vector database. Rewrite it to improve retrieval.

ORIGINAL QUERY: {original_query}
FAILED SUB-QUERY: {failed_subquery}
SUCCESSFUL CONTEXT (from other sub-queries, if any): {success_context}

Rules:
1. Use synonyms or alternative phrasing for the failed sub-query
2. Try a broader or narrower scope based on the context
3. Avoid repeating the exact same keywords that already failed
4. Output ONLY the new sub-query, one line, no explanations

NEW SUB-QUERY:"""

# Quality judge: Score answer on Groundedness (ISSUP) and Utility (ISUSE)
# Note: ISREL (relevance) is derived separately from cross-encoder scores stored in document metadata.
QUALITY_JUDGE_PROMPT: str = """You are a rigorous quality evaluator for AI-generated responses.

Score this answer on TWO dimensions using a strict JSON response format.

QUERY: {query}
ANSWER: {answer}
RETRIEVED CONTEXT: {context}

Respond with ONLY valid JSON (no markdown code blocks, no extra text, no trailing commas):
{{
    "issup": <float 0.0-1.0>,
    "isuse": <float 0.0-1.0>,
    "reasoning": "<one sentence explaining the score>"
}}

Scoring criteria:
- **ISSUP** (Groundedness, 0.0–1.0): How well is the answer supported by the RETRIEVED CONTEXT?
  - 1.0: Every claim in the answer is directly evidenced by the retrieved context.
  - 0.7: Most claims are supported; minor details may go slightly beyond the context.
  - 0.5: About half of the claims are supported; the rest are inferred or speculative.
  - 0.0: The answer contradicts or ignores the retrieved context entirely (hallucination).

- **ISUSE** (Utility / User Satisfaction, 0.0–1.0): Does the answer directly satisfy the user's intent?
  - 1.0: The answer fully resolves the query with the right scope, depth, and format.
  - 0.7: Mostly helpful but missing one minor aspect or slightly off in framing.
  - 0.5: Partially answers the query but is incomplete or tangential.
  - 0.0: The answer does not address the query at all.

IMPORTANT: If the retrieved context is empty or clearly unrelated, set ISSUP ≤ 0.3 to reflect lack of grounding.
"""

REFORMULATE_QUERY_PROMPT: str = """You are a query reformulation specialist for a multi-lingual document retrieval system.

TASK: Given the conversation history and a follow-up question, rephrase the follow-up question into a STANDALONE question optimized for vector database search. A standalone question can be understood without reading the chat history.

RULES:
1. **Preserve Language**: The reformulated question MUST be in the EXACT same language as the follow-up question. Do NOT translate under any circumstances.
2. **Replace References**: Substitute pronouns ("it", "they", "this", "đó", "chúng") and vague references with the explicit terms found in the chat history.
3. **Preserve Intent**: Keep all key terms and domain-specific vocabulary intact. Do not add, remove, or change the meaning of the question.
4. **Preserve Conversational Intent (CRITICAL)**: If the follow-up question is conversational, a greeting, or asking about the AI's state/knowledge (e.g., "How are you?", "Do you know my name?", "Who are you?"), return it EXACTLY as-is. DO NOT convert conversational queries into third-person factual questions (e.g., do not change "Do you know my name?" to "Do you know John?").
5. **Return as-is when appropriate**: If the follow-up question is already standalone (makes complete sense without context) or conversational, return it EXACTLY as provided — do not modify it.
6. **No meta-commentary**: Output ONLY the reformulated question. No explanations, no labels, no introductory phrases.

EXAMPLES:
- History: "User: What is the difference between RAG and fine-tuning?"
  Follow-up: "Which one is better for production?"
  Output: "Which is better for production use: RAG or fine-tuning?"

- History: "User: Hello, my name is Peter-> AI: Hello Peter."
  Follow-up: "Do you know my name?"
  Output: "Do you know my name?"

- History: "User: AI có thể dùng Neural Networks hoặc Decision Trees."
  Follow-up: "Vậy phương pháp thứ hai có ưu điểm gì?"
  Output: "Decision Trees có ưu điểm gì?"

- Follow-up: "What is machine learning?" (already standalone)
  Output: "What is machine learning?"

Chat History:
{chat_history}

Follow-up Question: {query}
Standalone Question:"""

# ============================================================================
# RAG RETRIEVAL CONFIGURATIONS - Optimize these for your use case
# ============================================================================
# Maximum number of chunks to retrieve (higher = more context but may cause CUDA OOM)
RAG_FINAL_CONTEXT_K: int = 8

# Number of chunks retrieved by initial broad search to send to cross-encoder
RAG_RERANK_TOP_N: int = 30

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
RAG_MAX_CTX_LEN: int = RAG_FINAL_CONTEXT_K * RAG_MAX_CHUNK_LEN

# ============================================================================
# SELF-RAG CONFIGURATIONS - Advanced multi-hop retrieval with quality scoring
# ============================================================================
# Maximum recursion depth for repair attempts (higher = more thorough but slower)
SELF_RAG_MAX_DEPTH: int = 2

# Number of parallel candidate answers to generate per hop (higher = more diversity but slower)
SELF_RAG_CANDIDATES: int = 3

# Maximum retry attempts for individual failing sub-queries during horizontal retrieval
SELF_RAG_MAX_RETRIES_PER_HOP: int = 2

# Quality threshold for Groundedness (ISSUP): How supported is the answer by retrieved documents?
SELF_RAG_THRESHOLD_ISSUP: float = 0.70

# Quality threshold for Relevance (ISREL): How relevant are retrieved chunks to the original query?
SELF_RAG_THRESHOLD_ISREL: float = 0.70

# Quality threshold for Utility (ISUSE): Does the answer meet the user's intent?
SELF_RAG_THRESHOLD_ISUSE: float = 0.70

# ============================================================================
# LLM CONFIGURATIONS
# ============================================================================


def generate_general_knowledge_fallback_prompt(
    user_query: str,
    chat_history: Optional[str] = None,
    query_complexity: str = "medium",
) -> str:
    """
    Generate a prompt for LLM to produce fallback answers when no relevant documents found.

    This function creates context-aware fallback answers instead of hard-coded messages,
    improving user experience by providing dynamic, thoughtful responses.

    Args:
        user_query: The original user question
        chat_history: Optional formatted conversation history for context
        query_complexity: 'simple', 'medium', or 'complex' to tailor response depth

    Returns:
        Prompt string for LLM to generate fallback answer
    """
    current_time = datetime.now(timezone.utc).strftime("%A, %B %d, %Y at %H:%M UTC")

    # Adjust response style based on query complexity
    complexity_guidance = {
        "simple": "Keep your answer concise and direct (2-3 sentences).",
        "medium": "Provide a balanced answer with key points (3-5 sentences).",
        "complex": "Give a comprehensive answer with multiple perspectives and details (5-7 sentences).",
    }
    depth_guidance = complexity_guidance.get(
        query_complexity, complexity_guidance["medium"]
    )

    # Build conversation context if available
    history_block = ""
    if chat_history:
        history_block = f"\nPREVIOUS CONVERSATION:\n{chat_history}\n"

    prompt = f"""You are {APP_NAME}, a helpful research assistant.

Current Time: {current_time}{history_block}

IMPORTANT: The user's question could not be answered from available documents.
Provide a helpful answer using your general knowledge.

{depth_guidance}

Use Markdown formatting for clarity. Be honest if the topic is outside your knowledge.

USER QUESTION: {user_query}

YOUR ANSWER:"""

    return prompt


def get_self_rag_system_prompt(
    personal_ctx: Optional[str] = None,
    current_time: Optional[str] = None,
) -> str:
    """
    Get system prompt tailored for Self-RAG Step 3 candidate generation.

    This prompt is used during Step 3 (candidate generation) of the Self-RAG pipeline.
    It is purely document-grounded and does NOT include general knowledge routing or
    greeting detection—Self-RAG handles those at earlier stages.

    The generator LLM is asked to include `[FOUND: YES]` or `[FOUND: NO]` at the very
    end of its answer. This tag is captured in generate_candidate_answers() to determine
    found_answer — whether citations should be shown to the user — independently of the
    quality gate scores (ISSUP/ISREL/ISUSE) that drive the repair logic.

    Args:
        personal_ctx: Personal context about the user (e.g., expertise level, role)
        current_time: Current datetime for temporal contextual answers

    Returns:
        System prompt string for use with ChatPromptTemplate.from_messages()
    """
    # Use provided time or default to current UTC time
    if current_time is None:
        current_time = datetime.now(timezone.utc).strftime("%A, %B %d, %Y at %H:%M UTC")

    # Build personal context section if available
    personal_context_block = ""
    if personal_ctx:
        personal_context_block = f"\n\nYOUR CONTEXT:\n{personal_ctx}\n"

    # System prompt for Self-RAG Step 3: Pure document-grounded generation
    # No general knowledge routing, no greeting detection—just high-quality answer generation
    prompt = f"""You are a knowledgeable expert answering a user's question directly and conversationally.

Current Time: {current_time}{personal_context_block}

YOUR TASK:
Answer the user's question based on the retrieved documents. Write as if you are a subject-matter expert explaining something to a curious person — not writing a formal report.

SHARED QUALITY STANDARDS (also enforced by Co-RAG pipeline):
{SHARED_RAG_STYLE_RULES}

ADDITIONAL RULES:
1. **Ground in Context First**: Base every claim on the retrieved documents. Inline citations like "(Page 46)" are fine where helpful.
2. **Acknowledge Gaps Honestly**: If the context doesn't fully cover the question, say so briefly and naturally (e.g., "The documents don't go into detail on X, but based on what's available...").
3. **Use Chat History When Relevant**: If the question references prior conversation, use the CHAT HISTORY to resolve the reference.
4. **Found Status**: At the very end of your answer, on its own line, include EXACTLY ONE of:
   - `[FOUND: YES]` — if your answer is primarily grounded in the retrieved documents
   - `[FOUND: NO]` — if the documents don't contain enough information and you relied on general knowledge

STRICT FORMAT RULES (violations will cause this answer to be discarded):
- Do NOT start with "Answer:", "**Answer:**", or any heading.
- Do NOT add a "Source Citations:", "References:", or "Additional Note:" section at the end.
- Do NOT add meta-commentary about the chat history (e.g., "The user mentioned their name is X").
- Write as flowing prose or a natural bulleted list — not a structured report.
- The `[FOUND: YES/NO]` tag must be the very last line of your response."""

    return prompt


LLM_MODEL_NAME: str = "qwen2.5:7b"
OLLAMA_BASE_URL: str = "http://localhost:11434"
LLM_BASE_URL: str = OLLAMA_BASE_URL  # Alias for Self-RAG

# Low temperature for factual, grounded answers, higher may be more creative but less accurate
LLM_AVG_TEMP: float = 0.7

# Context window size
LLM_NUM_CTX: int = 4096

# Set to True to enable detailed debug prints during RAG chain creation and query processing
PRINT_DEBUG: bool = True

# ============================================================================
# USER SETTINGS CONFIGURATIONS
# ============================================================================

# Retrieval k
RAG_FINAL_CONTEXT_K_MIN: int = 1
RAG_FINAL_CONTEXT_K_MAX: int = 20
RAG_FINAL_CONTEXT_K_STEP: int = 1
RAG_FINAL_CONTEXT_K_HELP_MSG: str = "Number of document chunks to retrieve for context. Higher values provide more information but may cause slower responses or GPU memory issues."


# Rerank Top N
RAG_RERANK_TOP_N_MIN: int = 5
RAG_RERANK_TOP_N_MAX: int = 100
RAG_RERANK_TOP_N_STEP: int = 5
RAG_RERANK_TOP_N_HELP_MSG: str = (
    "Number of chunks fetched by initial search to send to cross-encoder."
)

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
LLM_AVG_TEMP_MIN: float = 0.0
LLM_AVG_TEMP_MAX: float = 1.0
LLM_AVG_TEMP_STEP: float = 0.05
LLM_AVG_TEMP_HELP_MSG: str = "Controls creativity of AI responses. 0.0 = deterministic and focused on factual answers, 1.0 = more creative and diverse but may be less accurate. Adjust based on your use case and preference for creativity vs accuracy."

# System Prompt Override
PERSONAL_CTX_HELP_MSG: str = "Custom instructions or your personal background to guide the AI's behavior and personality."
PERSONAL_CTX_PLACEHOLDER: str = "E.g., I am a high school biology teacher..."

WEIGHT_SEMANTIC_MIN: float = 0.0
WEIGHT_SEMANTIC_MAX: float = 1.0
WEIGHT_SEMANTIC_STEP: float = 0.05
WEIGHT_SEMANTIC_HELP_MSG: str = "Balance between Semantic Vector Search (set value) and Keyword BM25 Search (remaining value). BM25 is better for exact technical terms while Semantic is better for concepts."

# ============================================================================
# SELF-RAG VALIDATION LIMITS & UI HELPERS
# ============================================================================

# Self-RAG max depth
SELF_RAG_MAX_DEPTH_MIN: int = 1
SELF_RAG_MAX_DEPTH_MAX: int = 5
SELF_RAG_MAX_DEPTH_STEP: int = 1
SELF_RAG_MAX_DEPTH_HELP_MSG: str = "Maximum number of recursive repair hops allowed for Self-RAG. Higher values provide more thorough answer refinement but increase latency. Recommended: 2-3."

# Self-RAG max candidates
SELF_RAG_CANDIDATES_MIN: int = 1
SELF_RAG_CANDIDATES_MAX: int = 5
SELF_RAG_CANDIDATES_STEP: int = 1
SELF_RAG_CANDIDATES_HELP_MSG: str = "Number of diverse candidate answers to generate per hop. Higher values increase answer quality diversity but add latency. Recommended: 2-3."

# Self-RAG max retries per hop
SELF_RAG_MAX_RETRIES_PER_HOP_MIN: int = 1
SELF_RAG_MAX_RETRIES_PER_HOP_MAX: int = 3
SELF_RAG_MAX_RETRIES_PER_HOP_STEP: int = 1
SELF_RAG_MAX_RETRIES_PER_HOP_HELP_MSG: str = "Maximum attempts to rewrite and retry failing sub-queries during retrieval. Higher values help recover from initial retrieval failure but add latency."

# Self-RAG thresholds for quality gating
SELF_RAG_THRESHOLD_ISSUP_MIN: float = 0.0
SELF_RAG_THRESHOLD_ISSUP_MAX: float = 1.0
SELF_RAG_THRESHOLD_ISSUP_STEP: float = 0.05
SELF_RAG_THRESHOLD_ISSUP_HELP_MSG: str = "Groundedness threshold: minimum confidence that the answer is supported by retrieved documents. Lower = lenient, Higher = strict."

SELF_RAG_THRESHOLD_ISREL_MIN: float = 0.0
SELF_RAG_THRESHOLD_ISREL_MAX: float = 1.0
SELF_RAG_THRESHOLD_ISREL_STEP: float = 0.05
SELF_RAG_THRESHOLD_ISREL_HELP_MSG: str = "Relevance threshold: minimum confidence that retrieved chunks are relevant to the original query. Lower = lenient, Higher = strict."

SELF_RAG_THRESHOLD_ISUSE_MIN: float = 0.0
SELF_RAG_THRESHOLD_ISUSE_MAX: float = 1.0
SELF_RAG_THRESHOLD_ISUSE_STEP: float = 0.05
SELF_RAG_THRESHOLD_ISUSE_HELP_MSG: str = "Utility threshold: minimum confidence that the answer is useful and addresses the user's intent. Lower = lenient, Higher = strict."

# ============================================================================
# SHARED RAG STYLE RULES — injected into both Self-RAG and Co-RAG prompts
# to enforce a unified "brand voice" across both pipelines.
# ============================================================================
SHARED_RAG_STYLE_RULES: str = """\
- Language Matching: Always respond in the EXACT same language as the user's query. Do NOT translate.
- No Structural Headers: Do NOT use "Answer:", "Source Citations:", "Additional Note:", or meta-commentary about the context. Write conversationally and directly.
- Grounding: Every factual claim must be supported by the provided context. Do not hallucinate or invent information beyond the documents.
- Conciseness: Avoid filler phrases, padding, or restating the question.
- Technical Terms: Preserve domain-specific vocabulary exactly as-is (e.g., RAG, LLM, API, FAISS). Do not translate them.
- Completeness: Cover all aspects of the query that the provided context supports. Do not stop before all relevant information from the context has been conveyed.
- Uncertainty: When the context is incomplete or ambiguous, be explicit about the gap (e.g., "The document does not cover...") rather than extrapolating beyond what is stated."""

# ============================================================================
# CO-RAG CONFIGURATIONS - Collaborative Generator ↔ Reviewer loop
# ============================================================================
# Maximum Generator ↔ Reviewer turns before accepting the current draft (0 = no review)
CO_RAG_MAX_RETRIES: int = 3

CO_RAG_MAX_RETRIES_MIN: int = 0
CO_RAG_MAX_RETRIES_MAX: int = 5
CO_RAG_MAX_RETRIES_STEP: int = 1
CO_RAG_MAX_RETRIES_HELP_MSG: str = (
    "Maximum Generator ↔ Reviewer collaboration turns. "
    "0 = single-pass generation with no review. Higher = more refined but slower."
)

# ============================================================================
# CO-RAG PROMPTS
# ============================================================================

# Generator (Mode A) — initial holistic answer grounded in the retrieved context
CO_RAG_GENERATOR_INITIAL_PROMPT: str = f"""You are the Lead Researcher. Using ONLY the provided document context, write a comprehensive answer to the user's query.

{{personal_ctx_section}}SHARED QUALITY STANDARDS:
{SHARED_RAG_STYLE_RULES}

ADDITIONAL RULES:
1. **Grounding**: Base every claim exclusively on the provided context. If context does not contain the answer, explicitly say so — do NOT invent information.
2. **Completeness**: Cover all aspects of the query that the context supports. Synthesize across multiple context passages if needed.
3. **Found Status**: At the very end of your answer, on its own line, include EXACTLY ONE of:
   - `[FOUND: YES]` — if your answer is primarily grounded in the retrieved documents
   - `[FOUND: NO]` — if the documents don't contain enough information to answer
   The `[FOUND: YES/NO]` tag must be the very last line of your response.

{{chat_history_section}}CONTEXT:
{{context}}

USER QUERY: {{query}}

ANSWER:"""

# Generator (Mode B) — targeted refinement based on Reviewer critique
CO_RAG_GENERATOR_REFINE_PROMPT: str = f"""You are the Lead Researcher in Editor Mode. Apply the Reviewer's critique as targeted redlines to your previous draft.

{{personal_ctx_section}}SHARED QUALITY STANDARDS:
{SHARED_RAG_STYLE_RULES}

ADDITIONAL RULES:
1. **Targeted Edits Only**: Fix ONLY the specific issues listed in the critique. Do NOT rewrite sections that the Reviewer marked as correct.
2. **Preserve Correct Content**: Keep all parts of the previous draft that the Reviewer did not flag as problematic.
3. **Grounding**: All added or changed content MUST be supported by the provided context. Do not introduce new claims beyond what the context contains.
4. **All Prior Issues**: Cross-check ALL turns in the critique history, not just the most recent. Ensure every previously flagged issue is resolved in this revision.
5. **Found Status**: At the very end of your revised answer, on its own line, include EXACTLY ONE of:
   - `[FOUND: YES]` — if your answer is primarily grounded in the retrieved documents
   - `[FOUND: NO]` — if the documents don't contain enough information to answer
   The `[FOUND: YES/NO]` tag must be the very last line of your response.

{{chat_history_section}}CONTEXT:
{{context}}

USER QUERY: {{query}}

PREVIOUS DRAFT:
{{draft}}

REVIEWER CRITIQUE:
{{critique}}

REVISED ANSWER:"""

# Reviewer — gap-analysis critique with critique history awareness
CO_RAG_REVIEWER_PROMPT: str = """You are the Critical Reviewer in a collaborative research loop.

YOUR TASK:
1. First, assess whether the draft adequately answers the user's query using the provided context.
2. Only if the draft is NOT adequate: identify the 1-3 most critical specific issues (factual errors, clear contradictions with context, or major missing points that the context explicitly covers).
3. If prior critiques exist, note what the Generator has already fixed — credit those improvements.
4. End your response with EXACTLY ONE status tag on its own line.

STATUS TAG DEFINITIONS:
- [STATUS: VERIFIED] — Draft adequately answers the query with reasonable grounding in the context. Minor additions or stylistic improvements are NOT a reason to withhold VERIFIED.
- [STATUS: PARTIAL_VERIFIED] — Draft is substantially correct but has 1-3 specific, clearly fixable factual issues. List only issues that are directly contradicted by or explicitly missing from the context. Do NOT use for stylistic preferences, extra detail, or minor wording.
- [STATUS: CRITICAL_ERROR] — Draft contains hallucinations, factual errors, or major omissions that significantly undermine the answer.

MANDATORY CONVERGENCE RULES:
- Turn 2 or later: If prior critiques have been substantially addressed and no new critical factual issues exist, you MUST use [STATUS: VERIFIED].
- Do NOT repeat a [STATUS: PARTIAL_VERIFIED] for the same issue raised in a prior turn. If the Generator did not fix it, escalate to [STATUS: CRITICAL_ERROR]. If the Generator fixed it, upgrade to [STATUS: VERIFIED].
- When in doubt on Turn 2 or later, prefer [STATUS: VERIFIED] over [STATUS: PARTIAL_VERIFIED].

CONTEXT:
{context}

USER QUERY: {query}

DRAFT TO REVIEW:
{draft}

PRIOR CRITIQUE HISTORY (previous turns, oldest first):
{critique_history}

CRITIQUE:"""

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
    "WARNING": {
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
