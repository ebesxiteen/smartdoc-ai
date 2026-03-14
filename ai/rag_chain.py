# rag_chain.py
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import List
from langchain_core.documents import Document


def load_vectorstore(vectorstore_path: str = "vectorstore") -> FAISS:
    """
    Load the saved FAISS vectorstore from disk.

    Args:
      vectorstore_path: Path to the vectorstore directory

    Returns:
      FAISS vectorstore object
    """
    print("📂 Loading FAISS vectorstore from disk...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={"device": "cuda"},
    )

    vectorstore = FAISS.load_local(
        vectorstore_path, embeddings, allow_dangerous_deserialization=True
    )
    print(f"   ✅ Vectorstore loaded!")

    return vectorstore


def format_context_with_sources(docs: List[Document]) -> str:
    """
    Format retrieved documents into a context string with source citations.

    Args:
      docs: List of Document objects

    Returns:
      Formatted context string with page numbers and citations
    """
    if not docs:
        return "No relevant context found."

    context_parts = []
    for _, doc in enumerate(docs, 1):
        page_num = doc.metadata.get("page", "Unknown")
        # Clean up the text: remove extra whitespace
        text = " ".join(doc.page_content.split())
        context_parts.append(f"[Source: Page {page_num}]\n{text}")

    return "\n\n".join(context_parts)


def create_rag_chain(vectorstore: FAISS):
    """
    Build the complete RAG chain: Retriever → Format Context → Prompt → LLM

    Args:
        vectorstore: FAISS vectorstore object

    Returns:
        Runnable chain object
    """
    print("\n🔗 Building RAG Chain...")

    # Step 1: Create retriever with our parameters
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},  # Top 5 chunks
    )

    # Step 2: Define the custom prompt template with source instructions
    prompt_template = PromptTemplate(
        template="""You are a helpful research assistant. Your role is to answer questions based ONLY on the provided document content.

CRITICAL CONSTRAINTS:
1. You MUST answer using ONLY the information from the provided context.
2. Do NOT use your pre-trained knowledge. Do NOT hallucinate or invent information.
3. If the answer is not found in the context, explicitly state: "The provided sources do not contain information about this."
4. When citing information, naturally reference the source page number in your response.
5. Be concise but comprehensive in your answers.
6. Answer in the same language as the user's question (detect Vietnamese, English, etc.).

CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION: {question}

YOUR ANSWER (grounded only in the provided context):""",
        input_variables=["context", "question"],
    )

    # Step 3: Initialize the LLM (Ollama running Qwen2.5)
    llm = OllamaLLM(
        model="qwen2.5:7b",
        base_url="http://localhost:11434",
        temperature=0.7,  # Slightly creative but grounded
    )

    # Step 4: Build the chain using LCEL
    # This chains: question → retriever → format context → prompt → llm → output
    rag_chain = (
        {
            "context": retriever
            | RunnablePassthrough()
            | (lambda docs: format_context_with_sources(docs)),
            "question": RunnablePassthrough(),
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    print("   ✅ RAG Chain created!")
    print("\n   Chain Flow:")
    print(
        "   Question → Retriever (FAISS) → Context Formatter → Prompt → LLM (Qwen2.5) → Answer"
    )

    return rag_chain


def answer_question(rag_chain, query: str):
    """
    Ask a question and get an answer from the RAG chain.

    Args:
      rag_chain: The compiled RAG chain
      query: User's question

    Returns:
      Answer string with citations
    """
    print(f"\n" + "=" * 70)
    print(f"QUESTION: {query}")
    print("=" * 70)

    print("\n🤔 Thinking...")
    answer = rag_chain.invoke(query)

    print("\n💡 ANSWER:")
    print("-" * 70)
    print(answer)
    print("-" * 70)

    return answer


if __name__ == "__main__":
    try:
        # Load the vectorstore
        vectorstore = load_vectorstore("vectorstore")

        # Create the RAG chain
        rag_chain = create_rag_chain(vectorstore)

        # Test with sample questions
        print("\n" + "=" * 70)
        print("TESTING RAG SYSTEM")
        print("=" * 70)

        answer_question(rag_chain, "What is the capital of France?")
        # test_queries = [
        #   "What are macronutrients?",
        #   "What are the benefits of protein?",
        #   "Explain carbohydrates and their role in nutrition."
        # ]

        # for query in test_queries:
        #   answer_question(rag_chain, query)
        #   print("\n")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
