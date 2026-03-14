# embed_and_store.py
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from ingest import load_and_chunk_pdf
from langchain_core.documents import Document
from typing import List, Tuple


def create_vectorstore(pdf_path: str, vectorstore_path: str = "vectorstore"):
    """
    Load PDF, chunk it, embed all chunks, and save to FAISS.

    Args:
      pdf_path: Path to the PDF file
      vectorstore_path: Directory to save FAISS index

    Returns:
      FAISS vectorstore object
    """

    # Step 1: Load and chunk the PDF
    print("📄 Step 1: Loading and chunking PDF...")
    chunks = load_and_chunk_pdf(pdf_path)

    # Step 2: Initialize the embedding model
    print("\n🤖 Step 2: Initializing embedding model...")
    print("   (First run downloads the model ~400MB - this may take a minute)")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={"device": "cuda"},  # Change to "cuda" if you have GPU
    )

    print("   ✅ Embedding model loaded!")
    print("   Embedding dimension: 768")

    # Step 3: Create FAISS vectorstore
    print(f"\n🗂️  Step 3: Creating FAISS index from {len(chunks)} chunks...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print("   ✅ FAISS index created!")

    # Step 4: Save to disk
    print(f"\n💾 Step 4: Saving to {vectorstore_path}/...")
    vectorstore.save_local(vectorstore_path)

    print("   ✅ Saved successfully!")
    print(f"   Files: {vectorstore_path}/index.faiss (index)")
    print(f"         {vectorstore_path}/index.pkl (metadata)")

    return vectorstore


def retrieve_similar_chunks(
    vectorstore: FAISS,
    query: str,
    k: int = 5,
    min_results: int = 3,
    score_threshold: float = 10.0,
):
    """
    Retrieve top K most similar chunks for a query with intelligent filtering.

    First tries quality-based filtering (threshold), then falls back to top K if needed.

    Args:
      vectorstore: FAISS vectorstore object
      query: User's question
      k: Maximum number of chunks to retrieve
      min_results: Minimum number of results to guarantee (fallback)
      score_threshold: Distance threshold for quality filtering (lower = better)

    Returns:
      List of Document objects (similar chunks)
    """
    print(f"\n🔍 Retrieving chunks for query: '{query}'")
    print(f"   (Quality threshold: {score_threshold}, Min results: {min_results})")

    # Step 1: Get top K results with scores
    results_with_scores: List[Tuple[Document, float]] = (
        vectorstore.similarity_search_with_score(query, k=k)
    )

    # Step 2: Filter by threshold (quality-first approach)
    filtered_results: List[Document] = []
    for doc, score in results_with_scores:
        content_length = len(doc.page_content)

        # FAISS score is distance: smaller = better match
        if score <= score_threshold and content_length > 5:
            doc.metadata["similarity_score"] = score
            filtered_results.append(doc)

    # Step 3: Fallback mechanism - if threshold filtered out too much, use top min_results
    if len(filtered_results) < min_results:
        print(f"   ⚠️  Only {len(filtered_results)} chunks passed threshold.")
        print(f"   📌 Falling back to top {min_results} results for context...")

        filtered_results = []
        for doc, score in results_with_scores[:min_results]:
            doc.metadata["similarity_score"] = score
            filtered_results.append(doc)

    print(f"\n   ✅ Retrieved {len(filtered_results)} chunks!\n")

    for i, doc in enumerate(filtered_results):
        score = doc.metadata.get("similarity_score", "N/A")
        print(f"   --- Chunk {i + 1} (Page {doc.metadata.get('page', 'N/A')}) ---")
        print(f"   {doc.page_content[:100]}...")
        if isinstance(score, float):
            print(f"   (Distance Score: {score:.4f})\n")
        else:
            print(f"   (Distance Score: {score})\n")

    return filtered_results


if __name__ == "__main__":
    pdf_file = "human-nutrition-text.pdf"
    vectorstore_path = "vectorstore"

    try:
        # Create and save vectorstore
        vectorstore = create_vectorstore(pdf_file, vectorstore_path)

        # Test retrieval with a sample query
        print("\n" + "=" * 60)
        print("TEST: Retrieving relevant chunks")
        print("=" * 60)

        test_query = "What are macronutrients?"
        retrieve_similar_chunks(
            vectorstore, test_query, k=5, min_results=3, score_threshold=10.0
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
