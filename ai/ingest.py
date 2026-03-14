# ingest.py
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_chunk_pdf(pdf_path: str):
    """
    Load a PDF and chunk it into overlapping segments.

    Args:
      pdf_path: Path to the PDF file

    Returns:
      List of Document objects with chunked text
    """
    # Step 1: Load the PDF
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    print(f"✅ Loaded PDF: {pdf_path}")
    print(f"   Total pages: {len(documents)}")
    print(f"   Total characters: {sum(len(doc.page_content) for doc in documents)}")

    # Step 2: Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 1000 characters per chunk
        chunk_overlap=200,  # 200 character overlap (20%)
        separators=[
            "\n\n",
            "\n",
            " ",
            "",
        ],  # Split by paragraph, then line, then space, then character
    )

    chunks = text_splitter.split_documents(documents)

    print("\n✅ Chunking complete!")
    print(f"   Total chunks: {len(chunks)}")
    print(
        f"   Average chunk size: {sum(len(chunk.page_content) for chunk in chunks) / len(chunks):.0f} characters"
    )

    return chunks


if __name__ == "__main__":
    # Test with a sample PDF
    pdf_file = "human-nutrition-text.pdf"  # Replace with your actual PDF path

    try:
        chunks = load_and_chunk_pdf(pdf_file)

        # Print first 2 chunks as a preview
        print("\n" + "=" * 50)
        print("PREVIEW: First 2 chunks")
        print("=" * 50)
        for i, chunk in enumerate(chunks[:2]):
            print(f"\n--- Chunk {i + 1} (Page {chunk.metadata.get('page', 'N/A')}) ---")
            print(chunk.page_content[:200] + "...")  # First 200 chars

    except FileNotFoundError:
        print(
            f"❌ Error: {pdf_file} not found. Please place a PDF in your project directory."
        )
    except Exception as e:
        print(f"❌ Error: {e}")
