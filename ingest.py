import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

load_dotenv()

DATA_DIR = Path(os.getenv("PDF_FOLDER", "./data")).resolve()
PERSIST_DIRECTORY = Path(os.getenv("CHROMA_DB_PATH", "./chroma_db")).resolve()
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))


class SimpleEmbeddings:
    """Wrapper around SentenceTransformer for LangChain compatibility."""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    
    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True).tolist()[0]


def main():
    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        return

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {DATA_DIR}")
        return

    docs = []
    for path in pdf_files:
        loader = PyPDFLoader(str(path))
        docs.extend(loader.load())

    if not docs:
        print("No pages were extracted from the provided PDFs")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks from {len(docs)} pages")

    embeddings = SimpleEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Process in batches due to Chroma's 166 document limit per batch
    batch_size = 160
    vectordb = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        if vectordb is None:
            vectordb = Chroma.from_documents(batch, embeddings, persist_directory=str(PERSIST_DIRECTORY))
        else:
            vectordb.add_documents(batch)
        print(f"Processed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")
    
    vectordb.persist()
    print(f"Vector store saved to {PERSIST_DIRECTORY}")


if __name__ == "__main__":
    main()