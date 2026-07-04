import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embeddings import SimpleEmbeddings

load_dotenv()

DATA_DIR = Path(os.getenv("PDF_FOLDER", "./data")).resolve()
DB_DIR = Path(os.getenv("CHROMA_DB_PATH", "./chroma_db")).resolve()
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))


def ingest_pdfs(pdf_paths=None):
    """Read PDFs, chunk them, and save to Chroma."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if pdf_paths is None:
        pdf_paths = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_paths:
        return "No PDF files found. Upload or add PDFs to the data folder."

    docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(str(path))
        docs.extend(loader.load())

    if not docs:
        return "No text could be read from the PDFs."

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)

    embeddings = SimpleEmbeddings()
    batch_size = 160
    vectordb = None

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        if vectordb is None:
            vectordb = Chroma.from_documents(
                batch, embeddings, persist_directory=str(DB_DIR)
            )
        else:
            vectordb.add_documents(batch)

    vectordb.persist()
    return f"Indexed {len(chunks)} chunks from {len(pdf_paths)} PDF(s)."


if __name__ == "__main__":
    print(ingest_pdfs())
