"""
RAG logic — read PDFs, store in ChromaDB, answer with Groq LLM.
"""

import os
import re
import shutil
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

DATA_DIR = Path("./data")
DB_DIR = Path("./chroma_db")

_embeddings = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings


def is_ready():
    return DB_DIR.exists() and any(DB_DIR.iterdir())


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    return re.sub(r"\s+", " ", text).strip()


def _load_pdf(path: Path) -> list[Document]:
    pages: list[Document] = []

    try:
        for doc in PyPDFLoader(str(path)).load():
            text = _clean_text(doc.page_content)
            if text:
                pages.append(
                    Document(
                        page_content=text,
                        metadata=doc.metadata or {"source": str(path)},
                    )
                )
    except Exception:
        pages = []

    if pages:
        return pages

    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages):
        text = _clean_text(page.extract_text() or "")
        if text:
            pages.append(
                Document(
                    page_content=text,
                    metadata={"source": str(path), "page": i},
                )
            )
    return pages


def index_pdfs(files: list[tuple[str, bytes]]):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for name, content in files:
        if not name or not content:
            continue
        path = DATA_DIR / Path(name).name
        path.write_bytes(content)
        saved_paths.append(path)

    if not saved_paths:
        return False, "No valid PDF files were uploaded."

    all_pages: list[Document] = []
    errors: list[str] = []

    for path in saved_paths:
        try:
            pages = _load_pdf(path)
            if pages:
                all_pages.extend(pages)
            else:
                errors.append(f"{path.name}: no readable text (scanned/image PDF?)")
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    if not all_pages:
        return False, "Could not extract text. " + "; ".join(errors)

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).split_documents(all_pages)
    chunks = [c for c in chunks if c.page_content.strip()]

    if not chunks:
        return False, "PDF had text but chunking failed."

    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)

    Chroma.from_documents(
        chunks,
        _get_embeddings(),
        persist_directory=str(DB_DIR),
    )

    msg = f"Indexed {len(chunks)} chunks from {len(saved_paths)} PDF(s)."
    if errors:
        msg += " Note: " + "; ".join(errors)
    return True, msg


def ask(question: str):
    if not is_ready():
        return "Upload and index a PDF first.", []

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "GROQ_API_KEY missing. Add it in .env or HF Space Secrets.", []

    db = Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=_get_embeddings(),
    )
    docs = db.similarity_search(question, k=4)

    if not docs:
        return "No matching content found in the indexed PDF.", []

    context = "\n\n".join(d.page_content for d in docs)
    sources = list({
        f"{Path(d.metadata.get('source', '?')).name} p.{d.metadata.get('page', '?')}"
        for d in docs
    })

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a medical research assistant. "
                        "Answer ONLY from the provided context. "
                        "If the answer is not in the context, say 'I don't know'. "
                        "Cite source file and page. Research use only."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
            temperature=0,
        )
        answer = response.choices[0].message.content or "No answer returned."
    except Exception as exc:
        return f"Groq API error: {exc}", sources

    return answer, sources
