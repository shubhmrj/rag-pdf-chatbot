"""
RAG logic — read PDFs, store in ChromaDB, answer with Groq LLM.
"""

import os
import re
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

DATA_DIR = Path("./data")
DB_DIR = Path("./chroma_db")

_embeddings = None
_chain = None


class LocalEmbeddings(Embeddings):
    def __init__(self):
        self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode([text], convert_to_numpy=True).tolist()[0]


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = LocalEmbeddings()
    return _embeddings


def reset_chain():
    global _chain
    _chain = None


def is_ready():
    return DB_DIR.exists() and any(DB_DIR.iterdir())


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _load_pdf(path: Path) -> list[Document]:
    """Extract text from a PDF using PyPDFLoader, with pypdf fallback."""
    pages: list[Document] = []

    try:
        for doc in PyPDFLoader(str(path)).load():
            text = _clean_text(doc.page_content)
            if text:
                pages.append(
                    Document(page_content=text, metadata=doc.metadata or {"source": str(path)})
                )
    except Exception:
        pages = []

    if pages:
        return pages

    try:
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
    except Exception as exc:
        raise ValueError(f"Could not read {path.name}: {exc}") from exc

    return pages


def index_pdfs(files: list[tuple[str, bytes]]):
    """Save PDF bytes to disk and build the Chroma vector store."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for name, content in files:
        if not name or not content:
            continue
        safe_name = Path(name).name
        path = DATA_DIR / safe_name
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
                errors.append(f"{path.name}: no readable text (maybe scanned/image PDF)")
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    if not all_pages:
        detail = "; ".join(errors) if errors else "unknown error"
        return False, f"Could not extract text from PDFs. {detail}"

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    ).split_documents(all_pages)

    chunks = [c for c in chunks if c.page_content.strip()]
    if not chunks:
        return False, "PDF had text but chunking failed. Try a different PDF."

    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)

    embeddings = _get_embeddings()
    batch_size = 100

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        if start == 0:
            Chroma.from_documents(batch, embeddings, persist_directory=str(DB_DIR))
        else:
            db = Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)
            db.add_documents(batch)

    reset_chain()
    msg = f"Indexed {len(chunks)} chunks from {len(saved_paths)} PDF(s)."
    if errors:
        msg += f" Warning: {'; '.join(errors)}"
    return True, msg


def ask(question: str):
    global _chain

    if not is_ready():
        return "Upload and index a PDF first.", []

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "GROQ_API_KEY is missing. Add it in .env or HF Space Secrets.", []

    if _chain is None:
        db = Chroma(persist_directory=str(DB_DIR), embedding_function=_get_embeddings())
        llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            groq_api_key=api_key,
            temperature=0,
        )

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a medical research assistant. Use ONLY the context below.

Context:
{context}

Question: {question}

Rules:
- Answer only from the context
- If not in the context, say "I don't know"
- Cite source file and page number
- For research only, not personal medical advice

Answer:""",
        )

        _chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )

    try:
        result = _chain.invoke({"query": question})
    except Exception as exc:
        reset_chain()
        return f"Groq API error: {exc}", []

    answer = result.get("result") or result.get("answer") or str(result)
    sources = list({
        f"{Path(d.metadata.get('source', '?')).name} p.{d.metadata.get('page', '?')}"
        for d in result.get("source_documents", [])
    })
    return answer, sources
