"""
RAG logic — the brain of the app.

Step 1: index_pdfs()  → read PDF, split text, save to ChromaDB
Step 2: ask()         → find relevant chunks, send to Gemini, return answer
"""

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

load_dotenv()

DATA_DIR = Path("./data")
DB_DIR = Path("./chroma_db")

_embeddings = None
_chain = None


def _get_embeddings():
    """Turn text into numbers (vectors) so we can search similar text."""
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    model = SentenceTransformer("all-MiniLM-L6-v2")

    class Embeddings:
        def embed_documents(self, texts):
            return model.encode(texts, convert_to_numpy=True).tolist()

        def embed_query(self, text):
            return model.encode([text], convert_to_numpy=True).tolist()[0]

    _embeddings = Embeddings()
    return _embeddings


def reset_chain():
    """Call after indexing new PDFs so the chat uses fresh data."""
    global _chain
    _chain = None


def is_ready():
    return DB_DIR.exists()


def index_pdfs(files):
    """
    Index PDF files.
    files = list of (filename, bytes) tuples
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for name, content in files:
        path = DATA_DIR / name
        path.write_bytes(content)
        saved_paths.append(path)

    pages = []
    for path in saved_paths:
        pages.extend(PyPDFLoader(str(path)).load())

    if not pages:
        return False, "Could not read text from the PDFs."

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).split_documents(pages)

    if not chunks:
        return False, "No text chunks were created from the PDFs."

    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)

    embeddings = _get_embeddings()
    batch_size = 100

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        if i == 0:
            Chroma.from_documents(
                batch, embeddings, persist_directory=str(DB_DIR)
            )
        else:
            db = Chroma(
                persist_directory=str(DB_DIR), embedding_function=embeddings
            )
            db.add_documents(batch)

    reset_chain()
    return True, f"Indexed {len(chunks)} chunks from {len(saved_paths)} PDF(s)."


def ask(question):
    """Answer a question using indexed PDFs."""
    global _chain

    if not is_ready():
        return "Upload and index a PDF first.", []

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "GOOGLE_API_KEY is missing. Add it in HF Space Secrets.", []

    if _chain is None:
        db = Chroma(
            persist_directory=str(DB_DIR), embedding_function=_get_embeddings()
        )
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
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
        return f"Could not get answer from Gemini: {exc}", []

    answer = result.get("result") or result.get("answer") or str(result)
    sources = list({
        f"{d.metadata.get('source', '?')} p.{d.metadata.get('page', '?')}"
        for d in result.get("source_documents", [])
    })
    return answer, sources
