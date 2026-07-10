import os
from pathlib import Path

from groq import Groq
from langchain_core.documents import Document

from .config import DATA_DIR, GROQ_BASE_URL, GROQ_MODEL, SEARCH_K
from .pdfs import load_pdf_documents, split_documents
from .vector_store import is_ready, replace_index, search_documents


def _save_uploaded_files(files: list[tuple[str, bytes]]) -> list[Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for name, content in files:
        if not name or not content:
            continue

        path = DATA_DIR / Path(name).name
        path.write_bytes(content)
        saved_paths.append(path)

    return saved_paths


def _load_all_pages(saved_paths: list[Path]) -> tuple[list[Document], list[str]]:
    all_pages: list[Document] = []
    errors: list[str] = []

    for path in saved_paths:
        try:
            pages = load_pdf_documents(path)
            if pages:
                all_pages.extend(pages)
            else:
                errors.append(f"{path.name}: no readable text found")
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    return all_pages, errors


def index_pdfs(files: list[tuple[str, bytes]]) -> tuple[bool, str]:
    saved_paths = _save_uploaded_files(files)
    if not saved_paths:
        return False, "No valid PDF files were uploaded."

    all_pages, errors = _load_all_pages(saved_paths)
    if not all_pages:
        return False, "Could not extract text. " + "; ".join(errors)

    chunks = split_documents(all_pages)
    if not chunks:
        return False, "PDF had text but chunking failed."

    replace_index(chunks)

    message = f"Indexed {len(chunks)} chunks from {len(saved_paths)} PDF(s)."
    if errors:
        message += " Note: " + "; ".join(errors)
    return True, message


def _format_sources(documents: list[Document]) -> list[str]:
    sources = []
    for doc in documents:
        source_name = Path(doc.metadata.get("source", "?")).name
        page_number = doc.metadata.get("page", "?")
        sources.append(f"{source_name} p.{page_number}")

    return sorted(set(sources))


def ask(question: str) -> tuple[str, list[str]]:
    if not is_ready():
        return "Upload and index a PDF first.", []

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "GROQ_API_KEY missing. Add it in .env or deployment secrets.", []

    documents = search_documents(question, k=SEARCH_K)
    if not documents:
        return "No matching content found in the indexed PDF.", []

    context = "\n\n".join(doc.page_content for doc in documents)
    sources = _format_sources(documents)

    try:
        client = Groq(
            api_key=api_key,
            base_url=GROQ_BASE_URL,
            max_retries=3,
        )
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a medical research assistant. "
                        "Answer only from the provided context. "
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
    except Exception as exc:
        message = str(exc)
        if "Connection" in message or "ConnectError" in message or "Network" in message:
            return (
                "Groq API error: connection failed. Check network access and Groq settings.",
                sources,
            )
        return f"Groq API error: {exc}", sources

    answer = response.choices[0].message.content or "No answer returned."
    return answer, sources
