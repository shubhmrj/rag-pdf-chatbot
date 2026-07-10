import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .config import CHUNK_OVERLAP, CHUNK_SIZE


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    return re.sub(r"\s+", " ", text).strip()


def load_pdf_documents(path: Path) -> list[Document]:
    pages: list[Document] = []

    try:
        for doc in PyPDFLoader(str(path)).load():
            text = clean_text(doc.page_content)
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
    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append(
                Document(
                    page_content=text,
                    metadata={"source": str(path), "page": page_number},
                )
            )

    return pages


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    return [chunk for chunk in chunks if chunk.page_content.strip()]
