import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

from embeddings import SimpleEmbeddings

load_dotenv()

DB_DIR = Path(os.getenv("CHROMA_DB_PATH", "./chroma_db")).resolve()

PROMPT = """You are a medical research assistant. Answer using ONLY the context below.

Context:
{context}

Question: {question}

Rules:
- Use only information from the context
- If the answer is not in the context, say "I don't know"
- Cite the source file and page number
- This is for research only, not personal medical advice

Answer:"""

_chain = None


def reset_chain():
    global _chain
    _chain = None


def _get_chain():
    global _chain
    if _chain is not None:
        return _chain

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set in Space secrets")

    if not DB_DIR.exists():
        raise FileNotFoundError("No indexed documents yet. Upload PDFs first.")

    embeddings = SimpleEmbeddings()
    vectordb = Chroma(
        persist_directory=str(DB_DIR), embedding_function=embeddings
    )
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 4})

    _chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template=PROMPT, input_variables=["context", "question"]
            )
        },
        return_source_documents=True,
    )
    return _chain


def ask(question):
    try:
        result = _get_chain()({"query": question})
    except Exception as exc:
        return str(exc), []

    sources = [
        f"{d.metadata.get('source', '?')} p.{d.metadata.get('page', '?')}"
        for d in result.get("source_documents", [])
    ]
    return result["result"], list(set(sources))
