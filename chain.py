import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer

load_dotenv()

PERSIST_DIRECTORY = Path(os.getenv("CHROMA_DB_PATH", "./chroma_db")).resolve()


class SimpleEmbeddings:
    """Wrapper around SentenceTransformer for LangChain compatibility."""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    
    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True).tolist()[0]


def _build_prompt():
    prompt_template = """You are an expert assistant answering questions
based on the provided document context only.

Context:
{context}

Question: {question}

Instructions:
- Answer using ONLY the context above
- If the answer isn't in the context, say "I don't know"
- Cite the source document and page number

Answer:"""
    return PromptTemplate(template=prompt_template, input_variables=["context", "question"])


_chain = None


def _get_chain():
    global _chain
    if _chain is not None:
        return _chain

    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set")

        embeddings = SimpleEmbeddings(model_name="all-MiniLM-L6-v2")
        if not PERSIST_DIRECTORY.exists():
            raise FileNotFoundError(f"Vector store directory not found: {PERSIST_DIRECTORY}")

        vectordb = Chroma(persist_directory=str(PERSIST_DIRECTORY), embedding_function=embeddings)
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 4})
        _chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": _build_prompt()},
            return_source_documents=True,
        )
    except Exception as exc:
        print(f"Warning: unable to initialize retrieval chain: {exc}")
        _chain = None

    return _chain


def qa_chain(payload):
    chain = _get_chain()
    if chain is None:
        return {
            "result": "I don't have indexed documents yet. Run ingest.py to build the knowledge base first.",
            "source_documents": [],
        }
    return chain(payload)