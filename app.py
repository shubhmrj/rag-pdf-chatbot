"""
Medical Research Assistant — simple RAG chatbot

How RAG works (3 steps):
  1. Upload PDF  →  text is extracted from pages
  2. Index       →  text is split into chunks and stored in a vector database
  3. Chat        →  relevant chunks are found and sent to Gemini for an answer
"""

import os
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

load_dotenv()

# --- paths ---
DATA_DIR = Path("./data")
DB_DIR = Path("./chroma_db")

# --- cached models (load once, reuse) ---
@st.cache_resource
def get_embedding_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    class Embeddings:
        def embed_documents(self, texts):
            return model.encode(texts, convert_to_numpy=True).tolist()

        def embed_query(self, text):
            return model.encode([text], convert_to_numpy=True).tolist()[0]

    return Embeddings()


@st.cache_resource
def get_qa_chain():
    if not DB_DIR.exists():
        return None

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Add GOOGLE_API_KEY in Hugging Face Space → Settings → Secrets")
        return None

    db = Chroma(persist_directory=str(DB_DIR), embedding_function=get_embedding_model())
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

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

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )


def index_pdfs(pdf_files):
    """Save PDFs and build the vector database."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for f in pdf_files:
        path = DATA_DIR / f.name
        path.write_bytes(f.getvalue())
        saved_paths.append(path)

    pages = []
    for path in saved_paths:
        pages.extend(PyPDFLoader(str(path)).load())

    if not pages:
        return False, "Could not read text from the PDFs."

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).split_documents(pages)

    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)

    db = Chroma.from_documents(
        chunks, get_embedding_model(), persist_directory=str(DB_DIR)
    )
    db.persist()

    get_qa_chain.clear()
    return True, f"Ready! Indexed {len(chunks)} chunks from {len(saved_paths)} PDF(s)."


def ask(question):
    chain = get_qa_chain()
    if chain is None:
        return "Please upload and index a PDF first (see Step 1 above).", []

    result = chain({"query": question})
    sources = list({
        f"{d.metadata.get('source', '?')} p.{d.metadata.get('page', '?')}"
        for d in result.get("source_documents", [])
    })
    return result["result"], sources


# --- UI ---
st.set_page_config(page_title="Medical Research Assistant", page_icon="🩺")
st.title("🩺 Medical Research Assistant")
st.write("Upload a medical research PDF, index it, then ask questions.")

if "history" not in st.session_state:
    st.session_state.history = []

# Step 1: always visible on main page
st.subheader("Step 1 — Upload & Index PDF")
uploaded = st.file_uploader("Choose PDF file(s)", type="pdf", accept_multiple_files=True)

if st.button("Index PDFs", type="primary"):
    if not uploaded:
        st.warning("Select at least one PDF first.")
    else:
        with st.spinner("Reading and indexing PDFs... wait 1-2 minutes."):
            ok, msg = index_pdfs(uploaded)
        st.success(msg) if ok else st.error(msg)

is_ready = DB_DIR.exists()
st.subheader("Step 2 — Ask Questions")
if is_ready:
    st.success("Documents indexed. You can chat now.")
else:
    st.info("No documents indexed yet. Complete Step 1 first.")

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your research paper...", disabled=not is_ready):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.spinner("Thinking..."):
        answer, sources = ask(prompt)
    if sources:
        answer += "\n\n**Sources:** " + ", ".join(sources)
    st.session_state.history.append({"role": "assistant", "content": answer})
    st.rerun()
