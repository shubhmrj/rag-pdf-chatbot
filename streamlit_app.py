from pathlib import Path

import streamlit as st

from ingest import DATA_DIR, ingest_pdfs
from rag import ask, reset_chain

st.set_page_config(
    page_title="Medical Research Assistant",
    page_icon="🩺",
    layout="wide",
)

st.title("Medical Research Assistant")
st.caption("Upload medical research PDFs, then ask questions about them.")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader(
        "Upload PDF research papers",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("Index documents", type="primary"):
        if not uploaded:
            st.warning("Upload at least one PDF first.")
        else:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            saved = []
            for file in uploaded:
                path = DATA_DIR / file.name
                path.write_bytes(file.getvalue())
                saved.append(path)

            with st.spinner("Indexing PDFs... this may take a minute."):
                msg = ingest_pdfs(saved)
                reset_chain()
            st.success(msg)

    pdf_count = len(list(DATA_DIR.glob("*.pdf"))) if DATA_DIR.exists() else 0
    st.info(f"{pdf_count} PDF(s) saved in data folder")

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your research papers..."):
    st.session_state.history.append({"role": "user", "content": prompt})

    with st.spinner("Searching documents..."):
        answer, sources = ask(prompt)

    if sources:
        answer += "\n\n**Sources:** " + ", ".join(sources)

    st.session_state.history.append({"role": "assistant", "content": answer})
    st.rerun()
