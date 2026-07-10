import streamlit as st

from rag import ask, index_pdfs, is_ready

st.set_page_config(page_title="Medical Research Assistant", page_icon="🩺")
st.title(" Medical Research Assistant")
st.write("Upload medical research PDFs, index them, then ask questions.")

if "history" not in st.session_state:
    st.session_state.history = []


st.subheader("Step 1 — Upload and index PDFs")
uploaded = st.file_uploader("Choose PDF file(s)", type=["pdf"], accept_multiple_files=True)

if st.button("Index PDFs", type="primary"):
    if not uploaded:
        st.warning("Select at least one PDF first.")
    else:
        files = uploaded if isinstance(uploaded, list) else [uploaded]
        with st.spinner("Indexing PDFs..."):
            try:
                payload = [(file.name, file.getvalue()) for file in files]
                ok, message = index_pdfs(payload)
                st.session_state.index_message = ("ok", message) if ok else ("error", message)
            except Exception as exc:
                st.session_state.index_message = ("error", str(exc))

if "index_message" in st.session_state:
    kind, message = st.session_state.index_message
    if kind == "ok":
        st.success(message)
    else:
        st.error(message)


st.subheader("Step 2 — Ask questions")
ready = is_ready()
if ready:
    st.success("Documents indexed. You can chat now.")
else:
    st.info("No documents indexed yet. Complete Step 1 first.")

for item in st.session_state.history:
    role = item.get("role", "assistant")
    content = item.get("content", "")
    label = "You" if role == "user" else "Assistant"
    st.markdown(f"**{label}:** {content}")

with st.form("chat_form"):
    prompt = st.text_input("Ask about your research paper...", disabled=not ready)
    submitted = st.form_submit_button("Send")

if submitted and prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.spinner("Thinking..."):
        try:
            answer, sources = ask(prompt)
            if sources:
                answer += "\n\n**Sources:** " + ", ".join(sources)
        except Exception as exc:
            answer = f"Error: {exc}"
    st.session_state.history.append({"role": "assistant", "content": answer})
