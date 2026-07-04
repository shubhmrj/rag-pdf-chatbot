import requests
import streamlit as st

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("PDF Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    try:
        resp = requests.post("http://localhost:8000/chat", json={"question": prompt}, timeout=60).json()
        answer = resp["answer"]
        if resp.get("sources"):
            answer += "\n\n**Sources:** " + ", ".join(resp["sources"])
        st.session_state.history.append({"role": "assistant", "content": answer})
    except Exception as exc:
        st.session_state.history.append({"role": "assistant", "content": f"Error: {exc}"})
    st.rerun()
