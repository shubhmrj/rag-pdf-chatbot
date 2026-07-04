"""
Streamlit UI — what the user sees in the browser.
Talks to api.py over http://localhost:8000
"""

import requests
import streamlit as st

API = "http://localhost:8000"

st.set_page_config(page_title="Medical Research Assistant", page_icon="🩺")
st.title("🩺 Medical Research Assistant")
st.write("Upload a medical research PDF, index it, then ask questions.")

if "history" not in st.session_state:
    st.session_state.history = []


def api_get(path):
    return requests.get(f"{API}{path}", timeout=10).json()


def api_post(path, **kwargs):
    return requests.post(f"{API}{path}", timeout=120, **kwargs).json()


# --- Step 1: Upload ---
st.subheader("Step 1 — Upload & Index PDF")
uploaded = st.file_uploader("Choose PDF file(s)", type="pdf", accept_multiple_files=True)

if st.button("Index PDFs", type="primary"):
    if not uploaded:
        st.warning("Select at least one PDF first.")
    else:
        with st.spinner("Indexing PDFs... wait 1-2 minutes."):
            try:
                files = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded]
                resp = api_post("/index", files=files)
                st.success(resp["message"]) if resp["ok"] else st.error(resp["message"])
            except Exception as exc:
                st.error(f"Could not reach API. Is the backend running? {exc}")

# --- Step 2: Chat ---
try:
    ready = api_get("/status")["ready"]
except Exception:
    ready = False

st.subheader("Step 2 — Ask Questions")
if ready:
    st.success("Documents indexed. You can chat now.")
else:
    st.info("No documents indexed yet. Complete Step 1 first.")

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your research paper...", disabled=not ready):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.spinner("Thinking..."):
        try:
            resp = api_post("/chat", json={"question": prompt})
            answer = resp["answer"]
            if resp.get("sources"):
                answer += "\n\n**Sources:** " + ", ".join(resp["sources"])
        except Exception as exc:
            answer = f"Error: {exc}"
    st.session_state.history.append({"role": "assistant", "content": answer})
    st.rerun()
