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


def call_api(method, path, timeout=120, **kwargs):
    url = f"{API}{path}"
    resp = requests.request(method, url, timeout=timeout, **kwargs)

    if not resp.text.strip():
        raise RuntimeError(f"Empty response from API ({resp.status_code})")

    try:
        data = resp.json()
    except ValueError:
        raise RuntimeError(f"Bad API response: {resp.text[:300]}")

    if resp.status_code >= 400:
        raise RuntimeError(data.get("detail", resp.text[:300]))

    return data


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
                resp = call_api("POST", "/index", files=files, timeout=300)
                st.success(resp["message"]) if resp["ok"] else st.error(resp["message"])
            except Exception as exc:
                st.error(f"Index error: {exc}")

# --- Step 2: Chat ---
try:
    ready = call_api("GET", "/status", timeout=10)["ready"]
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
            resp = call_api("POST", "/chat", json={"question": prompt}, timeout=300)
            answer = resp.get("answer", "No answer returned.")
            if resp.get("sources"):
                answer += "\n\n**Sources:** " + ", ".join(resp["sources"])
        except Exception as exc:
            answer = f"Error: {exc}"
    st.session_state.history.append({"role": "assistant", "content": answer})
    st.rerun()
