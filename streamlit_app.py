import requests
import streamlit as st

API = "http://localhost:8000"

st.set_page_config(page_title="Medical Research Assistant", page_icon="🩺")
st.title("🩺 Medical Research Assistant")
st.write("Upload a medical research PDF, index it, then ask questions.")

if "history" not in st.session_state:
    st.session_state.history = []


def call_api(method, path, timeout=120, **kwargs):
    resp = requests.request(method, f"{API}{path}", timeout=timeout, **kwargs)

    if not resp.text.strip():
        raise RuntimeError(f"Empty response (HTTP {resp.status_code})")

    try:
        data = resp.json()
    except ValueError as exc:
        raise RuntimeError(f"Bad JSON: {resp.text[:200]}") from exc

    if resp.status_code >= 400:
        detail = data.get("detail", resp.text[:200]) if isinstance(data, dict) else resp.text[:200]
        raise RuntimeError(str(detail))

    return data


# --- Step 1: Upload ---
st.subheader("Step 1 — Upload & Index PDF")
uploaded = st.file_uploader("Choose PDF file(s)", type=["pdf"], accept_multiple_files=True)

if st.button("Index PDFs", type="primary"):
    if not uploaded:
        st.warning("Select at least one PDF first.")
    else:
        file_list = uploaded if isinstance(uploaded, list) else [uploaded]
        with st.spinner("Indexing PDFs... wait 1-2 minutes."):
            try:
                payload = [
                    ("files", (f.name, f.getvalue(), "application/pdf"))
                    for f in file_list
                ]
                resp = call_api("POST", "/index", files=payload, timeout=300)
                if resp.get("ok"):
                    st.session_state.index_msg = ("ok", resp["message"])
                else:
                    st.session_state.index_msg = ("err", resp["message"])
            except Exception as exc:
                st.session_state.index_msg = ("err", str(exc))

if "index_msg" in st.session_state:
    kind, msg = st.session_state.index_msg
    if kind == "ok":
        st.success(msg)
    else:
        st.error(msg)

# --- Step 2: Chat ---
try:
    ready = bool(call_api("GET", "/status", timeout=10).get("ready"))
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
            sources = resp.get("sources") or []
            if sources:
                answer += "\n\n**Sources:** " + ", ".join(str(s) for s in sources)
        except Exception as exc:
            answer = f"Error: {exc}"
    st.session_state.history.append({"role": "assistant", "content": answer})
    st.rerun()
