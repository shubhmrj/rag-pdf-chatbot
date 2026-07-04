---
title: Medical Research Assistant
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
---

# Medical Research Assistant — RAG Chatbot

Ask questions about medical research PDFs. Answers come only from your uploaded documents.

## How it works

1. **Upload PDFs** in the sidebar (research papers, clinical studies, etc.)
2. Click **Index documents** — text is split, embedded, and stored in ChromaDB
3. **Ask questions** in the chat — the app retrieves relevant chunks and Gemini answers with citations

## Hugging Face Space setup

1. Create a Space with **Docker** SDK
2. Add a secret: `GOOGLE_API_KEY` = your [Google AI API key](https://aistudio.google.com/apikey)
3. Push this repo — the Space builds and runs Streamlit on port 8501
4. Open the Space URL and upload PDFs in the sidebar

**You do not commit PDFs to git.** Upload them in the app after deploy.  
(Rebuilds wipe uploaded files unless you use persistent storage.)

## Run locally

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY=your_key
streamlit run streamlit_app.py
```

Optional: put PDFs in `./data/` and run `python ingest.py` before starting the app.

## Project files

| File | Purpose |
|------|---------|
| `streamlit_app.py` | UI — upload PDFs and chat |
| `ingest.py` | PDF → chunks → ChromaDB |
| `rag.py` | Retrieve context + Gemini answer |
| `embeddings.py` | Local sentence embeddings |

## Note

This tool is for **research only**, not personal medical advice.
