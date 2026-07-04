---
title: Medical Research Assistant
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
---

# Medical Research Assistant

3 files, easy to understand:

| File | Role |
|------|------|
| `rag.py` | Brain — read PDF, store chunks, answer questions |
| `api.py` | Backend — FastAPI routes `/index` and `/chat` |
| `streamlit_app.py` | Frontend — upload UI + chat |

## How it works

```
User uploads PDF in Streamlit
    → api.py receives file
    → rag.py reads PDF, saves to ChromaDB

User asks question in Streamlit
    → api.py receives question
    → rag.py finds similar chunks → Gemini answers
```

## Hugging Face Space setup

1. Add secret: `GOOGLE_API_KEY`
2. Push code and wait for build
3. Upload PDF → click **Index PDFs**
4. Ask questions

## Run locally

```bash
pip install -r requirements.txt
set GOOGLE_API_KEY=your_key
bash start.sh
```

Open http://localhost:8501
