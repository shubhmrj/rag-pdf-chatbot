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

RAG chatbot for medical research PDFs. Uses **Groq** (free tier) for answers.

| File | Role |
|------|------|
| `rag.py` | Read PDF, store chunks, answer with Groq |
| `api.py` | FastAPI backend |
| `streamlit_app.py` | Upload + chat UI |

## Hugging Face Space setup

1. Settings → Secrets → add **`GROQ_API_KEY`** (from https://console.groq.com)
2. Push code and wait for build
3. Upload PDF → click **Index PDFs**
4. Ask questions

## Run locally

```bash
pip install -r requirements.txt
copy .env.example .env
# put your GROQ_API_KEY in .env
bash start.sh
```

Open http://localhost:8501

## Notes

- PDF must have **selectable text** (not scanned images)
- Default model: `llama-3.1-8b-instant` (fast, free on Groq)
