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

Simple RAG chatbot for medical research PDFs. **One file:** `app.py`

## How to use on Hugging Face Space

1. Add secret: **GOOGLE_API_KEY** (from https://aistudio.google.com/apikey)
2. Open your Space
3. **Step 1:** Upload PDF → click **Index PDFs** → wait 1-2 min
4. **Step 2:** Ask questions in chat

## How RAG works

```
PDF upload → split into chunks → save to ChromaDB
Question   → find similar chunks → Gemini writes answer + cites page
```

## Run locally

```bash
pip install -r requirements.txt
set GOOGLE_API_KEY=your_key
streamlit run app.py
```

## Files

| File | What it does |
|------|--------------|
| `app.py` | Everything — upload, index, chat |
| `requirements.txt` | Python packages |
| `Dockerfile` | HF Space build |
| `start.sh` | Starts the app |
