#!/usr/bin/env bash
set -euo pipefail

# Backend (RAG + API) on port 8000
uvicorn api:app --host 0.0.0.0 --port 8000 &

# Frontend (Streamlit UI) on port 8501
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
