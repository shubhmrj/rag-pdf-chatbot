#!/usr/bin/env bash
set -euo pipefail

# Start FastAPI backend
uvicorn api:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait until API is ready (max 30 seconds)
for i in $(seq 1 30); do
  if curl -sf http://localhost:8000/status > /dev/null 2>&1; then
    echo "API ready"
    break
  fi
  sleep 1
done

# Start Streamlit frontend
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0

kill $API_PID 2>/dev/null || true
