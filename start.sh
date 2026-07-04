#!/usr/bin/env bash
set -euo pipefail

# If GCP service account JSON is provided as a secret, write it to a file and export
if [ -n "${GCP_SERVICE_ACCOUNT_JSON-}" ]; then
  echo "$GCP_SERVICE_ACCOUNT_JSON" > /tmp/gcp_sa.json
  export GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp_sa.json
fi

# Start the FastAPI backend on port 8000
uvicorn app:app --host 0.0.0.0 --port 8000 &

# Start Streamlit frontend on port 8501
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
