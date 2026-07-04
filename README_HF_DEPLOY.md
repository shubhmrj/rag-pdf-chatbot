Deployment to Hugging Face Spaces
================================

Steps to deploy this project to your Hugging Face Space (account: https://huggingface.co/shubmrj):

1. Create a new Space
   - Go to https://huggingface.co/spaces and click "Create new Space".
   - Give it a name, select `Docker` as the SDK (important), and choose public/private as desired.

2. Push this repository to the Space
   - On the Space page you'll get a Git URL. Add it as a remote and push your repo, or create the Space from this repo.

3. Add secrets (Settings → Secrets)
   - `GCP_SERVICE_ACCOUNT_JSON`: paste the full service-account JSON of your Google Cloud service account (if you use Google Generative API). The `start.sh` writes this to `/tmp/gcp_sa.json` and sets `GOOGLE_APPLICATION_CREDENTIALS`.
   - `GOOGLE_API_KEY` or other provider keys as needed (e.g., `OPENAI_API_KEY`).

4. Ensure billing & quota
   - If using Google Generative Models (Gemini), enable billing on your Google Cloud project and request the appropriate quota.

5. Build & run
   - Spaces will build your Dockerfile and run `start.sh` (CMD in the Dockerfile). The container starts both the FastAPI backend (uvicorn) and the Streamlit frontend.

6. Notes and troubleshooting
   - The backend listens on port `8000` and the Streamlit frontend on `8501` inside the container. Hugging Face will expose the web UI.
   - If you run into quota/billing errors from the Google API, check Cloud Console and your secrets.
