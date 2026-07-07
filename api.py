"""
FastAPI backend — connects the UI to RAG logic.
"""

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from rag import ask, index_pdfs, is_ready

app = FastAPI()


class Question(BaseModel):
    question: str


@app.get("/status")
def status():
    return {"ready": is_ready()}


@app.post("/index")
async def index_documents(files: list[UploadFile] = File(...)):
    try:
        if not files:
            return {"ok": False, "message": "No files uploaded."}

        file_data: list[tuple[str, bytes]] = []
        for upload in files:
            content = await upload.read()
            if content and upload.filename:
                file_data.append((upload.filename, content))

        if not file_data:
            return {"ok": False, "message": "All uploaded files were empty."}

        ok, message = index_pdfs(file_data)
        return {"ok": ok, "message": message}
    except Exception as exc:
        return {"ok": False, "message": f"Index failed: {exc}"}


@app.post("/chat")
def chat(q: Question):
    try:
        answer, sources = ask(q.question)
        return {"ok": True, "answer": answer, "sources": sources}
    except Exception as exc:
        return {"ok": False, "answer": f"Chat failed: {exc}", "sources": []}
