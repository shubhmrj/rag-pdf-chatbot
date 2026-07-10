from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from rag import ask, index_pdfs, is_ready

app = FastAPI(title="Medical Research Assistant API")


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

        payload: list[tuple[str, bytes]] = []
        for upload in files:
            content = await upload.read()
            if content and upload.filename:
                payload.append((upload.filename, content))

        if not payload:
            return {"ok": False, "message": "All uploaded files were empty."}

        ok, message = index_pdfs(payload)
        return {"ok": ok, "message": message}
    except Exception as exc:
        return {"ok": False, "message": f"Index failed: {exc}"}


@app.post("/chat")
def chat(question: Question):
    try:
        answer, sources = ask(question.question)
        return {"ok": True, "answer": answer, "sources": sources}
    except Exception as exc:
        return {"ok": False, "answer": f"Chat failed: {exc}", "sources": []}
