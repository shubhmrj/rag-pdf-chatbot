from fastapi import FastAPI
from pydantic import BaseModel
from chain import qa_chain

app = FastAPI()


class Query(BaseModel):
    question: str


@app.post("/chat")
async def chat(q: Query):
    result = qa_chain({"query": q.question})
    sources = [
        f"{d.metadata.get('source', '?')} p.{d.metadata.get('page', '?')}"
        for d in result.get("source_documents", [])
    ]
    return {
        "answer": result["result"],
        "sources": list(set(sources)),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)