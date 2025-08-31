import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.rag_core import answer

app = FastAPI(title="Doc Assistant (RAG)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True,
)

class Query(BaseModel):
    query: str

@app.post("/ask")
def ask(q: Query):
    if "deployment request" in q.query.lower():
        # In your UI, redirect to deployment page instead of answering
        return {"redirect": "/deploy", "message": "Deployment request detected."}
    try:
        ans, ctx = answer(q.query)
        return {"answer": ans, "contexts": ctx}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
