import os
import json
import hashlib
from typing import List, Dict

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import faiss
import numpy as np
from openai import OpenAI

# Load env vars
load_dotenv()

# ---------- Config ----------
CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")  # e.g., https://yourcompany.atlassian.net/wiki
ATLASSIAN_EMAIL = os.getenv("ATLASSIAN_EMAIL")
ATLASSIAN_TOKEN = os.getenv("ATLASSIAN_API_TOKEN")
CONFLUENCE_PAGE_IDS = os.getenv("CONFLUENCE_PAGE_IDS", "").split(",")

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")

# Azure client
client = OpenAI(api_key=AZURE_OPENAI_KEY)

# ---------- FastAPI ----------
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ---------- Persistence ----------
INDEX_PATH = "faiss.index"
META_PATH = "page_metadata.json"

if os.path.exists(META_PATH):
    with open(META_PATH, "r", encoding="utf-8") as f:
        PAGE_METADATA = json.load(f)
else:
    PAGE_METADATA = {}

if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
else:
    index = None

DOCS: List[Dict] = []


# ---------- Utils ----------
def compute_hash(text: str) -> str:
    """Compute SHA256 hash of text for change detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch_confluence_page(page_id: str) -> Dict:
    """Fetch Confluence page by ID with metadata."""
    url = f"{CONFLUENCE_URL}/rest/api/content/{page_id}"
    params = {"expand": "body.storage,version"}
    auth = (ATLASSIAN_EMAIL, ATLASSIAN_TOKEN)

    resp = requests.get(url, params=params, auth=auth, timeout=10)
    resp.raise_for_status()
    page = resp.json()

    title = page["title"]
    html = page["body"]["storage"]["value"]
    text = BeautifulSoup(html, "html.parser").get_text()
    last_updated = page["version"]["when"]

    return {
        "id": page_id,
        "title": title,
        "text": text,
        "last_updated": last_updated,
    }


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors for cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def embed_texts(texts: List[str]) -> np.ndarray:
    """Get embeddings from Azure OpenAI and normalize for cosine similarity."""
    response = client.embeddings.create(
        model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        input=texts,
        extra_query={"api-version": AZURE_OPENAI_API_VERSION},
        base_url=f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_EMBEDDING_DEPLOYMENT}"
    )
    vectors = [np.array(e.embedding, dtype=np.float32) for e in response.data]
    return normalize(np.vstack(vectors))


def refresh_vectorstore():
    """Fetch Confluence pages, re-embed only changed pages, update FAISS + metadata."""
    global index, DOCS, PAGE_METADATA

    new_docs, new_embeddings = [], []

    for pid in CONFLUENCE_PAGE_IDS:
        if not pid.strip():
            continue

        page = fetch_confluence_page(pid)
        text_hash = compute_hash(page["text"])

        meta = PAGE_METADATA.get(pid, {})
        if meta.get("hash") == text_hash:
            continue  # no change

        # embed updated page
        embedding = embed_texts([page["text"]])
        new_docs.append(page)
        new_embeddings.append(embedding)

        PAGE_METADATA[pid] = {
            "title": page["title"],
            "hash": text_hash,
            "last_updated": page["last_updated"],
        }

    if index is None and new_embeddings:
        dim = new_embeddings[0].shape[1]
        index = faiss.IndexFlatIP(dim)  # cosine similarity via inner product

    if new_embeddings:
        stacked = np.vstack(new_embeddings)
        index.add(stacked)
        DOCS.extend(new_docs)

    if index is not None:
        faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(PAGE_METADATA, f, indent=2, ensure_ascii=False)


def retrieve(query: str, k=3):
    """Retrieve top-k docs using cosine similarity."""
    if index is None:
        return []
    q_emb = embed_texts([query])
    D, I = index.search(q_emb, k)
    return [DOCS[i] for i in I[0] if i < len(DOCS)]


def answer_query(query: str):
    """Answer a query with Azure OpenAI LLM using retrieved docs as context."""
    results = retrieve(query)
    context = "\n\n".join([f"Title: {d['title']}\n{d['text']}" for d in results])

    prompt = f"""
    You are a helpful assistant. Use the following Confluence documentation context to answer the query.

    Context:
    {context}

    Question: {query}
    Answer:
    """

    response = client.chat.completions.create(
        model=AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a documentation assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=400,
        extra_query={"api-version": AZURE_OPENAI_API_VERSION},
        base_url=f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_CHAT_DEPLOYMENT}"
    )

    answer = response.choices[0].message.content
    return answer, results


# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("qa.html", {"request": request})


@app.post("/", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):
    #refresh_vectorstore()
    answer, results = answer_query(query)
    return templates.TemplateResponse(
        "qa.html",
        {"request": request, "query": query, "answer": answer, "results": results}
    )

@app.post("/admin/refresh")
async def admin_refresh():
    refresh_vectorstore()
    return {"status": "Vector store refreshed"}
