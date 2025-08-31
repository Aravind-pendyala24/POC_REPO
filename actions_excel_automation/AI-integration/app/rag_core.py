import os, json
import faiss, numpy as np
from openai import OpenAI

TOP_K = int(os.getenv("TOP_K", "5"))
client = OpenAI()

# Load index + metadata once
faiss_index = faiss.read_index("index_store/faiss.index")
meta = json.load(open("index_store/meta.json", "r", encoding="utf-8"))

def embed_query(q: str):
    model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    v = client.embeddings.create(model=model, input=q).data[0].embedding
    v = np.array(v, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(v)
    return v

def retrieve(q: str):
    v = embed_query(q)
    D, I = faiss_index.search(v, TOP_K)
    ctxs = []
    for idx in I[0]:
        m = meta[idx]
        ctxs.append({"text": m.get("text", ""), "title": m.get("title"), "url": m.get("url"), "source": m.get("source")})
    return ctxs

def answer(q: str):
    # For compactness, reload the raw text of chunks: store them into meta.json if needed
    # Here, put the text into meta if desired. Alternatively, store a parallel chunks.json.
    # Minimal version: re-embed shows meta has no 'text'. Let's store text in meta during chunking.
    # If not stored, modify chunk_and_embed.py to include 'text' in meta.
    # Below assumes 'text' exists in meta.
    ctxs = retrieve(q)
    context = "\n\n".join([f"Title: {c.get('title')}\nURL: {c.get('url')}\n{c.get('text')}" for c in ctxs])

    messages = [
        {"role": "system", "content": "You are a documentation assistant. Answer strictly from the provided context. If unsure, say you don't know."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{q}"}
    ]
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(model=model, messages=messages)
    return resp.choices[0].message.content, ctxs
