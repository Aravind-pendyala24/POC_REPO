import os, json, pickle
import faiss
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

load_dotenv()

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
EMBED_MODEL   = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

client = OpenAI()

def load_docs():
    docs = []
    for name in ["confluence.json", "jira.json"]:
        path = f"ingest_out/{name}"
        if os.path.exists(path):
            docs.extend(json.load(open(path, "r", encoding="utf-8")))
    return docs

def chunk_docs(raw_docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks, meta = [], []
    for d in raw_docs:
        for c in splitter.split_text(d["text"]):
            chunks.append(c)
            meta.append({"title": d.get("title"), "url": d.get("url"), "source": d.get("source")})
    return chunks, meta

def embed(chunks):
    # Batch embeddings to avoid very long payloads
    vecs = []
    B = 64
    for i in range(0, len(chunks), B):
        batch = chunks[i:i+B]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vecs.extend([r.embedding for r in resp.data])
    return vecs

if __name__ == "__main__":
    raw = load_docs()
    chunks, meta = chunk_docs(raw)
    vecs = embed(chunks)
    dim = len(vecs[0])
    index = faiss.IndexFlatIP(dim)  # cosine similarity if you normalize; or use L2
    # normalize for cosine
    import numpy as np
    arr = np.array(vecs).astype("float32")
    faiss.normalize_L2(arr)
    index.add(arr)

    os.makedirs("index_store", exist_ok=True)
    faiss.write_index(index, "index_store/faiss.index")
    json.dump(meta, open("index_store/meta.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"Indexed {len(chunks)} chunks")
