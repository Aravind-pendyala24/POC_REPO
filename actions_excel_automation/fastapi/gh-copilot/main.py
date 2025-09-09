from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from confluence_utils import fetch_confluence_pages, get_page_content, get_page_metadata
from embedding_utils import embed_content, get_faiss_index, store_embeddings, search_similar
from llm_utils import get_llm_answer
from metadata_utils import load_metadata, save_metadata, get_content_hash

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static/index.html") as f:
        return HTMLResponse(f.read())

@app.post("/chat")
async def chat(query: str = Form(...)):
    # Fetch latest confluence pages
    pages = fetch_confluence_pages()
    metadata = load_metadata()

    # Check and embed new/updated pages
    new_metadata = {}
    for page in pages:
        page_id = page['id']
        content = get_page_content(page_id)
        content_hash = get_content_hash(content)
        meta = get_page_metadata(page)
        meta['hash'] = content_hash

        if page_id not in metadata or metadata[page_id]['hash'] != content_hash:
            # New or updated, so embed and store
            embedding = embed_content(content)
            store_embeddings(page_id, embedding, meta)
            new_metadata[page_id] = meta
        else:
            new_metadata[page_id] = metadata[page_id]

    save_metadata(new_metadata)
    
    # Search relevant content using FAISS
    top_pages = search_similar(query, k=3)
    context = "\n\n".join([p['content'] for p in top_pages])

    # Get answer from LLM
    answer = get_llm_answer(query, context)
    return JSONResponse({"answer": answer})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
