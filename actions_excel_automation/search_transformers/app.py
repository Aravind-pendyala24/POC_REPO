import os
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load env vars
load_dotenv()

app = Flask(__name__)
app.secret_key = "supersecret"

# ---------- Config ----------
CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")  # e.g., https://yourcompany.atlassian.net/wiki
ATLASSIAN_EMAIL = os.getenv("ATLASSIAN_EMAIL")
ATLASSIAN_TOKEN = os.getenv("ATLASSIAN_API_TOKEN")
CONFLUENCE_PAGE_IDS = os.getenv("CONFLUENCE_PAGE_IDS", "").split(",")  # comma-separated
auth = (ATLASSIAN_EMAIL, ATLASSIAN_TOKEN)

# Azure OpenAI configs
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")  # e.g., https://my-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")  # your deployed model name (e.g. "gpt-4o-mini")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")

# Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# ---------- Embedding Model (local for POC) ----------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- Fetch Confluence Page ----------
def fetch_confluence_page(page_id: str):
    """Fetches a Confluence page by page ID and returns plain text."""
    url = f"{CONFLUENCE_URL}/rest/api/content/{page_id}"
    params = {"expand": "body.storage"}
    try:
        resp = requests.get(url, params=params, auth=auth, timeout=10)
        resp.raise_for_status()
        page = resp.json()
        title = page["title"]
        html = page["body"]["storage"]["value"]
        text = BeautifulSoup(html, "html.parser").get_text()
        return {"text": f"{title}\n\n{text}", "source": f"Confluence {title}"}
    except Exception as e:
        return {"text": f"⚠️ Error fetching page {page_id}: {e}", "source": "Error"}

# ---------- Build Vector Store (FAISS) ----------
def build_vectorstore():
    docs = [fetch_confluence_page(pid) for pid in CONFLUENCE_PAGE_IDS if pid]
    texts = [d["text"] for d in docs]
    if not texts:
        return [], None

    embeddings = embedder.encode(texts, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return docs, index

DOCS, INDEX = build_vectorstore()

def retrieve_docs(query, k=3):
    """Retrieve top-k docs using FAISS embeddings."""
    if INDEX is None:
        return []
    query_vec = embedder.encode([query], convert_to_numpy=True)
    D, I = INDEX.search(query_vec, k)
    return [DOCS[i] for i in I[0]]

# ---------- Answer Query ----------
def answer_query(query: str):
    docs = retrieve_docs(query)
    context = "\n\n".join([d["text"] for d in docs])

    prompt = f"""
    You are a helpful assistant. Use the following Confluence documentation to answer the query.

    Context:
    {context}

    Question: {query}
    Answer:
    """

    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,  # Deployment name in Azure
            messages=[
                {"role": "system", "content": "You are a documentation assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400
        )
        answer = response.choices[0].message.content
        return answer, docs
    except Exception as e:
        return f"⚠️ Error while querying Azure OpenAI: {e}", []

# ---------- Routes ----------
@app.route("/", methods=["GET", "POST"])
def search():
    if request.method == "POST":
        query = request.form.get("query", "")
        answer, sources = answer_query(query)
        return render_template("qa.html", query=query, answer=answer, sources=sources)
    return render_template("qa.html")

if __name__ == "__main__":
    app.run(debug=True)
