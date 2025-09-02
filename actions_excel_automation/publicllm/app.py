import os
from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from rapidfuzz import process
from openai import OpenAI
from dotenv import load_dotenv

# Load env vars
load_dotenv()

app = Flask(__name__)

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------- Load PDF Document --------
PDF_PATH = "project_docs.pdf"  # Replace with your local PDF file
DOCS = []

def load_pdf(path):
    """Extracts text from PDF and stores page-wise docs."""
    reader = PdfReader(path)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            docs.append({"text": text, "source": f"Page {i+1}"})
    return docs

DOCS = load_pdf(PDF_PATH)


# -------- Retrieval (fuzzy search) --------
def retrieve_docs(query, k=3):
    texts = [doc["text"] for doc in DOCS]
    matches = process.extract(query, texts, limit=k)
    results = []
    for match, score, idx in matches:
        results.append(DOCS[idx])
    return results


# -------- ChatGPT Answer --------
def answer_query(query: str):
    docs = retrieve_docs(query)
    context = "\n\n".join([d["text"] for d in docs])

    prompt = f"""
    You are a helpful assistant. Use the following PDF documentation to answer the query.

    Context:
    {context}

    Question: {query}
    Answer:
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-4o", "gpt-3.5-turbo"
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400
        )
        answer = response.choices[0].message.content
        return answer, docs
    except Exception as e:
        return f"⚠️ Error while querying ChatGPT API: {e}", []


# -------- Routes --------
@app.route("/", methods=["GET", "POST"])
def search():
    if request.method == "POST":
        query = request.form.get("query", "")
        answer, sources = answer_query(query)
        return render_template("qa_pdf.html", query=query, answer=answer, sources=sources)
    return render_template("qa_pdf.html")


if __name__ == "__main__":
    app.run(debug=True)
