import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash
from rapidfuzz import process
from langchain_openai import AzureChatOpenAI

# Load environment
load_dotenv()

JIRA_URL = os.getenv("JIRA_URL")
CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")
ATLASSIAN_EMAIL = os.getenv("ATLASSIAN_EMAIL")
ATLASSIAN_TOKEN = os.getenv("ATLASSIAN_API_TOKEN")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

auth = (ATLASSIAN_EMAIL, ATLASSIAN_TOKEN)

app = Flask(__name__)
app.secret_key = "supersecret"

# -------- Fetch Docs --------
def fetch_jira_docs(jql="project=YOURPROJECT ORDER BY created DESC"):
    url = f"{JIRA_URL}/rest/api/3/search"
    headers = {"Accept": "application/json"}
    params = {"jql": jql, "maxResults": 5, "fields": "summary,description"}
    try:
        resp = requests.get(url, headers=headers, params=params, auth=auth, timeout=10)
        resp.raise_for_status()
        issues = resp.json().get("issues", [])
    except Exception as e:
        print(f"⚠️ Jira fetch failed: {e}")
        return []

    docs = []
    for issue in issues:
        key = issue["key"]
        summary = issue["fields"].get("summary", "")
        desc = issue["fields"].get("description", "")
        text = f"Jira Issue {key}: {summary}\n\n{desc}"
        docs.append({"text": text, "source": f"Jira {key}"})
    return docs

def fetch_confluence_docs(space="DOCS"):
    url = f"{CONFLUENCE_URL}/rest/api/content"
    params = {"spaceKey": space, "expand": "body.storage", "limit": 5}
    try:
        resp = requests.get(url, params=params, auth=auth, timeout=10)
        resp.raise_for_status()
        pages = resp.json().get("results", [])
    except Exception as e:
        print(f"⚠️ Confluence fetch failed: {e}")
        return []

    docs = []
    for page in pages:
        title = page["title"]
        html = page["body"]["storage"]["value"]
        text = BeautifulSoup(html, "html.parser").get_text()
        docs.append({"text": f"Confluence Page: {title}\n\n{text}", "source": f"Confluence {title}"})
    return docs

# -------- Prepare Docs --------
DOCS = fetch_jira_docs() + fetch_confluence_docs()

def retrieve_docs(query, k=3):
    texts = [doc["text"] for doc in DOCS]
    matches = process.extract(query, texts, limit=k)
    results = []
    for match, score, idx in matches:
        results.append(DOCS[idx])
    return results

def answer_query(query: str):
    docs = retrieve_docs(query)
    context = "\n\n".join([d["text"] for d in docs])

    llm = AzureChatOpenAI(
        deployment_name=AZURE_OPENAI_CHAT_DEPLOYMENT,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        temperature=0
    )

    prompt = f"""
    You are a helpful assistant. 
    Use the following project documentation (Jira + Confluence) to answer.

    Context:
    {context}

    Question: {query}
    """

    try:
        response = llm.predict(prompt)
        return response, docs
    except Exception as e:
        return f"⚠️ Error from Azure OpenAI: {e}", []

# -------- Routes --------
@app.route("/", methods=["GET", "POST"])
def search():
    if request.method == "POST":
        query = request.form.get("query", "")
        if "deployment request" in query.lower():
            return redirect(url_for("deployment"))
        else:
            answer, sources = answer_query(query)
            return render_template("qa.html", query=query, answer=answer, sources=sources)
    return render_template("qa.html")

@app.route("/deployment")
def deployment():
    return render_template("form3.html")  # your deployment form

if __name__ == "__main__":
    app.run(debug=True)
