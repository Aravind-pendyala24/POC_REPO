import os
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

JIRA_URL = os.getenv("JIRA_URL")
CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")
ATLASSIAN_EMAIL = os.getenv("ATLASSIAN_EMAIL")
ATLASSIAN_TOKEN = os.getenv("ATLASSIAN_API_TOKEN")

auth = (ATLASSIAN_EMAIL, ATLASSIAN_TOKEN)

# === Fetch Jira issues ===
def fetch_jira_docs(jql="project=YOURPROJECT ORDER BY created DESC"):
    url = f"{JIRA_URL}/rest/api/3/search"
    headers = {"Accept": "application/json"}
    params = {"jql": jql, "maxResults": 10, "fields": "summary,description"}
    resp = requests.get(url, headers=headers, params=params, auth=auth)
    resp.raise_for_status()
    issues = resp.json()["issues"]
    docs = []
    for issue in issues:
        key = issue["key"]
        summary = issue["fields"].get("summary", "")
        desc = issue["fields"].get("description", "")
        text = f"Jira Issue {key}: {summary}\n\n{desc}"
        docs.append({"text": text, "source": f"Jira {key}"})
    return docs

# === Fetch Confluence pages ===
def fetch_confluence_docs(space="DOCS"):
    url = f"{CONFLUENCE_URL}/rest/api/content"
    params = {"spaceKey": space, "expand": "body.storage", "limit": 10}
    resp = requests.get(url, params=params, auth=auth)
    resp.raise_for_status()
    pages = resp.json().get("results", [])
    docs = []
    for page in pages:
        title = page["title"]
        html = page["body"]["storage"]["value"]
        text = BeautifulSoup(html, "html.parser").get_text()
        docs.append({"text": f"Confluence Page: {title}\n\n{text}", "source": f"Confluence {title}"})
    return docs

# === Build Vector Store ===
def build_vectorstore():
    raw_docs = fetch_jira_docs() + fetch_confluence_docs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = []
    for d in raw_docs:
        for chunk in splitter.split_text(d["text"]):
            chunks.append({"content": chunk, "source": d["source"]})

    embeddings = AzureOpenAIEmbeddings(
        deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        model="text-embedding-3-small",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY")
    )

    vectorstore = FAISS.from_texts(
        [c["content"] for c in chunks],
        embeddings,
        metadatas=[{"source": c["source"]} for c in chunks]
    )
    return vectorstore

VECTORSTORE = build_vectorstore()

# === RAG Query ===
def answer(query: str):
    retriever = VECTORSTORE.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)

    llm = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        model="gpt-4",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        temperature=0
    )

    context_texts = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
    You are a helpful assistant. 
    Use the following context from Jira & Confluence docs to answer:
    {context_texts}

    Question: {query}
    """

    resp = llm.predict(prompt)
    return resp, docs
