import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# === Load & Index Documents ===
def load_docs(path="docs"):
    docs = []
    for file in os.listdir(path):
        full_path = os.path.join(path, file)
        if file.endswith(".pdf"):
            docs.extend(PyPDFLoader(full_path).load())
        elif file.endswith(".docx"):
            docs.extend(Docx2txtLoader(full_path).load())
        elif file.endswith(".txt"):
            docs.extend(TextLoader(full_path).load())
    return docs

def build_vectorstore():
    docs = load_docs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(chunks, embeddings)

# Create vectorstore (cache locally)
VECTORSTORE = build_vectorstore()

# === RAG Query ===
def answer(query: str):
    retriever = VECTORSTORE.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    context_texts = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
    You are a helpful assistant. 
    Use the following context from Jira & Confluence docs to answer:
    Context:
    {context_texts}

    Question: {query}

    Answer clearly and concisely.
    """

    resp = llm.predict(prompt)
    return resp, docs
