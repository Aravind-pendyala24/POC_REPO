from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI
import numpy as np
import faiss
import os
import json

INDEX_FILE = "faiss.index"
METADATA_FILE = "embeddings_metadata.json"

openai_client = AzureOpenAI(
    api_key="your-azure-openai-key",
    api_base="https://your-resource.openai.azure.com/",
    api_type="azure",
    api_version="2023-07-01-preview"
)

def embed_content(content):
    response = openai_client.embeddings.create(
        input=[content],
        engine="your-embedding-model"
    )
    return np.array(response['data'][0]['embedding'], dtype='float32')

def get_faiss_index(dim=1536):
    if os.path.exists(INDEX_FILE):
        return faiss.read_index(INDEX_FILE)
    return faiss.IndexFlatL2(dim)

def store_embeddings(page_id, embedding, meta):
    index = get_faiss_index(embedding.shape[0])
    index.add(np.array([embedding]))
    faiss.write_index(index, INDEX_FILE)

    # Save metadata
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}
    metadata[page_id] = meta
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f)

def search_similar(query, k=3):
    # Embed query
    query_vec = embed_content(query)
    index = get_faiss_index(query_vec.shape[0])
    D, I = index.search(np.array([query_vec]), k)
    # Load metadata and get content for top results
    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)
    results = []
    for idx in I[0]:
        page_id = list(metadata.keys())[idx]
        content = metadata[page_id].get('content', '')
        results.append({"id": page_id, "content": content, "meta": metadata[page_id]})
    return results
