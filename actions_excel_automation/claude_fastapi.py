import os
import json
import hashlib
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from openai import AzureOpenAI
import uvicorn

# Configuration
class Config:
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4")
    
    CONFLUENCE_BASE_URL = os.getenv("CONFLUENCE_BASE_URL", "")
    CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME", "")
    CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN", "")
    
    VECTOR_STORE_PATH = "vector_store"
    METADATA_PATH = "metadata.json"
    EMBEDDING_DIM = 1536  # Default for text-embedding-3-small

# Pydantic models
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

class PageMetadata(BaseModel):
    page_id: str
    title: str
    url: str
    last_modified: str
    content_hash: str
    space_key: str
    version: int

# Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
    api_key=Config.AZURE_OPENAI_API_KEY,
    api_version=Config.AZURE_OPENAI_API_VERSION
)

class ConfluenceClient:
    def __init__(self):
        self.base_url = Config.CONFLUENCE_BASE_URL
        self.auth = (Config.CONFLUENCE_USERNAME, Config.CONFLUENCE_API_TOKEN)
        
    def get_pages(self, space_key: str = None, limit: int = 100) -> List[Dict]:
        """Fetch pages from Confluence"""
        url = f"{self.base_url}/rest/api/content"
        params = {
            "limit": limit,
            "expand": "body.storage,version,space"
        }
        if space_key:
            params["spaceKey"] = space_key
            
        response = requests.get(url, auth=self.auth, params=params)
        response.raise_for_status()
        return response.json().get("results", [])
    
    def get_page_content(self, page_id: str) -> Dict:
        """Get detailed page content"""
        url = f"{self.base_url}/rest/api/content/{page_id}"
        params = {"expand": "body.storage,version,space"}
        
        response = requests.get(url, auth=self.auth, params=params)
        response.raise_for_status()
        return response.json()

class VectorStore:
    def __init__(self):
        self.index = None
        self.metadata = []
        self.embedding_dim = Config.EMBEDDING_DIM
        self.index_path = f"{Config.VECTOR_STORE_PATH}/faiss_index.bin"
        self.metadata_path = Config.METADATA_PATH
        
        # Create directories if they don't exist
        Path(Config.VECTOR_STORE_PATH).mkdir(exist_ok=True)
        
        self.load_index()
        self.load_metadata()
    
    def load_index(self):
        """Load existing FAISS index or create new one"""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
    
    def load_metadata(self):
        """Load existing metadata"""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = []
    
    def save_index(self):
        """Save FAISS index to disk"""
        faiss.write_index(self.index, self.index_path)
    
    def save_metadata(self):
        """Save metadata to disk"""
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def add_documents(self, embeddings: np.ndarray, metadata_list: List[Dict]):
        """Add documents to vector store"""
        self.index.add(embeddings)
        self.metadata.extend(metadata_list)
        self.save_index()
        self.save_metadata()
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        if self.index.ntotal == 0:
            return []
        
        scores, indices = self.index.search(query_embedding, k)
        results = []
        
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)
        
        return results
    
    def update_document(self, doc_id: str, embedding: np.ndarray, metadata: Dict):
        """Update existing document or add new one"""
        # For simplicity, we'll rebuild the index when updating
        # In production, you might want a more sophisticated approach
        updated = False
        for i, meta in enumerate(self.metadata):
            if meta.get('page_id') == doc_id:
                self.metadata[i] = metadata
                updated = True
                break
        
        if not updated:
            # Add new document
            self.index.add(embedding)
            self.metadata.append(metadata)
        else:
            # Rebuild index (simplified approach)
            all_embeddings = []
            for meta in self.metadata:
                if 'embedding' in meta:
                    all_embeddings.append(meta['embedding'])
            
            if all_embeddings:
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                embeddings_array = np.array(all_embeddings).astype('float32')
                self.index.add(embeddings_array)
        
        self.save_index()
        self.save_metadata()

class RAGChatBot:
    def __init__(self):
        self.confluence_client = ConfluenceClient()
        self.vector_store = VectorStore()
        
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using Azure OpenAI"""
        response = client.embeddings.create(
            input=text,
            model=Config.EMBEDDING_MODEL
        )
        return response.data[0].embedding
    
    def extract_text_from_html(self, html_content: str) -> str:
        """Extract plain text from HTML content"""
        import re
        # Simple HTML tag removal (you might want to use BeautifulSoup for better parsing)
        text = re.sub(r'<[^>]+>', '', html_content)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def generate_content_hash(self, content: str) -> str:
        """Generate hash for content to detect changes"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    async def sync_confluence_data(self, space_key: str = None):
        """Sync Confluence data with vector store"""
        print("Starting Confluence data sync...")
        
        try:
            pages = self.confluence_client.get_pages(space_key=space_key)
            
            for page in pages:
                page_id = page['id']
                title = page['title']
                
                # Get detailed content
                detailed_page = self.confluence_client.get_page_content(page_id)
                content = detailed_page.get('body', {}).get('storage', {}).get('value', '')
                
                # Extract text content
                text_content = self.extract_text_from_html(content)
                content_hash = self.generate_content_hash(text_content)
                
                # Check if page exists and has changed
                existing_metadata = next(
                    (meta for meta in self.vector_store.metadata if meta.get('page_id') == page_id),
                    None
                )
                
                if not existing_metadata or existing_metadata.get('content_hash') != content_hash:
                    print(f"Processing page: {title}")
                    
                    # Generate embedding
                    embedding = self.get_embedding(text_content)
                    
                    # Prepare metadata
                    metadata = {
                        'page_id': page_id,
                        'title': title,
                        'url': f"{Config.CONFLUENCE_BASE_URL}/pages/viewpage.action?pageId={page_id}",
                        'last_modified': detailed_page.get('version', {}).get('when', ''),
                        'content_hash': content_hash,
                        'space_key': detailed_page.get('space', {}).get('key', ''),
                        'version': detailed_page.get('version', {}).get('number', 0),
                        'content': text_content[:1000],  # Store first 1000 chars for display
                        'embedding': embedding
                    }
                    
                    # Update vector store
                    embedding_array = np.array([embedding]).astype('float32')
                    if not existing_metadata:
                        self.vector_store.add_documents(embedding_array, [metadata])
                    else:
                        self.vector_store.update_document(page_id, embedding_array, metadata)
                
        except Exception as e:
            print(f"Error syncing Confluence data: {e}")
            raise
        
        print("Confluence data sync completed!")
    
    def search_relevant_content(self, query: str, k: int = 5) -> List[Dict]:
        """Search for relevant content based on query"""
        query_embedding = self.get_embedding(query)
        query_array = np.array([query_embedding]).astype('float32')
        
        return self.vector_store.search(query_array, k=k)
    
    def generate_answer(self, query: str, context_docs: List[Dict]) -> str:
        """Generate answer using Azure OpenAI with context"""
        context = "\n\n".join([
            f"Title: {doc['title']}\nContent: {doc.get('content', '')}"
            for doc in context_docs
        ])
        
        system_prompt = """You are a helpful assistant that answers questions based on the provided context from Confluence pages. 
        Use the context to provide accurate and relevant answers. If the context doesn't contain enough information to answer the question, 
        say so and suggest what additional information might be needed."""
        
        user_prompt = f"""Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above."""
        
        response = client.chat.completions.create(
            model=Config.CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.choices[0].message.content

# Initialize the RAG chatbot
rag_bot = RAGChatBot()

# FastAPI app
app = FastAPI(title="RAG Chat Application", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    """Serve the chat interface"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG Chat Application</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                text-align: center;
            }
            .header h1 {
                margin: 0;
                font-size: 24px;
            }
            .chat-container {
                height: 500px;
                overflow-y: auto;
                padding: 20px;
                background: #f8f9fa;
            }
            .message {
                margin-bottom: 15px;
                padding: 12px 16px;
                border-radius: 18px;
                max-width: 80%;
                word-wrap: break-word;
            }
            .user-message {
                background: #007bff;
                color: white;
                margin-left: auto;
                text-align: right;
            }
            .bot-message {
                background: white;
                border: 1px solid #dee2e6;
                margin-right: auto;
            }
            .sources {
                background: #e9ecef;
                border-radius: 8px;
                padding: 10px;
                margin-top: 10px;
                font-size: 12px;
            }
            .source-item {
                margin: 5px 0;
                padding: 5px;
                background: white;
                border-radius: 4px;
            }
            .input-container {
                padding: 20px;
                background: white;
                border-top: 1px solid #dee2e6;
                display: flex;
                gap: 10px;
            }
            .query-input {
                flex: 1;
                padding: 12px;
                border: 2px solid #dee2e6;
                border-radius: 25px;
                font-size: 14px;
                outline: none;
            }
            .query-input:focus {
                border-color: #007bff;
            }
            .search-btn {
                padding: 12px 24px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 600;
                transition: all 0.3s;
            }
            .search-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            .search-btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
                color: #666;
            }
            .sync-container {
                padding: 15px 20px;
                background: #f8f9fa;
                border-top: 1px solid #dee2e6;
                text-align: center;
            }
            .sync-btn {
                padding: 8px 16px;
                background: #28a745;
                color: white;
                border: none;
                border-radius: 20px;
                cursor: pointer;
                font-size: 12px;
                margin: 0 5px;
            }
            .sync-btn:hover {
                background: #218838;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 RAG Chat Assistant</h1>
                <p>Ask questions about your Confluence content</p>
            </div>
            
            <div class="chat-container" id="chatContainer">
                <div class="message bot-message">
                    👋 Hello! I'm your RAG assistant. I can help you find information from your Confluence pages. 
                    Make sure to sync your data first, then ask me anything!
                </div>
            </div>
            
            <div class="loading" id="loading">
                🔄 Processing your request...
            </div>
            
            <div class="input-container">
                <input type="text" id="queryInput" class="query-input" placeholder="Ask me anything about your Confluence content..." onkeypress="handleKeyPress(event)">
                <button onclick="sendQuery()" class="search-btn" id="searchBtn">Search</button>
            </div>
            
            <div class="sync-container">
                <button onclick="syncData()" class="sync-btn" id="syncBtn">🔄 Sync Confluence Data</button>
                <span style="font-size: 12px; color: #666; margin-left: 10px;">Sync your latest Confluence content</span>
            </div>
        </div>

        <script>
            async function sendQuery() {
                const queryInput = document.getElementById('queryInput');
                const query = queryInput.value.trim();
                
                if (!query) return;
                
                const chatContainer = document.getElementById('chatContainer');
                const searchBtn = document.getElementById('searchBtn');
                const loading = document.getElementById('loading');
                
                // Add user message
                addMessage(query, 'user');
                queryInput.value = '';
                
                // Show loading
                searchBtn.disabled = true;
                loading.style.display = 'block';
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ query: query })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        addMessage(data.answer, 'bot', data.sources);
                    } else {
                        addMessage(`Error: ${data.detail}`, 'bot');
                    }
                } catch (error) {
                    addMessage('Sorry, there was an error processing your request.', 'bot');
                    console.error('Error:', error);
                } finally {
                    searchBtn.disabled = false;
                    loading.style.display = 'none';
                }
            }
            
            function addMessage(content, type, sources = []) {
                const chatContainer = document.getElementById('chatContainer');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${type}-message`;
                
                let html = `<div>${content}</div>`;
                
                if (sources && sources.length > 0) {
                    html += '<div class="sources"><strong>📚 Sources:</strong>';
                    sources.forEach(source => {
                        html += `<div class="source-item">
                            <strong>${source.title}</strong><br>
                            <a href="${source.url}" target="_blank">View Page</a> | 
                            Score: ${(source.similarity_score || 0).toFixed(3)}
                        </div>`;
                    });
                    html += '</div>';
                }
                
                messageDiv.innerHTML = html;
                chatContainer.appendChild(messageDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
            
            async function syncData() {
                const syncBtn = document.getElementById('syncBtn');
                syncBtn.disabled = true;
                syncBtn.textContent = '🔄 Syncing...';
                
                try {
                    const response = await fetch('/sync-confluence', {
                        method: 'POST'
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        addMessage('✅ Confluence data synced successfully!', 'bot');
                    } else {
                        addMessage(`❌ Sync failed: ${data.detail}`, 'bot');
                    }
                } catch (error) {
                    addMessage('❌ Sync failed: Network error', 'bot');
                    console.error('Sync error:', error);
                } finally {
                    syncBtn.disabled = false;
                    syncBtn.textContent = '🔄 Sync Confluence Data';
                }
            }
            
            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    sendQuery();
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat requests"""
    try:
        # Search for relevant content
        relevant_docs = rag_bot.search_relevant_content(request.query, k=5)
        
        if not relevant_docs:
            return ChatResponse(
                answer="I don't have any relevant information to answer your question. Please make sure to sync your Confluence data first.",
                sources=[]
            )
        
        # Generate answer
        answer = rag_bot.generate_answer(request.query, relevant_docs)
        
        # Prepare sources
        sources = [
            {
                "title": doc["title"],
                "url": doc["url"],
                "similarity_score": doc.get("similarity_score", 0)
            }
            for doc in relevant_docs
        ]
        
        return ChatResponse(answer=answer, sources=sources)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sync-confluence")
async def sync_confluence_data(background_tasks: BackgroundTasks):
    """Sync Confluence data in the background"""
    try:
        background_tasks.add_task(rag_bot.sync_confluence_data)
        return JSONResponse({"message": "Confluence sync started"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "vector_store_size": len(rag_bot.vector_store.metadata)}

if __name__ == "__main__":
    print("Starting RAG Chat Application...")
    print("Please set the following environment variables:")
    print("- AZURE_OPENAI_ENDPOINT")
    print("- AZURE_OPENAI_API_KEY")
    print("- CONFLUENCE_BASE_URL")
    print("- CONFLUENCE_USERNAME")
    print("- CONFLUENCE_API_TOKEN")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
