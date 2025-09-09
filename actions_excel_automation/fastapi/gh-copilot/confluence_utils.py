import requests
import hashlib

CONFLUENCE_BASE_URL = "https://your-domain.atlassian.net/wiki"
CONFLUENCE_API_TOKEN = "your-api-token"
CONFLUENCE_USER = "your-email"

def fetch_confluence_pages():
    url = f"{CONFLUENCE_BASE_URL}/rest/api/content"
    headers = {"Authorization": f"Basic {CONFLUENCE_API_TOKEN}"}
    params = {"type": "page", "expand": "version"}
    resp = requests.get(url, headers=headers, params=params)
    return resp.json().get("results", [])

def get_page_content(page_id):
    url = f"{CONFLUENCE_BASE_URL}/rest/api/content/{page_id}?expand=body.storage"
    headers = {"Authorization": f"Basic {CONFLUENCE_API_TOKEN}"}
    resp = requests.get(url, headers=headers)
    return resp.json()["body"]["storage"]["value"]

def get_page_metadata(page):
    return {
        "id": page["id"],
        "title": page["title"],
        "version": page["version"]["number"],
        "url": f'{CONFLUENCE_BASE_URL}/spaces/{page["space"]["key"]}/pages/{page["id"]}'
    }

def get_content_hash(content):
    return hashlib.sha256(content.encode()).hexdigest()
