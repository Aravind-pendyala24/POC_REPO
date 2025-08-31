import os, time, json
from atlassian import Confluence, Jira
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("ATLASSIAN_BASE_URL")
EMAIL    = os.getenv("ATLASSIAN_EMAIL")
TOKEN    = os.getenv("ATLASSIAN_API_TOKEN")
SPACES   = (os.getenv("CONFLUENCE_SPACE_KEYS") or "").split(",")
JQL      = os.getenv("JIRA_JQL") or "project = ABC"

OUT_DIR  = "ingest_out"
os.makedirs(OUT_DIR, exist_ok=True)

def fetch_confluence():
    conf = Confluence(url=BASE_URL, username=EMAIL, password=TOKEN, cloud=True)
    docs = []
    for space in SPACES:
        cql = f"space = {space} AND type = page"
        start = 0
        while True:
            res = conf.cql(cql, start=start, limit=50, expand="content.body.storage,version")
            results = res.get("results", [])
            for r in results:
                page = r["content"]
                title = page["title"]
                body_html = page["body"]["storage"]["value"]
                # Convert HTML → plain text (simplest: strip tags)
                text = strip_html(body_html)
                url = BASE_URL + page["_links"]["webui"]
                docs.append({"source": "confluence", "space": space, "title": title, "url": url, "text": text})
            if len(results) < 50:
                break
            start += 50
            time.sleep(0.2)
    with open(f"{OUT_DIR}/confluence.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)
    print(f"Confluence: {len(docs)} docs")

def strip_html(html):
    # Very simple stripper. For better quality, use BeautifulSoup.
    import re
    text = re.sub(r"<[^>]+>", " ", html)
    return " ".join(text.split())

def fetch_jira():
    jira = Jira(url=BASE_URL, username=EMAIL, password=TOKEN, cloud=True)
    start = 0
    docs = []
    while True:
        page = jira.jql(JQL, start=start, limit=50, fields="summary,description")
        issues = page.get("issues", [])
        for issue in issues:
            key = issue["key"]
            fields = issue["fields"]
            title = fields.get("summary") or key
            desc  = fields.get("description") or ""
            url   = f"{BASE_URL}/browse/{key}"
            text  = strip_html(desc) if desc else title
            docs.append({"source": "jira", "key": key, "title": title, "url": url, "text": text})
        if len(issues) < 50:
            break
        start += 50
        time.sleep(0.2)
    with open(f"{OUT_DIR}/jira.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)
    print(f"Jira: {len(docs)} docs")

if __name__ == "__main__":
    fetch_confluence()
    fetch_jira()
