import json
import os

METADATA_FILE = "embeddings_metadata.json"

def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_metadata(metadata):
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f)

def get_content_hash(content):
    import hashlib
    return hashlib.sha256(content.encode()).hexdigest()
