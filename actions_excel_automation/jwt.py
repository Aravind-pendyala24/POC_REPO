import os
import jwt   # PyJWT
import time
import requests

from dotenv import load_dotenv
load_dotenv()

GITHUB_APP_ID = os.getenv("GITHUB_APP_ID")  # numeric App ID
GITHUB_INSTALLATION_ID = os.getenv("GITHUB_INSTALLATION_ID")  # numeric installation ID
GITHUB_OWNER = os.getenv("GITHUB_OWNER")
GITHUB_PRIVATE_KEY_PATH = os.getenv("GITHUB_PRIVATE_KEY_PATH", "private-key.pem")

def generate_jwt():
    """Generate a JWT signed with the GitHub App's private key"""
    with open(GITHUB_PRIVATE_KEY_PATH, "r") as f:
        private_key = f.read()

    now = int(time.time())
    payload = {
        "iat": now,                  # issued at
        "exp": now + (10 * 60),      # JWT expiration (10 minutes max)
        "iss": GITHUB_APP_ID         # GitHub App ID
    }

    return jwt.encode(payload, private_key, algorithm="RS256")
    

def get_installation_token():
    """Exchange the JWT for an installation token"""
    jwt_token = generate_jwt()
    url = f"https://api.github.com/app/installations/{GITHUB_INSTALLATION_ID}/access_tokens"
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Accept": "application/vnd.github+json"
    }

    r = requests.post(url, headers=headers)
    if r.status_code != 201:
        raise Exception(f"Failed to get installation token: {r.status_code} {r.text}")

    return r.json()["token"]


def trigger_github_actions(app_name, version, environment, job_type, branch):
    repo_name = APP_MAP.get(app_name)
    if not repo_name:
        raise Exception(f"No repo found for application {app_name}")

    workflow_file = WORKFLOW_MAP.get(job_type)
    if not workflow_file:
        raise Exception(f"No workflow mapped for job type {job_type}")

    token = get_installation_token()

    url = f"https://api.github.com/repos/{GITHUB_OWNER}/{repo_name}/actions/workflows/{workflow_file}/dispatches"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }
    payload = {
        "ref": branch,
        "inputs": {
            "application": app_name,
            "version": version,
            "environment": environment
        }
    }

    r = requests.post(url, headers=headers, json=payload)
    if r.status_code in (200, 201, 204):
        return f"Workflow {workflow_file} triggered for {app_name} ({version}) in {environment} on {branch}"
    elif r.status_code == 404:
        raise Exception(f"Workflow file {workflow_file} not found in repo {repo_name}")
    elif r.status_code == 422:
        raise Exception(f"Invalid inputs or branch '{branch}' not found in repo {repo_name}: {r.text}")
    else:
        raise Exception(f"GitHub API failed: {r.status_code}, {r.text}")
