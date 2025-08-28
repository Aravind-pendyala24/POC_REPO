import os
import time
import jwt  # pip install PyJWT
import requests


class GitHubAppAuth:
    def __init__(self, app_id=None, installation_id=None, private_key_path=None):
        self.app_id = app_id or os.getenv("GITHUB_APP_ID")
        self.installation_id = installation_id or os.getenv("GITHUB_INSTALLATION_ID")
        self.private_key_path = private_key_path or os.getenv("GITHUB_PRIVATE_KEY_PATH", "private-key.pem")

        if not (self.app_id and self.installation_id and self.private_key_path):
            raise ValueError("Missing GitHub App configuration (APP_ID, INSTALLATION_ID, PRIVATE_KEY_PATH)")

    def _generate_jwt(self):
        """Generate a signed JWT for GitHub App authentication"""
        with open(self.private_key_path, "r") as f:
            private_key = f.read()

        now = int(time.time())
        payload = {
            "iat": now,                  # issued at time
            "exp": now + (10 * 60),      # JWT expiration (10 minutes)
            "iss": self.app_id           # GitHub App's identifier
        }

        return jwt.encode(payload, private_key, algorithm="RS256")

    def get_installation_token(self):
        """Get a short-lived installation token for API calls"""
        jwt_token = self._generate_jwt()
        url = f"https://api.github.com/app/installations/{self.installation_id}/access_tokens"
        headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Accept": "application/vnd.github+json"
        }

        response = requests.post(url, headers=headers)
        if response.status_code != 201:
            raise Exception(f"Failed to get installation token: {response.status_code}, {response.text}")

        token = response.json()["token"]
        return token
###### In app.py ##################

from github_app_auth import GitHubAppAuth

def trigger_github_actions(app_name, version, environment, job_type, branch):
    repo_name = APP_MAP.get(app_name)
    if not repo_name:
        raise Exception(f"No repo found for application {app_name}")

    workflow_file = WORKFLOW_MAP.get(job_type)
    if not workflow_file:
        raise Exception(f"No workflow mapped for job type {job_type}")

    gh = GitHubAppAuth()
    token = gh.get_installation_token()

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
        return f"✅ Workflow {workflow_file} triggered for {app_name} ({version}) in {environment} on {branch}"
    elif r.status_code == 404:
        raise Exception(f"❌ Workflow file {workflow_file} not found in repo {repo_name}")
    elif r.status_code == 422:
        raise Exception(f"❌ Invalid inputs or branch '{branch}' not found in repo {repo_name}: {r.text}")
    else:
        raise Exception(f"❌ GitHub API failed: {r.status_code}, {r.text}")
