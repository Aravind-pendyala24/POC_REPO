import time

def trigger_github_workflow(app_name, job_type, branch, environment):
    repo_name = APP_MAP.get(app_name)
    if not repo_name:
        raise Exception(f"No repo found for application {app_name}")

    workflow_file = WORKFLOW_MAP.get(job_type.lower())
    if not workflow_file:
        raise Exception(f"No workflow mapped for job type {job_type}")

    # Step 1: Trigger workflow
    url = f"https://api.github.com/repos/{GITHUB_OWNER}/{repo_name}/actions/workflows/{workflow_file}/dispatches"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    payload = {
        "ref": branch,
        "inputs": {
            "application": app_name,
            "environment": environment,
            "tool": "shell",
            "operation": job_type.lower()
        }
    }

    r = requests.post(url, headers=headers, json=payload)
    if r.status_code not in (200, 201, 204):
        raise Exception(f"GitHub API failed: {r.status_code}, {r.text}")

    # Step 2: Poll workflow runs to find the latest run ID
    runs_url = f"https://api.github.com/repos/{GITHUB_OWNER}/{repo_name}/actions/runs"
    run_id = None
    for _ in range(5):  # Try 5 times with a short delay
        time.sleep(2)
        runs_resp = requests.get(runs_url, headers=headers, params={"branch": branch})
        if runs_resp.status_code != 200:
            continue

        runs_data = runs_resp.json().get("workflow_runs", [])
        if runs_data:
            # Pick the latest run
            run_id = runs_data[0]["id"]
            break

    if not run_id:
        raise Exception("Triggered workflow, but could not fetch run ID.")

    run_url = f"https://github.com/{GITHUB_OWNER}/{repo_name}/actions/runs/{run_id}"
    return run_url, repo_name

def write_audit_excel(app_name, job_type, branch, environment, repo_name, run_url, status):
    filename = "audit_log.xlsx"
    lock = FileLock(f"{filename}.lock")

    with lock:
        try:
            wb = load_workbook(filename)
            ws = wb.active
        except Exception:
            wb = Workbook()
            ws = wb.active
            ws.append([
                "Timestamp", "Application", "JobType", "Branch",
                "Environment", "RepoName", "WorkflowURL", "Status"
            ])

        ws.append([
            datetime.now().isoformat(),
            app_name, job_type, branch, environment,
            repo_name, run_url, status
        ])
        wb.save(filename)


def handle_query(query: str):
    m = re.match(
        r"(Deploy|Test)\s+(\w+)\s+from branch\s+([\w\/-]+)\s+in\s+(\w+)\s+environment",
        query, re.IGNORECASE
    )

    # m = re.match(
    #     r"(Deploy|Test)\s+(\w+)\s+from branch\s+([\w\/-]+)\s+in\s+(\w+)\s+environment",
    #     query, re.IGNORECASE
    # )

    if m:
        job_type, app_name, branch, environment = m.groups()
        job_type = job_type.lower()

        try:
            run_url, repo_name = trigger_github_workflow(app_name, job_type, branch, environment)
            status = "Success: Workflow triggered"
            write_audit_excel(app_name, job_type, branch, environment, repo_name, run_url, status)
            return f"✅ {job_type.capitalize()} workflow for {app_name} in {environment} triggered successfully. <a href='{run_url}' target='_blank'>View Run</a>"
        except Exception as e:
            repo_name = APP_MAP.get(app_name, "N/A")
            status = f"Failed: {str(e)}"
            write_audit_excel(app_name, job_type, branch, environment, repo_name, "N/A", status)
            return f"❌ Failed to trigger workflow for {app_name} in {environment}: {str(e)}"

    # Otherwise → RAG fallback
    answer, _ = answer_query(query)
    return answer
