def handle_query(query: str):
    normalized = query.lower().strip()
    normalized = normalized.replace("env", "environment")

    m = re.search(
        r"(deploy|test)\s+([\w\s\-]+)\s+version\s+([\w\.\-]+)\s+in\s+(\w+)\s+environment",
        query, re.IGNORECASE
    )
    if m:
        job_type, app_name, version, environment = m.groups()
        job_type = job_type.lower()
        app_name = app_name.strip()

        app_config = APP_MAP.get(app_name)
        if not app_config:
            return f"❌ No configuration found for application '{app_name}'"

        repo_name = app_config.get("repo")
        branch = app_config.get("branch")

        try:
            run_url, repo_name = trigger_github_workflow(app_name, job_type, branch, environment, version, repo_name)
            status = "Success: Workflow triggered"
            write_audit_excel(app_name, version, job_type, branch, environment, repo_name, run_url, status)
            return f"✅ {job_type.capitalize()} workflow for {app_name} (version {version}) in {environment} triggered from branch <b>{branch}</b>. <a href='{run_url}' target='_blank'>View Run</a>"
        except Exception as e:
            status = f"Failed: {str(e)}"
            write_audit_excel(app_name, version, job_type, branch, environment, repo_name, "N/A", status)
            return f"❌ Failed to trigger workflow: {str(e)}"

    # Otherwise → fall back to Q&A
    answer, _ = answer_query(query)
    return answer

def trigger_github_workflow(app_name, job_type, branch, environment, version, repo_name):
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
            "version": version,
            "environment": environment,
            "tool": "shell",
            "operation": job_type.lower()
        }
    }

    r = requests.post(url, headers=headers, json=payload)
    if r.status_code not in (200, 201, 204):
        raise Exception(f"GitHub API failed: {r.status_code}, {r.text}")

    # Step 2: Poll workflow runs
    runs_url = f"https://api.github.com/repos/{GITHUB_OWNER}/{repo_name}/actions/runs"
    run_id = None
    for _ in range(5):
        time.sleep(2)
        runs_resp = requests.get(runs_url, headers=headers, params={"branch": branch})
        if runs_resp.status_code != 200:
            continue

        runs_data = runs_resp.json().get("workflow_runs", [])
        if runs_data:
            run_id = runs_data[0]["id"]
            break

    if not run_id:
        raise Exception("Workflow triggered, but no run ID found.")

    run_url = f"https://github.com/{GITHUB_OWNER}/{repo_name}/actions/runs/{run_id}"
    return run_url, repo_name

def write_audit_excel(app_name, version, job_type, branch, environment, repo_name, run_url, status):
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
                "Timestamp", "Application", "Version", "JobType", "Branch",
                "Environment", "RepoName", "RunURL", "Status"
            ])

        ws.append([
            datetime.now().isoformat(),
            app_name, version, job_type, branch, environment,
            repo_name, run_url, status
        ])
        wb.save(filename)
