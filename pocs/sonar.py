import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# SonarQube details
SONAR_URL = "http://<SONARQUBE_URL>"  # Replace with your SonarQube URL
SONAR_TOKEN = "your_sonar_token"  # Replace with your token
HEADERS = {"Authorization": f"Basic {SONAR_TOKEN.encode('utf-8').hex()}"}

# Metrics to fetch
METRICS = "coverage,bugs,vulnerabilities,code_smells,reliability_rating,security_rating"
MAX_THREADS = 10  # Adjust based on your system and API limits
INPUT_DASHBOARD_IDS = ["dashboard_id_1", "dashboard_id_2"]  # Replace with your dashboard IDs


def get_projects_from_dashboard(dashboard_id):
    """Fetch projects associated with a given dashboard ID."""
    url = f"{SONAR_URL}/api/projects/search?dashboardId={dashboard_id}"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json().get("components", [])


def get_branches(project_key):
    """Fetch all branches for a given project."""
    url = f"{SONAR_URL}/api/project_branches/list?project={project_key}"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    branches = response.json().get("branches", [])
    return [branch for branch in branches if branch["name"] == "main" or branch["name"].startswith("release/")]


def get_metrics_and_date(project_key, branch_name):
    """Fetch metrics and scan date for a specific branch of a project."""
    url = f"{SONAR_URL}/api/measures/component?component={project_key}&metricKeys={METRICS}&branch={branch_name}"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    component = response.json().get("component", {})
    measures = {metric["metric"]: metric["value"] for metric in component.get("measures", [])}
    analysis_date = component.get("analysisDate", "N/A")
    return measures, analysis_date


def process_project_branch(project, branches):
    """Fetch metrics and scan date for all branches of a project."""
    project_key = project["key"]
    project_name = project["name"]
    report_data = []

    for branch in branches:
        branch_name = branch["name"]
        try:
            metrics, analysis_date = get_metrics_and_date(project_key, branch_name)
            metrics.update({
                "Project Name": project_name,
                "Project Key": project_key,
                "Branch Name": branch_name,
                "Scan Date": analysis_date,
            })
            report_data.append(metrics)
        except Exception as e:
            print(f"Failed to fetch data for {project_key} (branch {branch_name}): {e}")

    return report_data


def process_dashboard(dashboard_id):
    """Process all projects in a dashboard."""
    report_data = []

    try:
        projects = get_projects_from_dashboard(dashboard_id)
    except Exception as e:
        print(f"Failed to fetch projects for dashboard {dashboard_id}: {e}")
        return []

    for project in projects:
        try:
            branches = get_branches(project["key"])
            report_data.extend(process_project_branch(project, branches))
        except Exception as e:
            print(f"Failed to process project {project['key']}: {e}")

    return report_data


def main():
    """Main function to generate the report."""
    report_data = []

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        future_to_dashboard = {
            executor.submit(process_dashboard, dashboard_id): dashboard_id
            for dashboard_id in INPUT_DASHBOARD_IDS
        }
        for future in as_completed(future_to_dashboard):
            try:
                report_data.extend(future.result())
            except Exception as e:
                dashboard_id = future_to_dashboard[future]
                print(f"Error processing dashboard {dashboard_id}: {e}")

    # Generate Excel report
    if report_data:
        df = pd.DataFrame(report_data)
        df.to_excel("SonarQube_Dashboard_Metrics_Report.xlsx", index=False)
        print("Report generated: SonarQube_Dashboard_Metrics_Report.xlsx")
    else:
        print("No data to generate the report.")


if __name__ == "__main__":
    main()
