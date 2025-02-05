import requests
from flask import Flask, render_template_string
import threading
import time

app = Flask(__name__)

# Dictionary of Regions → Environments → Categories (Frontend, Backend) → Applications & URLs
applications = {
    "Region1": {
        "Development": {
            "Frontend": {
                "User Portal": "https://www.youtube.com/",
                "Admin Dashboard": "https://dev.frontend.region1.example.com/admin"
            },
            "Backend": {
                "User API": "https://dev.backend.region1.example.com/user",
                "Order API": "https://dev.backend.region1.example.com/order"
            }
        },
        "Staging": {
            "Frontend": {
                "User Portal": "https://staging.frontend.region1.example.com/user",
                "Admin Dashboard": "https://staging.frontend.region1.example.com/admin"
            },
            "Backend": {
                "User API": "https://staging.backend.region1.example.com/user",
                "Order API": "https://staging.backend.region1.example.com/order"
            }
        },
        "Production": {
            "Frontend": {
                "User Portal": "https://prod.frontend.region1.example.com/user",
                "Admin Dashboard": "https://prod.frontend.region1.example.com/admin"
            },
            "Backend": {
                "User API": "https://prod.backend.region1.example.com/user",
                "Order API": "https://prod.backend.region1.example.com/order"
            }
        }
    }
}

# Dictionary to store API statuses
api_statuses = {
    region: {
        env: {
            category: {api: "Checking..." for api in apis}
            for category, apis in envs.items()
        }
        for env, envs in regions.items()
    }
    for region, regions in applications.items()
}

# Function to check API statuses every minute
def check_apis():
    global api_statuses
    while True:
        for region, envs in applications.items():
            for env, categories in envs.items():
                for category, apis in categories.items():
                    for api_name, url in apis.items():
                        try:
                            response = requests.get(url, timeout=10)
                            if response.status_code == 200:
                                api_statuses[region][env][category][api_name] = "✔️ Up"
                            else:
                                api_statuses[region][env][category][api_name] = f"❌ Down ({response.status_code})"
                        except requests.RequestException as e:
                            api_statuses[region][env][category][api_name] = f"❌ Down ({str(e)})"
        time.sleep(60)  # Wait for 60 seconds before checking again

# HTML Template using Bootstrap
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>API Status Dashboard</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <style>
        body { font-family: Arial, sans-serif; background-color: #f8f9fa; }
        .container { margin-top: 20px; }
        .status-up { color: green; font-weight: bold; }
        .status-down { color: red; font-weight: bold; }
        .table th { background-color: #343a40; color: white; }
        .table td a { text-decoration: none; }
        .env-column { background-color: #e9ecef; padding: 10px; border-radius: 10px; margin-bottom: 20px; }
        .env-title { font-size: 1.2rem; font-weight: bold; }
        .region-card { margin-bottom: 30px; }
        .status-badge { padding: 5px 10px; border-radius: 5px; font-weight: bold; }
        .up-badge { background-color: #28a745; color: white; }
        .down-badge { background-color: #dc3545; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center">🚀 API Status Dashboard</h1>
        <p class="text-center text-muted">🔄 Page updates every 60 seconds</p>

        {% for region, envs in api_statuses.items() %}
        <div class="card region-card">
            <div class="card-header bg-primary text-white">
                <h2 class="mb-0">{{ region }}</h2>
            </div>
            <div class="card-body">
                <div class="row">
                    {% for env, categories in envs.items() %}
                    <div class="col-md-4">
                        <div class="env-column">
                            <div class="env-title text-center">{{ env }}</div>
                            {% for category, apis in categories.items() %}
                            <h4 class="text-dark mt-3">{{ category }}</h4>
                            <table class="table table-bordered">
                                <thead>
                                    <tr>
                                        <th>API Name</th>
                                        <th>Status</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for api, status in apis.items() %}
                                    <tr>
                                        <td><a href="{{ applications[region][env][category][api] }}" target="_blank">
                                            {{ api }}
                                        </a></td>
                                        <td>
                                            <span class="status-badge {{ 'up-badge' if 'Up' in status else 'down-badge' }}">
                                                {{ status }}
                                            </span>
                                        </td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                            {% endfor %}
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
        {% endfor %}
    </div>

    <script>
        setInterval(function() {
            window.location.reload();
        }, 60000); // Refresh the page every 60 seconds
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(html_template, api_statuses=api_statuses, applications=applications)

if __name__ == '__main__':
    threading.Thread(target=check_apis, daemon=True).start()
    app.run(debug=True, use_reloader=False)
