import requests
from flask import Flask, render_template_string
import threading
import time

app = Flask(__name__)

# Dictionary of Regions → Environments → Applications → APIs with URLs
applications = {
    "Region1": {
        "Development": {
            "User API": "https://dev.api.region1.example.com/user",
            "Order API": "https://dev.api.region1.example.com/order",
        },
        "Staging": {
            "User API": "https://staging.api.region1.example.com/user",
            "Order API": "https://staging.api.region1.example.com/order",
        },
        "Production": {
            "User API": "https://prod.api.region1.example.com/user",
            "Order API": "https://prod.api.region1.example.com/order",
        }
    },
    "Region2": {
        "Development": {
            "User API": "https://dev.api.region2.example.com/user",
            "Payment API": "https://dev.api.region2.example.com/payment",
        },
        "Staging": {
            "User API": "https://staging.api.region2.example.com/user",
            "Payment API": "https://staging.api.region2.example.com/payment",
        },
        "Production": {
            "User API": "https://prod.api.region2.example.com/user",
            "Payment API": "https://prod.api.region2.example.com/payment",
        }
    }
}

# Dictionary to store the status of each API
api_statuses = {
    region: {
        env: {api: "Checking..." for api in apis}
        for env, apis in envs.items()
    }
    for region, envs in applications.items()
}

# HTML template using Bootstrap for a better design
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
        .card { margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center">API Status Dashboard</h1>
        <p class="text-center text-muted">Page updates every 60 seconds</p>

        {% for region, envs in api_statuses.items() %}
        <div class="card">
            <div class="card-header bg-primary text-white">
                <h2 class="mb-0">{{ region }}</h2>
            </div>
            <div class="card-body">
                {% for env, apis in envs.items() %}
                <h3 class="text-secondary">{{ env }}</h3>
                <table class="table table-bordered">
                    <thead class="table-dark">
                        <tr>
                            <th>API Name</th>
                            <th>URL</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for api, status in apis.items() %}
                        <tr>
                            <td>{{ api }}</td>
                            <td><a href="{{ applications[region][env][api] }}" target="_blank">{{ applications[region][env][api] }}</a></td>
                            <td class="{{ 'status-up' if 'Up' in status else 'status-down' }}">{{ status }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
                {% endfor %}
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

# Function to check API statuses every minute
def check_apis():
    global api_statuses
    while True:
        for region, envs in applications.items():
            for env, apis in envs.items():
                for api_name, url in apis.items():
                    try:
                        response = requests.get(url, timeout=10)
                        if response.status_code == 200:
                            api_statuses[region][env][api_name] = "Up (200 OK)"
                        else:
                            api_statuses[region][env][api_name] = f"Down ({response.status_code})"
                    except requests.RequestException as e:
                        api_statuses[region][env][api_name] = f"Down ({str(e)})"
        time.sleep(60)  # Wait for 60 seconds before checking again

@app.route('/')
def index():
    return render_template_string(html_template, api_statuses=api_statuses, applications=applications)

if __name__ == '__main__':
    # Start background thread to check API statuses
    threading.Thread(target=check_apis, daemon=True).start()
    # Start Flask web server
    app.run(debug=True, use_reloader=False)
