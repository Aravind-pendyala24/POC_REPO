import requests
from flask import Flask, render_template_string
import threading
import time

app = Flask(__name__)

# Dictionary of regions, environments, applications, and their URLs
applications = {
    "US": {
        "App1": {
            "Development": [
                "https://dev.app1.region1.example.com",
                "https://dev.app1.region1.google.com",
            ],
            "Staging": [
                "https://staging.app1.region1.example.com",
                "https://staging.app1.region1.google.com",
            ],
            "Production": [
                "https://www.app1.region1.example.com",
                "https://www.app1.region1.google.com",
            ]
        },
        "App2": {
            "Development": [
                "https://dev.app2.region1.example.com",
                "https://dev.app2.region1.google.com",
            ],
            "Staging": [
                "https://staging.app2.region1.example.com",
                "https://staging.app2.region1.google.com",
            ],
            "Production": [
                "https://www.app2.region1.example.com",
                "https://www.app2.region1.google.com",
            ]
        }
    },
    "EU": {
        "App1": {
            "Development": [
                "https://dev.app1.region2.example.com",
                "https://dev.app1.region2.google.com",
            ],
            "Staging": [
                "https://staging.app1.region2.example.com",
                "https://staging.app1.region2.google.com",
            ],
            "Production": [
                "https://www.app1.region2.example.com",
                "https://www.app1.region2.google.com",
            ]
        },
        "App2": {
            "Development": [
                "https://dev.app2.region2.example.com",
                "https://dev.app2.region2.google.com",
            ],
            "Staging": [
                "https://staging.app2.region2.example.com",
                "https://staging.app2.region2.google.com",
            ],
            "Production": [
                "https://www.app2.region2.example.com",
                "https://www.app2.region2.google.com",
            ]
        }
    }
}

# Dictionary to store the status of each URL in each environment for each application and region
url_statuses = {
    region: {
        app: {
            env: {url: "Unknown" for url in urls}
            for env, urls in envs.items()
        }
        for app, envs in apps.items()
    }
    for region, apps in applications.items()
}

# HTML template for the page
html_template = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>URL Status Checker</title>
    <style>
      body { font-family: Arial, sans-serif; font-size:15px}
      table { width: 100%; border-collapse: collapse; }
      th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
      th { background-color: #f2f2f2; }
      .status-up { color: green; }
      .status-down { color: red; }
    </style>
  </head>
  <body>
    <h1>URL Status Checker</h1>
    {% for region, apps in url_statuses.items() %}
    <h2>{{ region }}</h2>
    {% for app, envs in apps.items() %}
    <h3>{{ app }}</h3>
    {% for env, statuses in envs.items() %}
    <h4>{{ env }}</h4>
    <table>
      <tr>
        <th>URL</th>
        <th>Status</th>
      </tr>
      {% for url, status in statuses.items() %}
      <tr>
        <td>{{ url }}</td>
        <td class="{{ 'status-up' if 'Up' in status else 'status-down' }}">{{ status }}</td>
      </tr>
      {% endfor %}
    </table>
    {% endfor %}
    {% endfor %}
    {% endfor %}
    <p>Page updates every minute.</p>
    <script>
      setInterval(function() {
        window.location.reload();
      }, 60000); // Refresh the page every 60 seconds
    </script>
  </body>
</html>
"""

def check_urls():
    global url_statuses
    while True:
        for region, apps in applications.items():
            for app, envs in apps.items():
                for env, urls in envs.items():
                    for url in urls:
                        try:
                            response = requests.get(url, timeout=10)
                            if response.status_code == 200:
                                url_statuses[region][app][env][url] = "Up (Status Code: 200)"
                            else:
                                url_statuses[region][app][env][url] = f"Up (Status Code: {response.status_code})"
                        except requests.RequestException as e:
                            url_statuses[region][app][env][url] = f"Down ({str(e)})"
        time.sleep(60)  # Wait for 60 seconds before checking again

@app.route('/')
def index():
    return render_template_string(html_template, url_statuses=url_statuses)

if __name__ == '__main__':
    # Start the background thread to check URLs
    threading.Thread(target=check_urls, daemon=True).start()
    # Start the Flask web server
    app.run(debug=True, use_reloader=False)
