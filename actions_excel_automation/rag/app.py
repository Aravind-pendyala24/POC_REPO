import os
import logging
from flask import Flask, render_template, request, redirect, url_for, flash
from dotenv import load_dotenv
from app.rag_core import answer

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "super-secret")

# Logging
logging.basicConfig(filename="app.log", level=logging.INFO)

@app.route("/", methods=["GET", "POST"])
def search():
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if not query:
            flash("⚠️ Please enter a query.", "warning")
            return redirect(url_for("search"))

        if "deployment request" in query.lower():
            return redirect(url_for("deployment"))

        try:
            ans, ctx = answer(query)
            return render_template("qa.html", query=query, answer=ans, contexts=ctx)
        except Exception as e:
            logging.error(f"RAG error: {e}")
            flash("❌ Error answering query.", "danger")
            return redirect(url_for("search"))

    return render_template("qa.html")

@app.route("/deployment", methods=["GET", "POST"])
def deployment():
    if request.method == "POST":
        application = request.form.get("application")
        version = request.form.get("version")
        environment = request.form.get("environment")
        job_type = request.form.get("job_type")

        logging.info(f"Deployment request: {application} {version} {environment} {job_type}")
        flash("✅ Deployment submitted!", "success")
        return redirect(url_for("deployment"))

    import json
    with open("apps.json") as f:
        applications = json.load(f)

    environments = ["dev", "qa", "uat", "prod"]
    job_types = ["deployment", "test"]

    return render_template("form3.html",
                           applications=applications,
                           environments=environments,
                           job_types=job_types)

if __name__ == "__main__":
    app.run(debug=True, port=8080)
