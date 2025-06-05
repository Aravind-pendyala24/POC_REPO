from flask import Flask, send_from_directory, abort
import os

app = Flask(__name__)

# Allowed folders
ALLOWED_FOLDERS = {
    'xml': '/usr/share/xml',
    'xsl': '/usr/share/xsl',
    'common': '/usr/share/common'
}

HTML_DIR = '/usr/share'
ALLOWED_HTML_FILES = {'index.html', 'second.html'}

# 🔹 Default route to index.html
@app.route('/')
def serve_index():
    return send_from_directory(HTML_DIR, 'index.html')

# 🔹 Serve specific HTML files
@app.route('/<filename>')
def serve_html_file(filename):
    if filename not in ALLOWED_HTML_FILES:
        abort(403)

    full_path = os.path.abspath(os.path.join(HTML_DIR, filename))

    if not full_path.startswith(os.path.abspath(HTML_DIR)) or not os.path.isfile(full_path):
        abort(403)

    return send_from_directory(HTML_DIR, filename)

# 🔹 Serve xml/xsl/common files
@app.route('/<folder>/<path:filename>')
def serve_static_file(folder, filename):
    if folder not in ALLOWED_FOLDERS:
        abort(403)

    base_dir = ALLOWED_FOLDERS[folder]
    requested_path = os.path.abspath(os.path.join(base_dir, filename))

    if not requested_path.startswith(os.path.abspath(base_dir)) or not os.path.isfile(requested_path):
        abort(403)

    return send_from_directory(base_dir, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=30080)
