from flask import Flask, request, jsonify, send_from_directory
import os
import traceback

# Import the existing compiler from Axiom Zero
from lean_compiler import compile as compile_to_lean

app = Flask(__name__, static_folder='static')


@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


@app.route('/api/compile', methods=['POST'])
def compile_code():
    data = request.get_json()
    if not data or 'source' not in data:
        return jsonify({'error': 'No source code provided'}), 400

    source = data['source']

    if not isinstance(source, str) or not source.strip():
        return jsonify({'error': 'Source code must be a non-empty string'}), 400

    try:
        lean_code = compile_to_lean(source, "web_module")
        return jsonify({'lean_code': lean_code})
    except Exception as e:
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        return jsonify({
            'error': error_msg,
            'traceback': traceback_str,
        }), 500


if __name__ == '__main__':
    import sys
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug, host='127.0.0.1', port=port)

