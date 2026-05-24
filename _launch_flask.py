"""Launch the Flask server in a new window."""
import subprocess, sys, os

app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.py')

# Use subprocess with DETACHED_PROCESS flag (Windows)
# If DETACHED_PROCESS doesn't work, we fall back to CREATE_NO_WINDOW
try:
    proc = subprocess.Popen(
        [sys.executable, app_path],
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f'Flask server started (PID: {proc.pid})')
    print(f'Open http://127.0.0.1:5000 in your browser')
except Exception as e:
    print(f'Error: {e}')
