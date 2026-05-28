"""Verify the Flask API serves the corrected Lean code with (hpre : b > 0)."""
import subprocess
import time
import json
import urllib.request
import sys
import os

# Write output to a file to avoid console encoding issues
output_path = os.path.join(os.path.dirname(__file__), '_flask_final_output.txt')

def log(msg):
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')
    print(msg)

log("=== Starting Flask server test ===")

# Kill any existing Flask processes first
subprocess.run(['taskkill', '/f', '/im', 'python.exe'], 
               capture_output=True, shell=True)
time.sleep(2)

# Start Flask in background with -B flag
proc = subprocess.Popen(
    ['python', '-B', 'app.py'],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    cwd='D:/axiomzero2.0webapp'
)
time.sleep(3)

if proc.poll() is not None:
    log("ERROR: Flask server exited immediately")
    stdout, stderr = proc.communicate()
    log("STDOUT: " + stdout.decode('utf-8', errors='replace'))
    log("STDERR: " + stderr.decode('utf-8', errors='replace'))
    sys.exit(1)

log("Flask server started")

# Send request
payload = json.dumps({
    'source': '@requires(b > 0)\n@ensures(lambda result: (result * b <= a) and (a < (result + 1) * b))\ndef floor_div(a: int, b: int) -> int:\n    return a // b\n'
}).encode('utf-8')

req = urllib.request.Request(
    'http://127.0.0.1:5000/api/compile',
    data=payload,
    headers={'Content-Type': 'application/json'}
)

try:
    resp = urllib.request.urlopen(req, timeout=10)
    result = json.loads(resp.read().decode('utf-8'))
    lean_code = result.get('lean_code', '')
    
    log("=== Flask API Response ===")
    log(lean_code)
    log("")
    
    # Verify
    has_hpre = '(hpre : b > 0)' in lean_code
    has_fdiv = 'Int.fdiv' in lean_code
    has_hpre_in_proof = 'hpre' in lean_code.split(':=', 1)[1] if ':=' in lean_code else False
    
    log(f"Has (hpre : b > 0) in theorem signature: {has_hpre}")
    log(f"Has Int.fdiv: {has_fdiv}")
    log(f"hpre referenced in proof body: {has_hpre_in_proof}")
    
    if has_hpre and has_fdiv:
        log("SUCCESS: All checks passed!")
    else:
        log("FAILURE: Some checks failed")
        sys.exit(1)
        
except Exception as e:
    log(f"ERROR: {e}")
    sys.exit(1)
finally:
    proc.terminate()
    proc.wait()
    log("Flask server stopped")
