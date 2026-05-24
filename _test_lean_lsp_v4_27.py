#!/usr/bin/env python3
"""Test Lean 4 v4.27.0 LSP communication using thread-based reader."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import subprocess
import threading
import time

lean_path = "C:/Users/seanr/.elan/bin/lean"
lean_cmd = [lean_path, "+v4.27.0", "--server"]
print(f"Testing: {' '.join(lean_cmd)}")

proc = subprocess.Popen(
    lean_cmd,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

# Background reader thread for stdout
stdout_data = []
stdout_done = threading.Event()
read_errors = []

def read_stdout():
    try:
        while True:
            chunk = proc.stdout.read(65536)
            if not chunk:
                break
            stdout_data.append(chunk)
            stdout_done.set()
    except Exception as e:
        read_errors.append(str(e))

reader = threading.Thread(target=read_stdout, daemon=True)
reader.start()

# Build initialize message
init_body = json.dumps({
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "processId": os.getpid(),
        "clientInfo": {"name": "test", "version": "0.1"},
        "capabilities": {}
    }
})

encoded_body = init_body.encode("utf-8")
header = f"Content-Length: {len(encoded_body)}\r\n\r\n".encode("ascii")
frame = header + encoded_body

print(f"Body length: {len(encoded_body)} bytes")
print(f"Sending: Content-Length: {len(encoded_body)}")
print(f"Body: {init_body[:80]}...")

proc.stdin.write(frame)
proc.stdin.flush()

# Wait for response
time.sleep(3)

all_stdout = b"".join(stdout_data)
if all_stdout:
    print(f"\nGot stdout ({len(all_stdout)} bytes):")
    print(f"  Raw: {all_stdout[:300]}")
    # Try to parse
    try:
        text = all_stdout.decode('utf-8')
        print(f"  Text: {text[:300]}")
    except:
        print(f"  (binary data)")
else:
    print(f"\nNo stdout received in 3 seconds")

# Check stderr
try:
    err_data = proc.stderr.read()
    if err_data:
        print(f"\nStderr ({len(err_data)} bytes): {err_data[:300]}")
except:
    pass

if read_errors:
    print(f"\nRead errors: {read_errors}")

proc.terminate()
proc.wait(timeout=3)

# Print exit code
print(f"\nExit code: {proc.returncode}")
