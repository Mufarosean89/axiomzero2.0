#!/usr/bin/env python3
"""Debug the Lean 4 LSP protocol - test raw communication without select.select."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import subprocess
import threading
import time

lean_path = "C:/Users/seanr/.elan/bin/lean"

# Test 1: Just run lean --version
print("Test 1: Lean version")
r = subprocess.run([lean_path, "--version"], capture_output=True, text=True, timeout=10)
print(f"  stdout: {r.stdout.strip()}")
print(f"  stderr: {r.stderr.strip()}")
print()

# Test 2: Use lean --server with thread-based reading
print("Test 2: Thread-based LSP communication")
proc = subprocess.Popen(
    [lean_path, "--server"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

# Background reader thread for stdout
stdout_data = []
stdout_done = threading.Event()

def read_stdout():
    try:
        while True:
            chunk = proc.stdout.read(65536)
            if not chunk:
                break
            stdout_data.append(chunk)
            stdout_done.set()
    except (ValueError, OSError):
        pass

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

print(f"  Init body ({len(init_body.encode('utf-8'))} bytes): {init_body[:100]}...")

# Send with Content-Length header  
encoded_body = init_body.encode("utf-8")
header = f"Content-Length: {len(encoded_body)}\r\n\r\n".encode("ascii")
frame = header + encoded_body
print(f"  Frame ({len(frame)} bytes): {header.decode('ascii')}{init_body[:80]}...")

proc.stdin.write(frame)
proc.stdin.flush()

# Wait for response
time.sleep(3)

all_stdout = b"".join(stdout_data)
print(f"\n  Stdout ({len(all_stdout)} bytes): {all_stdout[:500]}")

# Check if any stderr
err_data = proc.stderr.read()
if err_data:
    print(f"\n  Stderr ({len(err_data)} bytes): {err_data[:300]}")

proc.terminate()
proc.wait(timeout=3)
print("\nDone")
