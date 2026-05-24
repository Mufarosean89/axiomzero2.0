import subprocess, time, sys, urllib.request, json, os

# Kill only the process on port 5000
try:
    result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True, timeout=5)
    for line in result.stdout.splitlines():
        if '0.0.0.0:5000' in line or '127.0.0.1:5000' in line or '[::]:5000' in line:
            parts = line.strip().split()
            if parts:
                pid = parts[-1]
                if pid.isdigit():
                    os.system(f'taskkill /F /PID {pid} 2>nul')
                    time.sleep(1)
except Exception:
    pass

# Start Flask in background
print('Starting Flask...')
proc = subprocess.Popen(
    [sys.executable, 'app.py'],
    cwd=r'D:\axiomzero2.0webapp',
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)
time.sleep(3)

poll = proc.poll()
if poll is not None:
    print(f'Flask exited with code {poll}')
    sys.exit(1)

print(f'Flask running (PID: {proc.pid})')

# Test the API
source = '''@requires(lambda n: isinstance(n, int))
@ensures(lambda n, result: result == n)
def identity(n: int) -> int:
    return n
'''
data = json.dumps({'source': source}).encode()
req = urllib.request.Request(
    'http://127.0.0.1:5000/api/compile',
    data=data,
    headers={'Content-Type': 'application/json'}
)
resp = urllib.request.urlopen(req, timeout=10)
result = json.loads(resp.read().decode())
lean_code = result.get('lean_code', '')

# Write to file
with open('_api_verify.txt', 'w', encoding='utf-8') as f:
    f.write(lean_code)
    f.write('\n')

# Print escaped version for terminal
escaped = lean_code.encode('unicode_escape').decode('ascii')
print('=== WEB APP OUTPUT (escaped) ===')
print(escaped)
print('=== END ===')
print()
print(f'rfl: {"rfl" in lean_code}')
print(f'theorems: {lean_code.count("theorem")}')
print(f'by: {"by" in lean_code}')

proc.terminate()
proc.wait()
print('Server stopped')
