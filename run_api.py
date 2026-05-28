import subprocess
import time
import requests
import os
import signal

# Start the application
app_process = subprocess.Popen(["python", "-B", "app.py"], cwd="D:/axiomzero2.0webapp")
time.sleep(3)

# Send request
url = "http://127.0.0.1:5000/api/compile"
payload = {
    "source": "@requires(b > 0)\n@ensures(lambda result: (result * b <= a) and (a < (result + 1) * b))\ndef floor_div(a: int, b: int) -> int:\n    return a // b\n"
}
try:
    response = requests.post(url, json=payload)
    print(response.text)
finally:
    app_process.terminate()
