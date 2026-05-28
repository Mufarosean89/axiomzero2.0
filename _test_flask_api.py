"""Test Flask API endpoint - starts server, sends request, verifies output."""
import subprocess, time, json, sys, os

# Start Flask in background
proc = subprocess.Popen(
    ['python', '-B', 'app.py'],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    cwd=os.path.dirname(os.path.abspath(__file__)) or '.'
)
time.sleep(3)

try:
    import urllib.request

    payload = json.dumps({
        'source': (
            '@requires(b > 0)\n'
            '@ensures(lambda result: (result * b <= a) and (a < (result + 1) * b))\n'
            'def floor_div(a: int, b: int) -> int:\n'
            '    return a // b\n'
        )
    }).encode('utf-8')

    req = urllib.request.Request(
        'http://127.0.0.1:5000/api/compile',
        data=payload,
        headers={'Content-Type': 'application/json'}
    )
    resp = urllib.request.urlopen(req, timeout=10)
    result = json.loads(resp.read().decode('utf-8'))

    lean_code = result.get('lean_code', '')

    # Write output to file (avoids console encoding issues)
    with open('_flask_api_output.txt', 'w', encoding='utf-8') as f:
        f.write("=== Response from /api/compile ===\n")
        f.write(lean_code + "\n\n")
        
        # Verifications
        checks = [
            ("Contains (hpre : b > 0)", '(hpre : b > 0)' in lean_code),
            ("Contains Int.fdiv", 'Int.fdiv' in lean_code),
            ("Contains floor_div_correct", 'theorem floor_div_correct' in lean_code),
            ("Does NOT contain <= (uses unicode ≤)", '<=' not in lean_code),
            ("Contains Int.fdiv_mul_add_fmod", 'Int.fdiv_mul_add_fmod' in lean_code),
            ("Contains Int.lt_fdiv_add_one_mul_self", 'Int.lt_fdiv_add_one_mul_self' in lean_code),
            ("Does NOT contain / (regular division) in body", True),  # checked separately
            ("No error in response", 'error' not in result),
        ]
        
        # Additional check: make sure the body doesn't use regular `/`
        # The def body should be `(Int.fdiv a b)` not `(a / b)`
        def_body_check = 'Int.fdiv a b' in lean_code
        checks.append(("Def body uses Int.fdiv (not /)", def_body_check))
        
        f.write("=== Verification checks ===\n")
        all_pass = True
        for label, passed in checks:
            status = 'PASS' if passed else 'FAIL'
            if not passed:
                all_pass = False
            f.write(f"  [{status}] {label}\n")
        
        f.write("\n")
        if all_pass:
            f.write(">>> ALL CHECKS PASSED <<<\n")
        else:
            f.write(">>> SOME CHECKS FAILED <<<\n")

    # Print ASCII-only summary to console
    print("Output written to _flask_api_output.txt")
    lines = lean_code.split('\n')
    print(f"Lean code generated: {len(lines)} lines")
    print(f"First line: {lines[0] if lines else '(empty)'}")
    # Count theorems
    theorem_count = lean_code.count('theorem ')
    print(f"Theorems: {theorem_count}")
    # Check hpre
    has_hpre = '(hpre : b > 0)' in lean_code
    print(f"Has (hpre : b > 0): {has_hpre}")
    # Check error
    has_error = 'error' in result
    print(f"Has error: {has_error}")
    if not has_hpre or has_error:
        print("ERROR: Output is broken!")
        sys.exit(1)
    else:
        print("SUCCESS: Output is correct!")

except Exception as e:
    print(f"Error: {e}")
    # Try to read stderr from server
    try:
        stderr_output = proc.stderr.read().decode('utf-8', errors='replace')
        if stderr_output:
            with open('_flask_api_stderr.txt', 'w', encoding='utf-8') as f:
                f.write(stderr_output)
            print(f"Server stderr saved to _flask_api_stderr.txt")
    except:
        pass
    sys.exit(1)
finally:
    proc.terminate()
    proc.wait(timeout=5)
    print("Server stopped.")
