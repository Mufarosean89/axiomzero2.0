"""Compare test results before and after changes."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

import importlib

# ── Run tests with current code (with fixes) ──
import test_phase4 as t1
importlib.reload(t1)
t1.passed = 0
t1.failed = 0
t1.test_count = 0

# Monkey-patch print to capture output
import io
buf1 = io.StringIO()
old_print = print
def capture1(*args, **kwargs):
    kwargs['file'] = buf1
    old_print(*args, **kwargs)

import builtins
builtins.print = capture1
try:
    t1.main()
finally:
    builtins.print = old_print

with_fixes = buf1.getvalue()

# ── Count failures in current code ──
current_fails = []
for line in with_fixes.split('\n'):
    if 'FAIL' in line:
        current_fails.append(line.strip())

print("=== FAILURES WITH FIXES ===")
for f in current_fails:
    print(f"  FAIL: {f}")

print(f"\nTotal failures with fixes: {len(current_fails)}")
