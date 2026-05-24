"""Quick verification script for the isinstance fix."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from lean_compiler import compile as axiom_compile

source = '''\
@requires(lambda a, b: isinstance(a, int) and isinstance(b, int))
@ensures(lambda a, b, result: result == a + b)
@ensures(lambda a, b, result: result == b + a)          # commutativity
def add(a: int, b: int) -> int:
    return a + b
'''

result = axiom_compile(source, 'web_module', fill_holes=True)
with open('_verify_output.txt', 'w', encoding='utf-8') as f:
    f.write(result)
print("Output written to _verify_output.txt")
print("---")
# Also print ASCII-printable summary
lines = result.split('\n')
for line in lines:
    # Check for Python concepts leaking into Lean
    if 'isinstance' in line.lower():
        print(f"WARNING: isinstance found in output: {line}")
    if 'hpre' in line:
        print(f"WARNING: hpre found in output: {line}")
    if 'sorry' in line and 'by' not in line.split('sorry')[0][-20:]:
        print(f"INFO: sorry still present: {line}")
    print(line)
