import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('lean_compiler/ir_to_lean.py', 'r', encoding='utf-8') as f:
    content = f.read()
    lines = content.split('\n')

for i, line in enumerate(lines[510:535], start=511):
    print(f'Line {i}: {repr(line)}')
