import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('lean_compiler/ir_to_lean.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix: join the broken f-string at lines 525-526
# Line 525 ends with "{hyp_name})' - missing the closing \n"
# Line 526 is just '\n"' 
old = '''                    f\"            Int.fmod_eq_emod_of_nonneg (le_of_lt {hyp_name})
\\n\"'''
new = '''                    f\"            Int.fmod_eq_emod_of_nonneg (le_of_lt {hyp_name})\\n\"'''

if old in content:
    content = content.replace(old, new)
    with open('lean_compiler/ir_to_lean.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('FIX APPLIED - f-string joined onto one line')
else:
    print('OLD STRING NOT FOUND - checking exact representation...')
    # Show what's actually there
    lines = content.split('\\n')
    for i, line in enumerate(lines[524:528], start=525):
        print(f'Line {i}: {repr(line)}')
