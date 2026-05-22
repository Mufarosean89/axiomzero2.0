#!/usr/bin/env python3
"""Save sample entries from lean_workbook.json to a clean text file."""
import json
import sys

decoder = json.JSONDecoder()

with open('datasets/lean_workbook/lean_workbook.json', 'rb') as f:
    buffer = ""
    while True:
        chunk = f.read(65536).decode('utf-8', errors='replace')
        if not chunk:
            break
        buffer += chunk
        idx = buffer.find('[')
        if idx >= 0:
            buffer = buffer[idx+1:]
            break

    lines = []
    with_proof_idx = 0
    
    while with_proof_idx < 5:
        buffer = buffer.lstrip()
        if not buffer:
            chunk = f.read(65536).decode('utf-8', errors='replace')
            if not chunk:
                break
            buffer += chunk
            continue
        
        if buffer.startswith(']'):
            break
        
        try:
            obj, pos = decoder.raw_decode(buffer)
            proof = obj.get('proof', [])
            
            if proof and len(proof) > 0:
                with_proof_idx += 1
                # Encode with utf-8 for file output
                lines.append(f"=== ENTRY {with_proof_idx} ===")
                lines.append(f"formal_statement: {obj.get('formal_statement', '')}")
                lines.append(f"proof: {proof}")
                lines.append(f"tags: {obj.get('tags', [])}")
                lines.append("")
            
            buffer = buffer[pos:]
            buffer = buffer.lstrip(', \t\n\r')
        except json.JSONDecodeError:
            chunk = f.read(65536).decode('utf-8', errors='replace')
            if not chunk:
                break
            buffer += chunk

output = '\n'.join(lines)
with open('_workbook_samples2.txt', 'w', encoding='utf-8') as f:
    f.write(output)
print(f"Wrote {len(lines)} lines to _workbook_samples2.txt")
