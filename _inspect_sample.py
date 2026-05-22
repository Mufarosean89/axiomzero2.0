#!/usr/bin/env python3
"""Extract sample entries from lean_workbook.json for inspection."""
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

    samples_found = 0
    with_proof_idx = 0
    without_proof_idx = 0
    
    while samples_found < 300:
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
            
            if proof and len(proof) > 0 and with_proof_idx < 3:
                with_proof_idx += 1
                samples_found += 1
                print(f"=== ENTRY WITH PROOF #{with_proof_idx} ===")
                print(f"formal_statement: {obj.get('formal_statement', '')}")
                print(f"proof: {proof}")
                print(f"tags: {obj.get('tags', [])}")
                print()
            
            if (not proof or len(proof) == 0) and without_proof_idx < 2:
                without_proof_idx += 1
                samples_found += 1
                print(f"=== ENTRY WITHOUT PROOF #{without_proof_idx} ===")
                print(f"formal_statement: {obj.get('formal_statement', '')[:300]}")
                print()
            
            buffer = buffer[pos:]
            buffer = buffer.lstrip(', \t\n\r')
        except json.JSONDecodeError:
            chunk = f.read(65536).decode('utf-8', errors='replace')
            if not chunk:
                break
            buffer += chunk
