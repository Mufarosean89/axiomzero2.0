#!/usr/bin/env python3
"""Scan lean_workbook.json to understand its structure without loading it all into memory."""
import json
import sys
import re

def scan_large_json_array(filepath, max_objects=200):
    """Scan a large JSON array file, yielding first N objects."""
    decoder = json.JSONDecoder()
    with open(filepath, 'rb') as f:
        # Read in chunks to build buffer
        buffer = ""
        # Find the opening bracket
        while True:
            chunk = f.read(65536).decode('utf-8', errors='replace')
            if not chunk:
                break
            buffer += chunk
            idx = buffer.find('[')
            if idx >= 0:
                buffer = buffer[idx+1:]
                break
        
        count = 0
        while count < max_objects:
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
                yield obj
                count += 1
                buffer = buffer[pos:]
                # Skip comma and whitespace
                buffer = buffer.lstrip(', \t\n\r')
            except json.JSONDecodeError:
                # Need more data
                chunk = f.read(65536).decode('utf-8', errors='replace')
                if not chunk:
                    break
                buffer += chunk

# Scan the file
filepath = "datasets/lean_workbook/lean_workbook.json"

total_count = 0
with_proofs = 0
empty_proofs = 0
sample_entry = None
all_keys = set()

for obj in scan_large_json_array(filepath, max_objects=5000):
    total_count += 1
    all_keys.update(obj.keys())
    proof = obj.get('proof', [])
    if proof and len(proof) > 0:
        with_proofs += 1
    else:
        empty_proofs += 1
    if sample_entry is None:
        sample_entry = obj
    if total_count % 1000 == 0:
        print(f"  Scanned {total_count} entries...", file=sys.stderr)

print(f"\n=== Results ===")
print(f"Total entries scanned: {total_count}")
print(f"All keys found: {sorted(all_keys)}")
print(f"Entries with non-empty proofs: {with_proofs}")
print(f"Entries with empty proofs: {empty_proofs}")
print(f"\n=== Sample Entry (first) ===")
for k, v in sample_entry.items():
    if isinstance(v, str) and len(v) > 200:
        print(f"  {k}: {v[:200]}...")
    elif isinstance(v, list):
        print(f"  {k}: {v[:5]}... ({'empty' if not v else f'{len(v)} items'})")
    else:
        print(f"  {k}: {v}")
