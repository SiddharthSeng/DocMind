"""
bge_hf_check.py — HF Inference API smoke test for BAAI/bge-small-en-v1.5
==========================================================================
Run this first to confirm the API is reachable and the response schema
is what _HFBGESmallEmbeddingFunction expects (list of float vectors).

Usage:
    # With token (higher rate limits — recommended):
    set HF_API_TOKEN=hf_xxxx
    python scripts/bge_hf_check.py

    # Without token (anonymous — works but lower limits):
    python scripts/bge_hf_check.py
"""

import os, sys, json, time
import requests

HF_MODEL  = "BAAI/bge-small-en-v1.5"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
token = os.getenv("HF_API_TOKEN", "")

headers = {"Content-Type": "application/json"}
if token:
    headers["Authorization"] = f"Bearer {token}"
    print(f"Using token: {token[:8]}...  (authenticated)")
else:
    print("No HF_API_TOKEN found — running anonymous (lower rate limits)")

test_sentences = [
    "What is the uptime commitment for Enterprise and Pro tier customers?",
    "What third-party subprocessors does DocMind use?",
]

print(f"\nCalling {HF_API_URL} ...")
t0 = time.time()
resp = requests.post(
    HF_API_URL,
    headers=headers,
    json={"inputs": test_sentences, "options": {"wait_for_model": True}},
    timeout=60,
)
elapsed = time.time() - t0

print(f"Status: {resp.status_code}  ({elapsed:.1f}s)")

if resp.status_code != 200:
    print(f"ERROR body: {resp.text[:500]}")
    sys.exit(1)

result = resp.json()

print(f"Response type : {type(result)}")
print(f"Outer length  : {len(result)}  (should be {len(test_sentences)})")

if isinstance(result, list) and len(result) > 0:
    first = result[0]
    print(f"Inner type    : {type(first)}")
    if isinstance(first, list):
        print(f"Embedding dim : {len(first)}  (bge-small = 384 expected)")
        print(f"First 5 values: {[round(v,4) for v in first[:5]]}")
        print("\n✓ Schema OK — _HFBGESmallEmbeddingFunction will work correctly.")
    else:
        print(f"Unexpected inner type: {first}")
        sys.exit(1)
else:
    print(f"Unexpected response: {result}")
    sys.exit(1)
