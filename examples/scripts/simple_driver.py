#!/usr/bin/env python3
"""
driver_test.py
==============

Runs inside the swarm allocation (node-0).  It

  1. reads the load-balancer URL from the ENDPOINT env-var (set by llm_swarm)
  2. POSTs a few prompts to the vLLM REST endpoint  <ENDPOINT>/generate
  3. prints the responses and writes a JSON-Lines file for later inspection

No external dependencies beyond the standard `json` and the ubiquitous
`requests` package.

If your image lacks `requests`, just `pip install --user requests` once
in your user environment—Python packages are visible inside the container
because `$HOME` is **not** mounted.
"""

import json
import os
import pathlib
import sys
import time

import requests

endpoint = os.environ.get("ENDPOINT")  # injected by llm_swarm.lb
if endpoint is None:
    sys.exit("❌  ENDPOINT env-var not set")

url = f"{endpoint.rstrip('/')}/v1/chat/completions"
print(f"🔌  Using inference URL: {url}")

prompts = [
    "Hello, world!",
    "Explain reinforcement learning in one paragraph.",
    "Translate 'cat' into French.",
]

out_path = pathlib.Path("driver_outputs.jsonl").absolute()
with out_path.open("w") as fout:
    for i, p in enumerate(prompts, 1):
        payload = {
            "prompt": p,
            "max_tokens": 64,
            "temperature": 0.7,
        }
        t0 = time.time()
        resp = requests.post(url, json=payload, timeout=120)
        elap = time.time() - t0
        resp.raise_for_status()
        txt = resp.json()["text"]

        print(f"\n--- Prompt {i} ---------------------------")
        print(p)
        print("\n--- Completion ---------------------------")
        print(txt.strip())
        print(f"⏱  {elap:.2f} s")

        fout.write(json.dumps({"prompt": p, "completion": txt}) + "\n")

print(f"\n✅  All done. Results in {out_path}")

# rm -rf /leonardo_work/iGen_train/shared_hf_cache/hub/.locks/
# HF_HOME=/leonardo_work/iGen_train/shared_hf_cache/ uv run huggingface-cli download Qwen/Qwen3-235B-A22B --repo-type model --max-workers 4
