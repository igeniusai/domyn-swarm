#!/usr/bin/env python3
# Copyright 2025 iGenius S.p.A
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
simple_driver.py
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
