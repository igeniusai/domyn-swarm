"""
generate_test_sample.py
-------------------------
Create a tiny Parquet dataset with one column ('messages') for quick LLM testing.
Run:  python save_llm_test_messages.py
"""

import sys
import subprocess
import pandas as pd

# ------------------------------------------------------------------
# 0) Ensure we have a Parquet engine available
# ------------------------------------------------------------------
def _install(package: str):
    """
    Utility: pip-install a missing package at runtime.
    Falls back to abort if installation fails (e.g., offline).
    """
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError as e:
        sys.exit(
            f"[ERROR] Could not install '{package}'. "
            "Install PyArrow or FastParquet manually and re-run."
        )

try:
    import pyarrow  # noqa: F401
    parquet_engine = "pyarrow"
except ImportError:
    try:
        import fastparquet  # noqa: F401
        parquet_engine = "fastparquet"
    except ImportError:
        print("[INFO] Neither PyArrow nor FastParquet found. Attempting to install PyArrow…")
        _install("pyarrow")
        import pyarrow  # noqa: F401
        parquet_engine = "pyarrow"

# ------------------------------------------------------------------
# 1) Craft sample messages
# ------------------------------------------------------------------
messages = [
    "Translate the following English sentence to French: 'Hello, world!'",
    "Summarize the following paragraph in one sentence: "
    "'Large language models are transforming natural language processing by enabling machines to understand and generate human-like text.'",
    "Write a haiku about rain.",
    "What is the capital of Japan?",
    "Explain quantum entanglement in simple terms suitable for a 12-year-old.",
    "Sort this list of numbers in ascending order: [42, 7, 19, 3, 25].",
    "Give me a Python function that checks if a number is prime.",
    "Fix the bug in this code snippet: 'def add(a, b): return a - b'.",
    "Generate a list of five creative business ideas involving AI in healthcare.",
    "What are the top three benefits of regular exercise?",
    "Write a short story (50-100 words) about a time-traveling cat.",
    "Classify the sentiment (positive, negative, or neutral) of this sentence: "
    "'I absolutely love this product!'"
]

# ------------------------------------------------------------------
# 2) Build DataFrame & save to Parquet
# ------------------------------------------------------------------
completion_df = pd.DataFrame({"messages": messages})
completion_out = "completion.parquet"
completion_df.to_parquet(completion_out, index=False, engine=parquet_engine)

chat_completion_df = pd.DataFrame({"messages": [{"role": "user", "content": message} for message in messages]})
chat_completion_out = "chat_completion.parquet"
chat_completion_df.to_parquet(chat_completion_out, index=False, engine=parquet_engine)

# ------------------------------------------------------------------
# 3) Quick sanity check
# ------------------------------------------------------------------
print(f"[OK] Wrote {len(completion_df)} rows to '{completion_out}' using engine: {parquet_engine}")
print(completion_df.head(5), "\n")

print(f"[OK] Wrote {len(chat_completion_df)} rows to '{chat_completion_out}' using engine: {parquet_engine}")
print(chat_completion_df.head(5), "\n")