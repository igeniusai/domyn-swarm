"""
generate_test_sample.py
-------------------------
Create a tiny Parquet dataset with one column ('messages') for quick LLM testing.
Run:  python save_llm_test_messages.py
"""

import subprocess
import sys

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
    except subprocess.CalledProcessError:
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
        print(
            "[INFO] Neither PyArrow nor FastParquet found. Attempting to install PyArrow…"
        )
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
    "'I absolutely love this product!'",
    "Create a SQL query to find the top 10 customers by total purchase amount from a sales table.",
    "What is the weather like in New York City today?",
    "Generate a random password that is 12 characters long, including uppercase, lowercase, numbers, and special characters.",
    "What is the meaning of life?",
    "Write a brief history of the internet in 5 sentences.",
    "What is the Pythagorean theorem?",
    "Create a regex pattern to match valid email addresses.",
    "What is the capital of France?",
    "Write a Python script that reads a CSV file and prints the first 5 rows.",
    "What is the square root of 256?",
    "Generate a list of 10 random integers between 1 and 100.",
    "What is the largest mammal on Earth?",
    "Write a function that calculates the factorial of a number.",
    "What is the chemical formula for water?",
    "What is the distance between the Earth and the Moon?",
    "Write a short poem about the beauty of nature.",
    "What is the capital of Italy?",
    "What is the boiling point of water in degrees Celsius?",
    "What is the smallest prime number?",
    "What is the largest planet in our solar system?",
    "What is the process of photosynthesis?",
    "What is the capital of Germany?",
    "What is the currency of Japan?",
    'Convert this JSON string to a Python dictionary: \'{"name": "Alice", "age": 30}\'',
    "Write a tweet about climate change using less than 280 characters.",
    "Explain the difference between supervised and unsupervised learning.",
    "Describe how a refrigerator works in simple terms.",
    "Generate a list of rhyming words for 'light'.",
    "Write an email to request a meeting with your manager next week.",
    "What is the derivative of x^2 + 3x + 2?",
    "Create a markdown table with three columns: Name, Age, Country.",
    "Translate this sentence into Spanish: 'The book is on the table.'",
    "List five famous works by Leonardo da Vinci.",
    "Generate a JSON schema for a user profile with name, email, and age.",
    "What are Newton's three laws of motion?",
    "Give a pros and cons list for remote work.",
    "Write a product description for a smart water bottle.",
    "Explain the concept of blockchain to a non-technical audience.",
    "Write a limerick about a dog who loves to swim.",
    "What’s the output of this Python code? 'print(2 ** 3 ** 2)'",
    "Generate a slogan for a company that sells solar panels.",
    "Suggest a title for a science fiction novel about AI rebellion.",
    "Write a function in JavaScript to reverse a string.",
    "Describe how to bake a chocolate cake from scratch.",
    "What's the difference between HTTP and HTTPS?",
    "Generate a motivational quote for Monday mornings.",
    "Write a LinkedIn summary for a data analyst.",
    "What are three common symptoms of the flu?",
    "Describe the plot of 'Romeo and Juliet' in two sentences.",
    "List three use cases for edge computing.",
    "Create a YAML configuration for a Docker Compose setup with a web and db service.",
    "Generate a fictional name and backstory for a fantasy RPG character.",
    "What’s the chemical symbol for gold?",
]

# ------------------------------------------------------------------
# 2) Build DataFrame & save to Parquet
# ------------------------------------------------------------------
completion_df = pd.DataFrame({"messages": messages})
completion_out = "completion.parquet"
completion_df.to_parquet(completion_out, index=False, engine=parquet_engine)

chat_completion_df = pd.DataFrame(
    {"messages": [[{"role": "user", "content": message}] for message in messages]}
)
chat_completion_out = "chat_completion.parquet"
chat_completion_df.to_parquet(chat_completion_out, index=False, engine=parquet_engine)

# ------------------------------------------------------------------
# 3) Quick sanity check
# ------------------------------------------------------------------
print(
    f"[OK] Wrote {len(completion_df)} rows to '{completion_out}' using engine: {parquet_engine}"
)
print(completion_df.head(5), "\n")

print(
    f"[OK] Wrote {len(chat_completion_df)} rows to '{chat_completion_out}' using engine: {parquet_engine}"
)
print(chat_completion_df.head(5), "\n")
