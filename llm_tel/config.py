import os
import sys

ROTATING_MODELS = [
    "google/gemini-2.0-flash-001",  # fast & cheap
    "anthropic/claude-sonnet-4",  # good at paraphrasing
]

JUDGE_MODEL = "openai/gpt-4o-mini"  # reserved for scoring only
MAX_CYCLES = 30
STOP_SCORE = 2  # ≤ 2 ⇒ terminate
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/smusali/llm-telephone",
    "X-Title": "LLM Telephone (dev)",
}


def get_api_key() -> str:
    """Return the OpenRouter API key from the environment.

    Exits with code 2 and an actionable message if the key is not set.
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("OPENROUTER_API_KEY is not set; export it or add it to .env")
        sys.exit(2)
    return key


def __getattr__(name: str):
    if name == "OPENROUTER_API_KEY":
        return get_api_key()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

