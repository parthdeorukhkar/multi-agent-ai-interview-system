"""
Central config + LLM client. Uses Groq's free-tier API (console.groq.com) via
its OpenAI-compatible endpoint, so no paid API key is required.
"""
import os
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

MIN_TURNS = 5
MAX_TURNS = 7

DRY_RUN = os.environ.get("INTERVIEW_DRY_RUN", "0") == "1"


LLM_TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT_S", "30"))


def get_client() -> OpenAI:
    if not GROQ_API_KEY and not DRY_RUN:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at "
            "https://console.groq.com/keys, or set INTERVIEW_DRY_RUN=1 to test "
            "without calling the API."
        )
    # max_retries=0: retries/backoff/fallback are handled (and logged) by llmops.client
    return OpenAI(api_key=GROQ_API_KEY or "dry-run", base_url=GROQ_BASE_URL,
                  timeout=LLM_TIMEOUT_S, max_retries=0)


def chat(system_prompt: str, user_message: str, temperature: float = 0.7,
         json_mode: bool = False, *, task: str = "generic", ctx: dict | None = None,
         prompt_version: str | None = None, validator=None) -> str:
    """
    Single entry point for every LLM call. Routing (fast vs strong model),
    retries, fallback, escalation and telemetry all live in llmops.client;
    this wrapper keeps the original signature working for existing callers.
    """
    from llmops.client import complete
    return complete(task, system_prompt, user_message, temperature=temperature,
                    json_mode=json_mode, ctx=ctx, prompt_version=prompt_version,
                    validator=validator).text