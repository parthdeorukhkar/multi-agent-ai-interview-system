"""
Model registry: two tiers (fast / strong) plus approximate pricing.

Groq's free tier means your real bill is $0, but we still track the cost the
same tokens would have on a paid plan. That is what makes routing decisions
measurable ("this change cut estimated cost 60% at equal eval accuracy").

Prices are USD per 1M tokens and are APPROXIMATE - check
https://groq.com/pricing and edit PRICING if they change.
"""
import os
from dataclasses import dataclass

TIERS = ("fast", "strong")

# model name -> (usd per 1M input tokens, usd per 1M output tokens)
PRICING = {
    "llama-3.1-8b-instant": (0.05, 0.08),
    "llama-3.3-70b-versatile": (0.59, 0.79),
}
_DEFAULT_PRICE = (0.59, 0.79)  # unknown model: assume strong-tier pricing


@dataclass(frozen=True)
class ModelSpec:
    name: str
    tier: str


def _name_for(tier: str) -> str:
    if tier == "fast":
        return os.environ.get("GROQ_MODEL_FAST", "llama-3.1-8b-instant")
    # GROQ_MODEL is the original single-model variable; keep honouring it.
    return os.environ.get("GROQ_MODEL_STRONG") or os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


def get_model(tier: str) -> ModelSpec:
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {TIERS}")
    return ModelSpec(name=_name_for(tier), tier=tier)


def other_tier(tier: str) -> str:
    return "strong" if tier == "fast" else "fast"


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p_in, p_out = PRICING.get(model, _DEFAULT_PRICE)
    return (input_tokens * p_in + output_tokens * p_out) / 1_000_000


def estimate_tokens(text: str) -> int:
    """Rough fallback (~4 chars/token) when the API doesn't report usage."""
    return max(1, len(text) // 4)
