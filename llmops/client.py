"""
Resilient completion:

  route -> call chosen tier (retry w/ backoff on transient errors)
        -> if it still fails: fall back to the other tier        (availability)
        -> if output fails the caller's validator on the FAST tier:
           escalate to the strong tier                            (quality)

Every attempt is logged via telemetry. When ROUTING_MODE is forced to a single
tier, fallback/escalation are disabled so single-model baselines stay pure.
"""
import os
import random
import time
from dataclasses import dataclass
from typing import Callable

import config
from . import telemetry
from .models import estimate_cost, estimate_tokens, get_model, other_tier
from .router import route, routing_mode

MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "2"))
BACKOFF_BASE_S = float(os.environ.get("LLM_BACKOFF_S", "1.0"))

Validator = Callable[[str], tuple[bool, str]]


@dataclass
class Completion:
    text: str
    model: str
    tier: str
    route_reason: str
    escalated: bool = False
    fell_back: bool = False


def _is_retryable(exc: Exception) -> bool:
    try:
        import openai
    except ImportError:
        return False
    return isinstance(exc, (openai.RateLimitError, openai.APITimeoutError,
                            openai.APIConnectionError, openai.InternalServerError))


_client = None


def _get_client():
    global _client
    if _client is None:
        _client = config.get_client()
    return _client


def _invoke(model: str, system_prompt: str, user_message: str,
            temperature: float, json_mode: bool) -> tuple[str, int, int]:
    """One raw model call -> (text, input_tokens, output_tokens)."""
    if config.DRY_RUN:
        from agents import dry_run_mock
        text = dry_run_mock.respond(system_prompt, user_message, json_mode)
        return text, estimate_tokens(system_prompt + user_message), estimate_tokens(text)

    kwargs = dict(
        model=model,
        temperature=temperature,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_message}],
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = _get_client().chat.completions.create(**kwargs)
    text = (resp.choices[0].message.content or "").strip()
    usage = getattr(resp, "usage", None)
    in_tok = getattr(usage, "prompt_tokens", None) or estimate_tokens(system_prompt + user_message)
    out_tok = getattr(usage, "completion_tokens", None) or estimate_tokens(text)
    return text, in_tok, out_tok


def _attempt(spec, system_prompt, user_message, temperature, json_mode,
             validator: Validator | None, meta: dict) -> tuple[str, bool]:
    """Call `spec` with retries. Returns (text, valid). Raises after retries are exhausted."""
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 2):
        t0 = time.perf_counter()
        try:
            text, in_tok, out_tok = _invoke(spec.name, system_prompt, user_message, temperature, json_mode)
        except Exception as exc:  # noqa: BLE001 - we classify below
            telemetry.record(model=spec.name, tier=spec.tier, attempt=attempt, status="error",
                             error=f"{type(exc).__name__}: {exc}"[:300],
                             latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                             input_tokens=0, output_tokens=0, est_cost_usd=0.0, **meta)
            last_exc = exc
            if _is_retryable(exc) and attempt <= MAX_RETRIES:
                time.sleep(BACKOFF_BASE_S * (2 ** (attempt - 1)) * (0.5 + random.random()))
                continue
            raise

        ok, why = (True, "") if validator is None else validator(text)
        telemetry.record(model=spec.name, tier=spec.tier, attempt=attempt,
                         status="ok" if ok else "invalid", error=None if ok else why,
                         latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                         input_tokens=in_tok, output_tokens=out_tok,
                         est_cost_usd=estimate_cost(spec.name, in_tok, out_tok), **meta)
        return text, ok
    raise last_exc  # pragma: no cover


def complete(task: str, system_prompt: str, user_message: str, *,
             temperature: float = 0.7, json_mode: bool = False,
             ctx: dict | None = None, prompt_version: str | None = None,
             validator: Validator | None = None) -> Completion:
    decision = route(task, ctx)
    spec = get_model(decision.tier)
    auto = routing_mode() == "auto"

    def meta(**extra):
        base = dict(task=task, route_reason=decision.reason, complexity=decision.complexity,
                    prompt_version=prompt_version, escalated_from=None, fallback_from=None)
        base.update(extra)
        return base

    fell_back = False
    try:
        text, ok = _attempt(spec, system_prompt, user_message, temperature, json_mode, validator, meta())
    except Exception:
        if not auto:
            raise
        fb = get_model(other_tier(spec.tier))
        text, ok = _attempt(fb, system_prompt, user_message, temperature, json_mode, validator,
                            meta(fallback_from=spec.name))
        spec, fell_back = fb, True

    escalated = False
    if not ok and auto and spec.tier == "fast":
        strong = get_model("strong")
        text, ok = _attempt(strong, system_prompt, user_message, temperature, json_mode, validator,
                            meta(escalated_from=spec.name))
        spec, escalated = strong, True

    return Completion(text=text, model=spec.name, tier=spec.tier,
                      route_reason=decision.reason, escalated=escalated, fell_back=fell_back)
