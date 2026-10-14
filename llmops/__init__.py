"""
LLMOps layer: model routing, call telemetry (latency / tokens / cost),
prompt versioning, and resilient completion (retry -> fallback -> escalation).

Everything the agents need goes through `llmops.client.complete()`.
"""
