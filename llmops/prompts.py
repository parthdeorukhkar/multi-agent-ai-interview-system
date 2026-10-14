"""
Prompt versioning. A prompt's version is a short content hash, e.g.
"evaluator@3fa9c1d2". It's stamped on every logged LLM call and every eval run,
so a quality/cost change can always be traced to the prompt edit that caused it
without maintaining version numbers by hand.
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


@dataclass(frozen=True)
class Prompt:
    name: str
    text: str
    version: str


def load_prompt(name: str) -> Prompt:
    text = (PROMPTS_DIR / f"{name}_prompt.txt").read_text()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return Prompt(name=name, text=text, version=f"{name}@{digest}")


def all_versions() -> dict:
    return {n: load_prompt(n).version for n in ("interviewer", "evaluator", "coach")}
