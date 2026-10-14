# AI Mock Interview Coach

A multi-agent system that conducts a real, adaptive, speech-to-speech mock interview
and delivers structured coaching feedback at the end.

## Setup

1. Clone/unzip this repo and `cd` into it.
2. Create a virtual environment and activate it:
    python -m venv venv
    venv\Scripts\Activate.ps1 # Windows PowerShell
3. Install dependencies:
    pip install -r requirements.txt
4. Copy `.env.example` to `.env` and add a free Groq API key (get one at https://console.groq.com/keys):
   GROQ_API_KEY=your_key_here

## Running it

Full real speech-to-speech interview (mic in, spoken questions out):
    python main.py

Test modes (no mic/speaker needed, useful for quick checks):
    $env:INTERVIEW_TEXT_MODE=1; python main.py # real AI, typed answers
    $env:INTERVIEW_DRY_RUN=1; python main.py # no API key/network at all, mocked agents

After every session the console (and the saved transcript) shows run metrics:
LLM calls, tokens, estimated cost, p50/p95 latency, and how many calls went to
the fast vs strong model. See **LLMOps** below.

You'll be asked for a target role, an optional resume (paste a file path to a
.pdf/.docx/.txt, or type a quick background snippet, or skip), and a focus area
(behavioral / technical / case / mixed). The interview runs 5-7 turns, then a
coaching report is generated and the full session is saved to `transcripts/`.

NOTE: Please first enable microphone permissions for working of speech-to-speech interview mode
1.Go to Setting --> Privacy & Security --> microphone --> enable Let Desktop apps access microphone.

## Architecture

Three agents, each with a genuinely different role, output format, and point in
the flow — not three calls in a chain:

- **Interviewer** (`agents/interviewer.py`, `prompts/interviewer_prompt.txt`)
  Drives the conversation. Holds the running history. Before each question, it
  receives the Evaluator's private signal on the previous answer and uses it to
  decide whether to probe deeper (weak answer), move on / raise difficulty
  (strong answer), or pivot (off-topic / "I don't know"). Speaks in natural
  interview dialogue only — never reveals scores or system internals.

- **Evaluator** (`agents/evaluator.py`, `prompts/evaluator_prompt.txt`)
  Runs silently after every answer. Never speaks to the candidate. Scores each
  answer 1-5 across five dimensions (correctness, depth, structure, relevance,
  communication) plus a verdict, returned as strict JSON. Explicitly instructed
  to handle vague/hedging answers, off-topic answers, "I don't know", and
  partial correctness without collapsing to all-or-nothing scoring.

- **Coach** (`agents/coach.py`, `prompts/coach_prompt.txt`)
  Runs once, after the interview ends. Takes the full transcript plus every
  Evaluator score and produces a structured markdown report: overall
  impression, strengths, gaps, concrete practice items, and a score summary —
  each point grounded in a specific turn, not generic advice.

### Orchestration (`orchestrator.py`)

for each turn (5-7 total):
Interviewer.next_question(turn, last_eval) -> spoken via TTS
candidate speaks -> transcribed via STT
Evaluator.score(question, answer) -> silent JSON, feeds back into next loop iteration
Coach.generate_report(full_history, all_scores) -> printed + saved to transcripts/

The orchestrator is the only place that knows about turn count, agent
sequencing, and termination — agents themselves are pure text-in/text-out and
don't know about each other.

## LLMOps: operating it, not just running it

Every LLM call goes through one function (`config.chat` -> `llmops/client.py`), which gives the whole system:

| Concern | What it does | Where |
|---|---|---|
| **Telemetry** | One JSONL record per call: task, model, route reason, prompt version, latency, input/output tokens, estimated cost, status, retry/escalation/fallback flags | `llmops/telemetry.py` -> `logs/llm_calls.jsonl` |
| **Cost & latency reporting** | Per-session summary printed + saved in the transcript; `python -m llmops.report` aggregates the whole log by model and task (p50/p95) | `llmops/report.py` |
| **Prompt versioning** | Each prompt file gets a content-hash version (`evaluator@3fa9c1d2`) stamped on every call and every eval run, so a quality/cost change can be traced to the prompt edit that caused it | `llmops/prompts.py` |
| **Resilience** | Timeouts, exponential-backoff retries on 429/5xx/network errors, then fallback to the other model tier | `llmops/client.py` |
| **Strict output contracts** | The Evaluator's JSON is schema-validated (five integer scores 1-5, known verdict); invalid output is never silently trusted | `agents/evaluator.py` |

Costs use Groq's paid-tier prices (`llmops/models.py`, approximate) even though the free tier bills $0, so that "what would this cost at scale" is measurable.

## Automatic model routing

Different calls need different amounts of intelligence, so the system no longer uses one 70B model for everything.
`llmops/router.py` picks a tier per call with a small, deterministic, unit-tested policy:

| Task | Goes to **fast** (Llama 3.1 8B) when... | Goes to **strong** (Llama 3.3 70B) when... |
|---|---|---|
| Evaluator | answer is trivial / "I don't know"; or short behavioral answer | technical / case answers (correctness needs domain knowledge); long answers |
| Interviewer | warm-up opener; simple probe or pivot after a weak / no-answer turn | needs to raise difficulty after a strong answer (technical/case), or personalize from a resume |
| Coach | never | always (runs once; it is the user-facing deliverable) |

Two safety nets mean a routing mistake costs latency, not correctness:
1. **Escalation** - if the fast model returns invalid JSON, or "unclear" for an answer long enough to judge, the call is re-run on the strong model.
2. **Fallback** - if a tier is down / rate-limited after retries, the other tier is used.

Every decision's reason is logged. `ROUTING_MODE=strong|fast` forces a single model so you can A/B the router against the one-model baseline (`python -m evals.run_evals --compare-routing`).

## Agent evaluations

"It seems to work" is replaced by a repeatable test suite in `evals/` that measures the agents against labelled data:

| Suite | What is measured | Data |
|---|---|---|
| `evaluator` | verdict accuracy, score-bound accuracy (per-dimension min/max), JSON validity, and **ordering** (a good answer must outscore a bad answer to the same question - catches a scorer that collapses to "3") | 13 labelled answers incl. hedging, "I don't know", off-topic, STT garble, wrong, partial, STT artifacts; 3 ordering pairs |
| `interviewer` | one question only, **no leaking** of scores/internals, no markdown (it is spoken), no mid-interview feedback, doesn't repeat itself, probes the gap after a weak answer, asks to repeat on garbled input, closes politely on the last turn | 7 scenarios |
| `coach` | all sections present, bullet counts, cites specific turns, no platitudes, and **hallucination checks**: reported score averages must equal the real averages, and every quoted phrase must exist in the transcript | 2 fixed transcripts |
| `router` | the right model tier is chosen for each task/context (the routing "tool selection" check) | 18 cases |
| `e2e` | task completion: scripted candidate personas (strong / weak / "I don't know") run the whole pipeline; all turns completed, valid scores, complete report, within a cost budget, and strong persona scores > weak personas | 3 personas |
| *(optional)* `--judge` | LLM-as-judge (strong model) rubric scores for follow-up quality and coaching specificity | same cases |

```bash
python -m evals.run_evals --dry-run             # offline, no key: tests the harness + plumbing against a mock LLM
python -m evals.run_evals                       # live, real models (needs GROQ_API_KEY)
python -m evals.run_evals --compare             # exit 1 if any quality metric regressed vs evals/baselines/<mode>.json
python -m evals.run_evals --update-baseline     # accept current results as the new baseline
python -m evals.run_evals --compare-routing     # strong-only vs auto routing: accuracy vs cost vs latency (live)
```

Each run writes `evals/results/<timestamp>.{json,md}` (metrics, failing cases with reasons, model + prompt versions, cost/latency).
**Regression gating:** change a prompt or the routing policy, rerun with `--compare`, and a drop in any metric fails the run.
Baselines are stored per mode because dry-run numbers describe the mock, not the model - **run `python -m evals.run_evals --update-baseline` once with your key to create `evals/baselines/live.json`.**

### Tests & CI
`python -m pytest` runs unit tests for the router, retry/fallback/escalation logic, telemetry and cost maths, the eval checks themselves, and a full dry-run interview. GitHub Actions (`.github/workflows/ci.yml`) runs tests + offline evals on every push; live evals can be triggered manually with a `GROQ_API_KEY` secret.

## Speech-to-speech

- **STT**: `faster-whisper` (local, free, no API key) — records from the mic
  until you press Enter or pause for ~5s, then transcribes.
- **TTS**: `edge-tts` (free, no API key) — synthesizes and plays each question
  aloud via `pygame` before recording starts.
- `INTERVIEW_TEXT_MODE=1` swaps both for typed input / printed output, so the
  agent logic can be tested independently of audio hardware.

## Key design decisions & tradeoffs

- **Groq (free tier) over paid APIs** — keeps the whole project runnable with
  zero cost, at the tradeoff of using open models (Llama 3.3 70B / 3.1 8B) rather than
  a frontier one; prompt discipline plus evals compensate for this.
- **Rule-based router over an LLM router** — routing runs on every call, so it must be
  free, instant, deterministic and testable. Tradeoff: heuristics can misjudge difficulty;
  escalation-on-invalid-output and the routing evals bound that risk.
- **Eval checks are mostly deterministic** (regex / schema / arithmetic) with an *optional*
  LLM judge — deterministic checks are reproducible and free, so they can gate CI;
  the judge covers what they can't, at a cost and with its own noise.
- **Bounded turn count (5-7, randomized)** rather than letting the Interviewer
  decide freely when to stop — keeps sessions predictable and comparable,
  tradeoff being slightly less agent autonomy.
- **Evaluator score is never shown to the candidate mid-interview** — mirrors a
  real interview (you don't get live scoring), and prevents the Interviewer's
  adaptive behavior from being second-guessed or gamed by the candidate.
- **DRY_RUN / TEXT_MODE toggles** — built in specifically to let the multi-agent
  orchestration be verified independently of API keys or audio hardware, since
  that's the part actually being evaluated.
- **Local STT (faster-whisper) over a cloud STT API** — keeps the "free" and
  "real speech-to-speech" requirements both true simultaneously, at the cost of
  slightly lower transcription accuracy than a paid cloud model.

## Example transcripts

See `transcripts/` for auto-saved sessions. Three representative examples:

- `transcripts/<strong-session>.md` — candidate gave specific, structured answers
- `transcripts/<weak-session>.md` — candidate gave vague/short answers, showing
  the Interviewer's adaptive follow-up behavior
- `transcripts/<edge-case-session>.md` — candidate said "I don't know" /
  answered off-topic, showing the Evaluator's `no_answer`/`off_topic` handling
## Browser frontend (React)

A full browser interface is available in `frontend/`. It supports spoken questions, microphone recording with local Whisper transcription, resume upload, adaptive interview turns, and a final coaching report. See [`FRONTEND_SETUP.md`](FRONTEND_SETUP.md) for Windows setup and run commands. Start the API with `uvicorn api:app --reload --host 127.0.0.1 --port 8000`, then run the frontend with `cd frontend; npm install; npm run dev`.
