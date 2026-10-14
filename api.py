"""FastAPI bridge for the browser-based AI Mock Interview Coach."""
import os, tempfile, uuid
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from agents.interviewer import InterviewerAgent
from agents.evaluator import EvaluatorAgent
from agents.coach import CoachAgent
from llmops import telemetry
from llmops.prompts import all_versions
from resume_reader import read_resume
import config

app = FastAPI(title="AI Mock Interview Coach API", version="1.0.0")
_local_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
]

_deployed_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv("FRONTEND_URL", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(dict.fromkeys(_local_origins + _deployed_origins)),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS: dict[str, dict] = {}

class StartRequest(BaseModel):
    target_role: str = Field(min_length=2, max_length=120)
    focus_area: str = Field(default="mixed", pattern="^(behavioral|technical|case|mixed)$")
    resume_text: str = Field(default="", max_length=12000)
    total_turns: int = Field(default=5, ge=3, le=7)

class AnswerRequest(BaseModel):
    answer: str = Field(min_length=1, max_length=12000)

@app.get("/api/health")
def health():
    return {"ok": True, "dry_run": config.DRY_RUN, "api_key_configured": bool(config.GROQ_API_KEY)}

@app.post("/api/resume")
async def upload_resume(file: UploadFile = File(...)):
    name = Path(file.filename or "resume.txt").name
    ext = Path(name).suffix.lower()
    if ext not in {".pdf", ".docx", ".txt"}:
        raise HTTPException(400, "Upload a PDF, DOCX, or TXT resume.")
    content = await file.read()
    if len(content) > 8 * 1024 * 1024:
        raise HTTPException(413, "Resume must be smaller than 8 MB.")
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
            f.write(content); tmp = f.name
        text = read_resume(tmp)
        if not text.strip(): raise HTTPException(422, "No readable text found in this resume.")
        return {"filename": name, "text": text[:10000], "characters": len(text)}
    except HTTPException: raise
    except Exception as exc: raise HTTPException(422, f"Could not read resume: {exc}")
    finally:
        if tmp and os.path.exists(tmp): os.remove(tmp)

@app.post("/api/interviews")
def start_interview(req: StartRequest):
    if not config.GROQ_API_KEY and not config.DRY_RUN:
        raise HTTPException(503, "GROQ_API_KEY is missing. Add it to your .env file, then restart the backend.")
    sid = str(uuid.uuid4())
    telemetry.set_session(sid); mark = telemetry.mark()
    interviewer = InterviewerAgent(req.target_role.strip(), req.focus_area, req.resume_text.strip())
    evaluator = EvaluatorAgent(req.target_role.strip(), req.focus_area)
    coach = CoachAgent(req.target_role.strip(), req.focus_area)
    state = {"id": sid, "target_role": req.target_role.strip(), "focus_area": req.focus_area, "resume_text": req.resume_text[:3000], "total_turns": req.total_turns, "turn": 1, "interviewer": interviewer, "evaluator": evaluator, "coach": coach, "history": [], "scores": [], "last_eval": None, "question": None, "mark": mark, "timestamp": __import__('datetime').datetime.now().isoformat(timespec="seconds"), "report": None}
    try:
        state["question"] = interviewer.next_question(1, req.total_turns, None)
    except Exception as exc:
        raise HTTPException(502, f"Could not generate the first question: {exc}")
    SESSIONS[sid] = state
    return {"session_id": sid, "question": state["question"], "turn": 1, "total_turns": req.total_turns, "target_role": state["target_role"], "focus_area": req.focus_area}

def get_session(sid: str):
    state = SESSIONS.get(sid)
    if not state: raise HTTPException(404, "Interview session not found. Start a new interview.")
    return state

@app.post("/api/interviews/{sid}/answer")
def submit_answer(sid: str, req: AnswerRequest):
    s = get_session(sid)
    if s["report"] is not None: raise HTTPException(409, "This interview is already complete.")
    answer = req.answer.strip()
    if not answer: raise HTTPException(400, "Please record or type an answer first.")
    question = s["question"]
    try:
        s["interviewer"].record_answer(question, answer)
        score = s["evaluator"].score(question, answer)
        s["history"].append({"question": question, "answer": answer})
        s["scores"].append(score); s["last_eval"] = score
        if s["turn"] >= s["total_turns"]:
            s["report"] = s["coach"].generate_report(s["history"], s["scores"])
            from orchestrator import _save_transcript
            summary = telemetry.summarize(telemetry.records_since(s["mark"]))
            result = {"complete": True, "report": s["report"], "history": s["history"], "scores": s["scores"], "llm_summary": summary, "prompt_versions": all_versions(), "target_role": s["target_role"], "focus_area": s["focus_area"], "timestamp": s["timestamp"]}
            try: _save_transcript({**result, "resume_snippet": s["resume_text"]}, verbose=False)
            except Exception: pass
            return result
        s["turn"] += 1
        s["question"] = s["interviewer"].next_question(s["turn"], s["total_turns"], s["last_eval"])
        return {"complete": False, "question": s["question"], "turn": s["turn"], "total_turns": s["total_turns"], "answered": len(s["history"])}
    except Exception as exc:
        # If scoring fails, don't advance the interview or silently lose the candidate's answer.
        if len(s["history"]) < len(s["interviewer"].history): s["interviewer"].history.pop()
        raise HTTPException(502, f"Could not process this answer: {exc}")

@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    data = await file.read()
    if not data: raise HTTPException(400, "Audio recording was empty.")
    if len(data) > 25 * 1024 * 1024: raise HTTPException(413, "Audio must be smaller than 25 MB.")
    suffix = Path(file.filename or "answer.webm").suffix or ".webm"
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(data); tmp = f.name
        from speech.stt import _get_model
        model = _get_model()
        segments, _ = model.transcribe(tmp, language="en", vad_filter=True)
        transcript = " ".join(seg.text.strip() for seg in segments).strip()
        if not transcript: raise HTTPException(422, "No speech detected. Try again or type your answer instead.")
        return {"text": transcript}
    except HTTPException: raise
    except Exception as exc: raise HTTPException(500, f"Audio transcription failed: {exc}")
    finally:
        if tmp and os.path.exists(tmp): os.remove(tmp)

@app.get("/api/interviews/{sid}/transcript")
def transcript(sid: str):
    s = get_session(sid)
    return {"history": s["history"], "scores": s["scores"], "complete": s["report"] is not None}
