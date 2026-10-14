# InterviewOS frontend

A clean React + Vite frontend for the existing AI Mock Interview Coach. It preserves the project's interviewer, evaluator, coach, routing, and transcript pipeline and adds a browser UI.

## Requirements
- Python 3.10+
- Node.js 18+
- A Groq API key (unless running in demo mode)

## 1. Backend setup (Windows PowerShell)
From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and set `GROQ_API_KEY=your_actual_key`. Do not put API keys in the frontend. To test UI wiring without model calls, set `INTERVIEW_DRY_RUN=1` in `.env` (the interview content will be deterministic mock output).

Start the backend in terminal 1:

```powershell
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Backend health check: http://127.0.0.1:8000/api/health

## 2. Frontend setup
In terminal 2:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL Vite prints (normally http://localhost:5173). Use Chrome or Edge and allow microphone access. Browser speech synthesis reads questions aloud; microphone recordings are transcribed by the project's local `faster-whisper` model. The first transcription may take a little longer while the model downloads/loads. You can always type answers instead.

## Features
- Role and focus selection; 3–7 questions
- PDF, DOCX, and TXT resume upload and extraction
- Spoken questions and browser microphone recording
- Local Whisper transcription with editable transcript before submission
- Adaptive interview questions using the existing evaluator feedback
- Final coaching report, expandable transcript, and Markdown export
- Health status, clear error states, and no API key exposed to the browser

## Notes
- Microphone access works on `localhost` or HTTPS, not arbitrary insecure network origins.
- The backend currently stores active sessions in memory. Restarting the backend ends active sessions; completed transcripts are saved under `transcripts/`.
- The browser frontend needs the backend running at the same time.
