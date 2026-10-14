"""
Speech-to-text using faster-whisper — fully local, free, no API key.

Recording stops on WHICHEVER comes first:
  - you press Enter (manual stop — most reliable, recommended)
  - 2 seconds of silence after you've spoken (auto stop)
  - max_seconds elapses (safety cap)
"""
import os
import sys
import tempfile
import threading
import time
import wave

TEXT_MODE = os.environ.get("INTERVIEW_TEXT_MODE", "0") == "1"

_model = None


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel("base.en", device="cpu", compute_type="int8")
    return _model


def np_concat(chunks):
    import numpy as np
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 1), dtype="int16")


def _wait_for_enter(stop_flag: dict):
    """Runs in a background thread; sets stop_flag['manual'] = True the moment
    the user presses Enter, regardless of what recording is doing."""
    try:
        input()
    except EOFError:
        pass
    stop_flag["manual"] = True


def _record_until_silence(max_seconds: int = 90, silence_threshold: float = 500.0,
                           silence_duration: float = 10.0, sample_rate: int = 16000) -> str:
    import sounddevice as sd
    import numpy as np

    print(">>> RECORDING NOW. Speak your answer.")
    print(">>> Press ENTER at any time when you're done (or just pause for 10s).")

    chunk_ms = 100
    chunk_samples = int(sample_rate * chunk_ms / 1000)
    silence_chunks_needed = int(silence_duration * 1000 / chunk_ms)

    stop_flag = {"manual": False}
    listener = threading.Thread(target=_wait_for_enter, args=(stop_flag,), daemon=True)
    listener.start()

    buffer = []
    silent_run = 0
    recorded_any_speech = False
    start_time = time.time()

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16") as stream:
        while True:
            if stop_flag["manual"]:
                print(">>> Got it — stopped recording.")
                break
            if time.time() - start_time > max_seconds:
                print(">>> Max recording time reached, stopping.")
                break

            data, _ = stream.read(chunk_samples)
            buffer.append(data.copy())
            volume = float(np.abs(data).mean())
            if volume > silence_threshold:
                recorded_any_speech = True
                silent_run = 0
            else:
                silent_run += 1
            if recorded_any_speech and silent_run >= silence_chunks_needed:
                print(">>> Detected pause, stopping automatically.")
                break

    audio = np_concat(buffer)
    tmp_path = tempfile.mktemp(suffix=".wav")
    with wave.open(tmp_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return tmp_path

def _countdown(seconds: int):
    for i in range(seconds, 0, -1):
        print(f"Recording your response in {i}s...", end="\r", flush=True)
        time.sleep(1)
    print(" " * 40, end="\r")  # clear the countdown line

def listen_for_answer() -> str:
    if TEXT_MODE:
        return input("Your answer (text mode): ").strip()

    # brief pause so the mic doesn't pick up the tail end of TTS playback
    _countdown(5)

    wav_path = _record_until_silence()
    try:
        print("[transcribing...]")
        model = _get_model()
        segments, _ = model.transcribe(wav_path, language="en", vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text if text else "[no speech detected]"
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)