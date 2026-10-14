"""
Text-to-speech using edge-tts — free, no API key, natural voices.
"""
import os
import asyncio
import tempfile
import time

TEXT_MODE = os.environ.get("INTERVIEW_TEXT_MODE", "0") == "1"
VOICE = os.environ.get("INTERVIEW_TTS_VOICE", "en-US-GuyNeural")


async def _synthesize(text: str, out_path: str):
    import edge_tts
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save(out_path)


def _play(path: str):
    try:
        import pygame
        import time
        pygame.mixer.init()
        sound = pygame.mixer.Sound(path)
        duration = sound.get_length()
        sound.play()
        time.sleep(duration + 0.4)  # small buffer past actual audio length
        pygame.mixer.quit()
    except Exception as e:
        print(f"[voice: playback unavailable ({e}), text shown above]")


def speak(text: str):
    print(f"\nInterviewer: {text}\n")
    if TEXT_MODE:
        return
    tmp_path = None
    try:
        tmp_path = tempfile.mktemp(suffix=".mp3")
        asyncio.run(_synthesize(text, tmp_path))
        _play(tmp_path)
    except Exception as e:
        print(f"[voice: TTS unavailable ({e}), continuing in text mode for this turn]")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)