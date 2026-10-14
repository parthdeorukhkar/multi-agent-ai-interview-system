import os
import sys
from pathlib import Path

# Tests never call a real LLM and never write the real call log.
os.environ["INTERVIEW_DRY_RUN"] = "1"
os.environ["INTERVIEW_TEXT_MODE"] = "1"
os.environ["LLM_LOG"] = "0"
os.environ.pop("ROUTING_MODE", None)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
