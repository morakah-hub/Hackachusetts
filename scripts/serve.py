r"""Start the Armstrong Studio server from any working directory (used by start.ps1 and for development).

    .venv\Scripts\python.exe scripts\serve.py [--port 8765] [--no-browser]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from armstrong.server import main  # noqa: E402

if __name__ == "__main__":
    main()
