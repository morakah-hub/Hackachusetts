r"""ArmStrong: browser app for placing objects on a reach-sized table and running simple tasks.

    ..\..\.venv\Scripts\python.exe studio.py                 # http://127.0.0.1:8770 (opens the browser)
    ..\..\.venv\Scripts\python.exe studio.py --port 8771 --no-browser

Serves the API under /api and the built frontend (<repo>/frontend/dist) at /. For frontend development run
`npm run dev` in <repo>/frontend, which proxies /api to this server. Local only: no authentication.
"""

import argparse
import threading
import webbrowser

import uvicorn

from studio.server import create_app
from studio.settings import StudioSettings


def main() -> None:
    ap = argparse.ArgumentParser(description="ArmStrong (local)")
    ap.add_argument("--host", default="127.0.0.1", help="interface to bind (default: this computer only)")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    settings = StudioSettings.load()
    app = create_app(settings)
    url = f"http://{args.host}:{args.port}"
    print(f"ArmStrong at {url}  (Ctrl+C to stop; data in {settings.data_dir})")
    if args.host not in ("127.0.0.1", "localhost"):
        print("WARNING: the studio has no authentication; anyone who can reach this address can control it.")
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
