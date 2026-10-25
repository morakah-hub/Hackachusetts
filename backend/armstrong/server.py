"""Entry point: `python -m armstrong.server [--port 8765] [--no-browser]`.

The simulation worker (MuJoCo + OpenGL renderer contexts) runs on the main thread; Uvicorn serves the API and
the built frontend from a background thread, bound to 127.0.0.1 only.
"""

from __future__ import annotations

import argparse
import threading
import time
import webbrowser

import uvicorn

from .api import ROOT, Studio, create_app
from .rules import RobotRules, World
from .sim import SimWorker
from .store import StudioStore


def main() -> None:
    ap = argparse.ArgumentParser(description="Armstrong Studio local workcell server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser")
    ap.add_argument("--data", default=str(ROOT / "runtime" / "studio"), help="where the scene and uploads are kept")
    args = ap.parse_args()
    if args.host not in ("127.0.0.1", "localhost"):
        raise SystemExit("Armstrong Studio is a local single-user app; bind to 127.0.0.1 only.")
    rules, world = RobotRules(), World()
    store = StudioStore(args.data, rules, world)
    worker = SimWorker(store)
    app = create_app(Studio(worker, store, world, rules))
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="warning")
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None  # signals are handled by the main thread below
    t = threading.Thread(target=server.run, name="armstrong-http", daemon=True)
    t.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    url = f"http://{args.host}:{args.port}/"
    print("", flush=True)
    print(f"  Armstrong Studio is running at {url}", flush=True)
    print("  Keep this window open while you use the app. Close it (or press Ctrl+C) to stop.", flush=True)
    print("", flush=True)
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        worker.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.should_exit = True
        t.join(3)


if __name__ == "__main__":
    main()
