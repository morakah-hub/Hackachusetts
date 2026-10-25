r"""ArmStrong as one Windows executable: the PyInstaller entry point (build-exe.ps1 in the repo root builds it).

It runs the same server as studio.py. Inside the exe it serves the bundled frontend, keeps the layout, library and
uploads in %LOCALAPPDATA%\ArmStrong, reuses an ArmStrong that is already running, and opens the browser once the
server answers. It always binds to 127.0.0.1 because the server has no authentication.

    ..\..\.venv\Scripts\python.exe armstrong.py        # the same, from source
    ArmStrong.exe --port 8771 --no-browser
"""

import argparse
import os
import socket
import sys
import threading
import time
import traceback
import urllib.request
import webbrowser
from pathlib import Path

HOST = "127.0.0.1"
DEFAULT_PORT = 8770
FROZEN = bool(getattr(sys, "frozen", False))
# Health checks go straight to this computer, never through a configured proxy.
_LOCAL = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _use_bundle_paths() -> None:
    """In the exe: the frontend comes from the bundle, and user data lives in %LOCALAPPDATA%\\ArmStrong."""
    if not FROZEN:
        return
    os.environ.setdefault("STUDIO_FRONTEND_DIST", str(Path(getattr(sys, "_MEIPASS")) / "frontend_dist"))
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    os.environ.setdefault("STUDIO_DATA_DIR", str(Path(base) / "ArmStrong"))


def _healthy(port: int, timeout: float = 1.0) -> bool:
    try:
        with _LOCAL.open(f"http://{HOST}:{port}/api/health", timeout=timeout) as r:
            return r.status == 200 and b'"ok"' in r.read()
    except Exception:  # noqa: BLE001 - not (yet) an ArmStrong server
        return False


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((HOST, port))
        except OSError:
            return False
    return True


def _announce(port: int, open_browser: bool, timeout: float = 600.0) -> None:
    """Wait until the server answers, then say so and open the browser."""
    deadline = time.time() + timeout
    while not _healthy(port, timeout=2.0):
        if time.time() > deadline:
            return
        time.sleep(0.5)
    url = f"http://{HOST}:{port}"
    print(f"ArmStrong is ready at {url}", flush=True)
    if open_browser:
        webbrowser.open(url)


def main() -> int:
    ap = argparse.ArgumentParser(description="ArmStrong: robot arm simulation (local)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    _use_bundle_paths()
    # ArmStrong never renders with MuJoCo (the browser draws the scene), so skip MuJoCo's OpenGL/GLFW setup.
    os.environ.setdefault("MUJOCO_GL", "disable")
    port = args.port
    if not _port_free(port):
        if _healthy(port):
            print(f"ArmStrong is already running at http://{HOST}:{port}")
            if not args.no_browser:
                webbrowser.open(f"http://{HOST}:{port}")
            return 0
        port = next((p for p in range(port + 1, port + 50) if _port_free(p)), 0)
        if not port:
            raise RuntimeError(f"no free port between {args.port} and {args.port + 49}")
        print(f"Port {args.port} is busy; using {port} instead.")
    print("ArmStrong: robot arm simulation")
    print("Starting. The first start can take up to a minute.", flush=True)

    import uvicorn  # imported after the messages so they appear at once

    from studio.server import create_app
    from studio.settings import StudioSettings

    settings = StudioSettings.load()
    print(f"Your layout, objects and uploads are saved in {settings.data_dir}")
    print("Close this window or press Ctrl+C to quit.", flush=True)
    threading.Thread(target=_announce, args=(port, not args.no_browser), daemon=True).start()
    uvicorn.run(create_app(settings), host=HOST, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception:  # noqa: BLE001 - show the error before the console window closes
        traceback.print_exc()
        if FROZEN:
            input("ArmStrong could not start. Press Enter to close this window.")
        sys.exit(1)
