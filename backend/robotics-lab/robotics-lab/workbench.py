r"""Robotics Lab Workbench (v0.7): local browser UI for running and inspecting simulations.

    .\.venv\Scripts\python.exe workbench.py                # serves http://127.0.0.1:8765 and opens the browser
    .\.venv\Scripts\python.exe workbench.py --port 8800 --no-browser
    Workbench.bat                                          # same, double-clickable

Stop with Ctrl+C in this window or the Quit button in the page: the simulation worker, its renderers and the
server are closed before the process exits. Everything runs locally; no network access is needed.
"""

import argparse
import signal
import threading
import webbrowser

from workbench.server import Workbench


def main() -> None:
    ap = argparse.ArgumentParser(description="Robotics Lab Workbench (local)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    wb = Workbench(port=args.port)
    stop = threading.Event()
    wb.on_quit = stop.set
    wb.serve_background()
    print(f"Robotics Lab Workbench running at {wb.url}  (Ctrl+C or the Quit button to stop)")
    if not args.no_browser:
        webbrowser.open(wb.url)
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    try:
        while not stop.wait(0.5):
            pass
    finally:
        print("closing: stopping the simulation worker and the server ...")
        wb.close()
        print(f"closed (worker alive: {wb.session.alive})")


if __name__ == "__main__":
    main()
