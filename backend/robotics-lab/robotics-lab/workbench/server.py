"""Local HTTP server for the workbench UI (v0.7): Python standard library only, offline.

    .\\.venv\\Scripts\\python.exe workbench.py            # http://127.0.0.1:8765, opens the browser

Serves the static single-page UI (workbench/static) and a small JSON API. Handler threads never touch
MuJoCo objects: they call `Session` methods, which enqueue actions for the worker thread or return copies.
Binds to 127.0.0.1 only.
"""

from __future__ import annotations

import json
import mimetypes
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from workbench import inspect as insp
from workbench.api import RequestError, RunRequest, capabilities
from workbench.commands import CommandConsole
from workbench.runs import list_runs
from workbench.session import Session

STATIC = Path(__file__).resolve().parent / "static"


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return str(o)


def _clean(o):
    """Replace NaN/inf (not valid JSON) by None, recursively."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating,)):
        return _clean(float(o))
    if isinstance(o, (np.integer, np.bool_)):
        return o.item()
    return o


class Workbench:
    """Session + command console + HTTP server, with an orderly shutdown."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, session: Session | None = None):
        self.session = session or Session()
        self.console = CommandConsole(self.session)
        handler = type("Handler", (_Handler,), {"wb": self})
        self.httpd = ThreadingHTTPServer((host, port), handler)
        self.httpd.daemon_threads = True
        self.url = f"http://{host}:{self.httpd.server_address[1]}/"
        self._thread: threading.Thread | None = None
        self.on_quit = None  # set by the launcher; called by POST /api/quit

    def serve_background(self) -> None:
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="workbench-http", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.session.close()
        if self._thread is not None:
            self._thread.join(5)


class _Handler(BaseHTTPRequestHandler):
    wb: Workbench
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet console
        pass

    # ------------------------------------------------------------------ helpers

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, code: int = 200) -> None:
        self._send(code, json.dumps(_clean(data), default=_json_default).encode(), "application/json")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    # ------------------------------------------------------------------ routes

    def do_GET(self):  # noqa: N802
        url = urllib.parse.urlparse(self.path)
        q = {k: v[-1] for k, v in urllib.parse.parse_qs(url.query).items()}
        s = self.wb.session
        try:
            if url.path in ("/", "/index.html"):
                return self._static("index.html")
            if url.path.startswith("/static/"):
                return self._static(url.path[len("/static/"):])
            if url.path == "/api/capabilities":
                return self._json(capabilities())
            if url.path == "/api/status":
                st = s.snapshot_status()
                st["messages"] = [m for m in s.messages if m["t"] > float(q.get("since", 0))]
                return self._json(st)
            if url.path == "/api/frame":
                fid, jpg = s.latest_frame()
                if not jpg:
                    return self._send(204, b"", "image/jpeg")
                return self._send(200, jpg, "image/jpeg", {"X-Frame-Id": str(fid)})
            if url.path == "/api/telemetry":
                return self._json(insp.telemetry(s.recording(), max_points=int(q.get("max", 1500))))
            if url.path == "/api/runs":
                return self._json(list_runs(s.runs_dir))
            if url.path == "/api/inspect" and q.get("kind") == "experiments":
                from workbench.experiments import panel
                return self._json({"sections": panel(s)})
            if url.path == "/api/preflight":
                return self._json({"checks": insp.preflight(s.request)})
            if url.path == "/api/inspect":
                return self._json(insp.inspect(s.recording(), q.get("kind", "motion"), q.get("t"), q))
            if url.path == "/api/explain":
                return self._json(insp.explain(s.recording(), q.get("t")))
            if url.path == "/api/help":
                return self._json({"text": self.wb.console.help_text()})
            return self._json({"error": "not found"}, 404)
        except RequestError as exc:
            return self._json({"error": str(exc)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            return None

    def do_POST(self):  # noqa: N802
        url = urllib.parse.urlparse(self.path)
        s = self.wb.session
        try:
            body = self._body()
            if url.path == "/api/configure":
                req = s.configure(**body)
                return self._json({"request": req.to_dict(), "summary": req.summary()})
            if url.path == "/api/request":
                req = s.set_request(RunRequest.from_dict(body))
                return self._json({"request": req.to_dict(), "summary": req.summary()})
            if url.path == "/api/action":
                action = body.pop("action")
                s.act(action, **body)
                return self._json({"ok": True, "action": action})
            if url.path == "/api/command":
                return self._json(self.wb.console.execute(body.get("text", "")))
            if url.path == "/api/quit":
                self._json({"ok": True, "text": "closing the workbench"})
                if self.wb.on_quit is not None:
                    self.wb.on_quit()
                return None
            if url.path == "/api/scan":
                return self._json(insp.reach_scan(s.request))
            if url.path == "/api/sandbox":
                return self._json(insp.sandbox(body))
            if url.path == "/api/experiment":
                return self._json(self.wb.console.start_experiment(body.get("name", ""), body.get("cases")))
            return self._json({"error": "not found"}, 404)
        except RequestError as exc:
            return self._json({"error": str(exc)}, 400)
        except (ValueError, KeyError, TypeError) as exc:
            return self._json({"error": f"bad request: {exc}"}, 400)
        except (BrokenPipeError, ConnectionResetError):
            return None

    def _static(self, name: str):
        path = (STATIC / name).resolve()
        if STATIC.resolve() not in path.parents or not path.is_file():
            return self._json({"error": "not found"}, 404)
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix == ".js":
            ctype = "text/javascript"
        return self._send(200, path.read_bytes(), ctype + ("; charset=utf-8" if ctype.startswith("text") else ""))
