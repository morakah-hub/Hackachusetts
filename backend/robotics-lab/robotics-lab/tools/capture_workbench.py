r"""Screenshots and a short demo of the workbench UI, driven through headless Chrome/Edge (v0.7).

    .\.venv\Scripts\python.exe tools\capture_workbench.py --out results\workbench\screens

Starts its own Workbench on a free port, drives it with the same HTTP API and console commands a user would
type, and captures the browser page with the DevTools protocol (Page.captureScreenshot). Writes PNG
screenshots and an animated WebP of the page (`workbench_demo.webp`). Needs Chrome or Edge and the
`websockets` package; everything is local.
"""

import argparse
import asyncio
import base64
import io
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from workbench.server import Workbench  # noqa: E402

BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]


def post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def status(base):
    return json.loads(urllib.request.urlopen(base + "api/status", timeout=30).read())


def wait_state(base, states, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = status(base)
        if st["state"] in states:
            return st
        time.sleep(0.3)
    raise RuntimeError("timeout waiting for " + str(states))


class Page:
    def __init__(self, ws):
        self.ws, self.n = ws, 0

    async def call(self, method, **params):
        self.n += 1
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == self.n:
                return msg.get("result", {})

    async def js(self, expr):
        return await self.call("Runtime.evaluate", expression=expr, awaitPromise=True)

    async def shot(self) -> Image.Image:
        r = await self.call("Page.captureScreenshot", format="png")
        return Image.open(io.BytesIO(base64.b64decode(r["data"]))).convert("RGB")


async def main_async(args):
    import websockets
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wb = Workbench(port=0)
    wb.serve_background()
    base = wb.url
    browser = next((b for b in BROWSERS if Path(b).is_file()), None) or shutil.which("chrome")
    prof = tempfile.mkdtemp()
    proc = subprocess.Popen([browser, "--headless=new", "--disable-gpu", f"--user-data-dir={prof}",
                             "--remote-debugging-port=9333", "--window-size=1600,1000", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    frames = []
    try:
        for _ in range(50):
            try:
                tabs = json.loads(urllib.request.urlopen("http://127.0.0.1:9333/json", timeout=2).read())
                page = next(t for t in tabs if t["type"] == "page")
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.2)
        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=50_000_000) as ws:
            p = Page(ws)
            await p.call("Emulation.setDeviceMetricsOverride", width=1600, height=1000, deviceScaleFactor=1, mobile=False)
            await p.call("Page.navigate", url=base)
            await asyncio.sleep(2.5)

            async def snap(name=None, hold=1):
                img = await p.shot()
                for _ in range(hold):
                    frames.append(img.resize((1200, 750)))
                if name:
                    img.save(out / f"{name}.png")

            async def cmd(text, pause=0.8):
                await p.js(f"runCommand({json.dumps(text)})")
                await asyncio.sleep(pause)

            await snap("01_start", 3)
            await cmd("load obstacle", 1.5)
            await p.js("document.getElementById('btn-scan').click()")
            await asyncio.sleep(4.0)
            await snap("01b_workspace_scan_obstacle", 3)
            await cmd("load pick; controller mpc; observer on; payload 1.15", 1.5)
            await snap("02_configured", 3)
            await cmd("run")
            t_end = time.time() + 30
            while time.time() < t_end:
                st = status(base)
                await snap(None)
                if st["state"] in ("finished", "failed"):
                    break
                if st["phase"] == "transport" and not (out / "03_running_transport.png").exists():
                    await snap("03_running_transport")
                await asyncio.sleep(0.35)
            wait_state(base, {"finished", "failed"})
            await asyncio.sleep(1.5)
            await snap("04_finished_result", 4)
            await cmd("replay; seek 6.0", 2.0)
            await p.js("selectTab('mpc')")
            await asyncio.sleep(2.5)
            await snap("05_replay_mpc_inspector", 5)
            await p.js("document.getElementById('tab-body').scrollTop = 900")
            await asyncio.sleep(0.6)
            await snap("06_mpc_objective_constraints", 4)
            await p.js("selectTab('dynamics')")
            await asyncio.sleep(2.0)
            await snap("07_dynamics_observer", 4)
            await p.js("selectTab('explain')")
            await asyncio.sleep(1.5)
            await snap("08_explain_moment", 4)
            await p.js("selectTab('motion')")
            await asyncio.sleep(1.5)
            await snap("09_motion_jacobian", 3)
            await p.js("selectTab('planning')")
            await asyncio.sleep(1.5)
            await snap("10_planning_clearance", 3)
            await p.js("selectTab('sandbox')")
            await asyncio.sleep(2.5)
            await snap("11_sandbox", 3)
            await cmd("compare observer", 1.0)
            t_end = time.time() + 120
            while time.time() < t_end:
                await snap(None)
                st = status(base)
                if "experiment 'observer' done" in " ".join(m["text"] for m in st["messages"]):
                    break
                await asyncio.sleep(1.0)
            await asyncio.sleep(2.5)
            await p.js("selectTab('experiments')")
            await asyncio.sleep(2.5)
            await snap("12_experiment_observer", 5)
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
        wb.close()
    from PIL import _webp
    enc = _webp.WebPAnimEncoder((1200, 750), 0, 0, False, 3, 5, False, False)
    t = 0
    for f in frames:
        enc.add(f.getim(), t, False, 70, 100, 4)
        t += 400
    enc.add(None, t, False, 70, 100, 0)
    (out / "workbench_demo.webp").write_bytes(enc.assemble("", "", ""))
    print(f"wrote {len(list(out.glob('*.png')))} screenshots and workbench_demo.webp ({len(frames)} frames, "
          f"{t / 1000:.0f} s at 2.5 frames/s) to {out}; worker alive after close: {wb.session.alive}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/workbench/screens")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
