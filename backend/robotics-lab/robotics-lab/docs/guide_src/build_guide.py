"""Build docs/Robotics_Lab_Complete_Technical_Guide.pdf from the HTML sources in this folder.

Sources (all editable):
    chapters/*.html   one HTML fragment per chapter/appendix, in file-name order
    guide.css         typography, page layout (CSS paged media), boxes, tables
    figures/          figures (SVG/PNG); make_figures.py regenerates the computed ones
    vendor/tex-svg.js MathJax 3.2.2 (offline copy; renders $...$ and \\[...\\] to SVG)

Steps:
    1. Assemble one HTML document: title page, generated table of contents, chapters.
       Chapters (<h1 class="chapter">), sections (<h2>), subsections (<h3>), figures
       (<figure> with <figcaption>) and tables (<table> with <caption>) are numbered here;
       empty links <a class="ref" href="#id"></a> are filled with "Figure 7.2", "Section 7.3", ...
    2. Print it to PDF with headless Chrome through the DevTools protocol, after MathJax
       has finished typesetting (Page.printToPDF, CSS page size, document outline).
    3. Read the page of every heading from the PDF's named destinations (pypdf) and print
       again with the page numbers filled into the table of contents. Repeat until stable.

Run from the repo root (Chrome or Edge and the `websockets` package are needed; pypdf is
optional and only fills the TOC page numbers):
    .\\.venv\\Scripts\\python.exe docs\\guide_src\\build_guide.py
"""

import asyncio
import base64
import html
import json
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent
OUT_HTML = HERE / "_build" / "Robotics_Lab_Complete_Technical_Guide.html"
OUT_PDF = DOCS / "Robotics_Lab_Complete_Technical_Guide.pdf"
BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            "google-chrome", "chromium", "msedge"]

TITLE = "Robotics Lab"
SUBTITLE = "Complete Technical Guide"


# ------------------------------------------------------------------ assembly


def slug(text: str) -> str:
    s = re.sub(r"<[^>]+>", "", text)
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s[:60] or "sec"


def number_document(body: str):
    """Number headings, figures and tables; return (html, toc entries, labels for refs)."""
    toc, labels = [], {}
    chapter, appendix, section, subsection, fig, tab = 0, 0, 0, 0, 0, 0
    current = ""          # "7" or "A"
    used_ids = set()
    out, pos = [], 0
    token = re.compile(r"<h1([^>]*)>(.*?)</h1>|<h2([^>]*)>(.*?)</h2>|<h3([^>]*)>(.*?)</h3>"
                       r"|<div class=\"part\"([^>]*)>(.*?)</div>|<figcaption([^>]*)>(.*?)</figcaption>"
                       r"|<caption([^>]*)>(.*?)</caption>|<figure([^>]*)>|<table([^>]*)>", re.S)
    pending_fig_id, pending_tab_id = None, None

    def get_id(attrs: str, fallback: str) -> tuple[str, str]:
        m = re.search(r'\bid="([^"]+)"', attrs)
        if m:
            ident = m.group(1)
            attrs = attrs.replace(m.group(0), "")
        else:
            ident = fallback
            k = 2
            while ident in used_ids:
                ident = f"{fallback}-{k}"
                k += 1
        used_ids.add(ident)
        return ident, attrs

    for m in token.finditer(body):
        out.append(body[pos:m.start()])
        pos = m.end()
        if m.group(2) is not None:  # h1
            attrs, text = m.group(1), m.group(2).strip()
            is_app = 'class="appendix"' in attrs
            if is_app:
                appendix += 1
                current = "ABCDEFGHIJ"[appendix - 1]
                label, prefix = f"Appendix {current}", f"Appendix {current}"
            else:
                chapter += 1
                current = str(chapter)
                label, prefix = f"Chapter {current}", f"Chapter {current}"
            section = subsection = fig = tab = 0
            ident, attrs = get_id(attrs, f"ch-{current.lower()}")
            labels[ident] = label
            toc.append({"level": 1, "id": ident, "num": current, "title": text, "appendix": is_app})
            out.append(f'<h1 id="{ident}"{attrs}><span class="chapnum">{prefix}</span>'
                       f'<span class="chaptitle">{text}</span></h1>')
        elif m.group(4) is not None:  # h2
            attrs, text = m.group(3), m.group(4).strip()
            section += 1
            subsection = 0
            num = f"{current}.{section}"
            ident, attrs = get_id(attrs, f"s{num.replace('.', '-')}-{slug(text)}")
            labels[ident] = f"Section {num}"
            toc.append({"level": 2, "id": ident, "num": num, "title": text})
            out.append(f'<h2 id="{ident}"{attrs}><span class="secnum">{num}</span> {text}</h2>')
        elif m.group(6) is not None:  # h3
            attrs, text = m.group(5), m.group(6).strip()
            subsection += 1
            num = f"{current}.{section}.{subsection}"
            ident, attrs = get_id(attrs, f"s{num.replace('.', '-')}-{slug(text)}")
            labels[ident] = f"Section {num}"
            out.append(f'<h3 id="{ident}"{attrs}><span class="secnum">{num}</span> {text}</h3>')
        elif m.group(8) is not None:  # part divider
            attrs, text = m.group(7), m.group(8).strip()
            ident, attrs = get_id(attrs, f"part-{slug(text)}")
            toc.append({"level": 0, "id": ident, "title": re.sub(r"<small>.*", "", text, flags=re.S).strip()})
            out.append(f'<div class="part" id="{ident}"{attrs}>{text}</div>')
        elif m.group(10) is not None:  # figcaption
            fig += 1
            num = f"{current}.{fig}"
            if pending_fig_id:
                labels[pending_fig_id] = f"Figure {num}"
            out.append(f'<figcaption{m.group(9)}><span class="fignum">Figure {num}.</span> {m.group(10).strip()}</figcaption>')
            pending_fig_id = None
        elif m.group(12) is not None:  # table caption
            tab += 1
            num = f"{current}.{tab}"
            if pending_tab_id:
                labels[pending_tab_id] = f"Table {num}"
            out.append(f'<caption{m.group(11)}><span class="tabnum">Table {num}.</span> {m.group(12).strip()}</caption>')
            pending_tab_id = None
        elif m.group(13) is not None:  # <figure ...>
            attrs = m.group(13)
            idm = re.search(r'\bid="([^"]+)"', attrs)
            pending_fig_id = idm.group(1) if idm else None
            if pending_fig_id:
                used_ids.add(pending_fig_id)
            out.append(m.group(0))
        else:  # <table ...>
            attrs = m.group(14)
            idm = re.search(r'\bid="([^"]+)"', attrs)
            pending_tab_id = idm.group(1) if idm else None
            if pending_tab_id:
                used_ids.add(pending_tab_id)
            out.append(m.group(0))
    out.append(body[pos:])
    text = "".join(out)

    missing = []

    def fill(m):
        ident = m.group(1)
        if ident not in labels:
            missing.append(ident)
            return f'<a class="ref" href="#{ident}">??{ident}</a>'
        return f'<a class="ref" href="#{ident}">{labels[ident]}</a>'

    text = re.sub(r'<a class="ref" href="#([^"]+)"></a>', fill, text)
    if missing:
        print("WARNING: unresolved references:", sorted(set(missing)))
    return text, toc, labels


def toc_html(toc: list[dict], pages: dict[str, int] | None) -> str:
    rows = []
    for e in toc:
        pg = "" if not pages else str(pages.get(e["id"], ""))
        if e["level"] == 0:
            rows.append(f'<li class="toc-part"><a href="#{e["id"]}"><span class="t">{e["title"]}</span>'
                        f'<span class="p">{pg}</span></a></li>')
        elif e["level"] == 1:
            lab = f'Appendix {e["num"]}' if e.get("appendix") else e["num"]
            cls = "toc-ch toc-app" if e.get("appendix") else "toc-ch"
            rows.append(f'<li class="{cls}"><a href="#{e["id"]}"><span class="n">{lab}</span>'
                        f'<span class="t">{e["title"]}</span><span class="p">{pg}</span></a></li>')
        else:
            rows.append(f'<li class="toc-sec"><a href="#{e["id"]}"><span class="n">{e["num"]}</span>'
                        f'<span class="t">{e["title"]}</span><span class="p">{pg}</span></a></li>')
    return '<nav class="toc" id="contents"><h1 class="toc-title">Contents</h1><ul>' + "\n".join(rows) + "</ul></nav>"


def assemble(pages: dict[str, int] | None = None) -> tuple[str, list[dict]]:
    chapters = sorted((HERE / "chapters").glob("*.html"))
    front = [c for c in chapters if c.name.startswith("00")]
    body_files = [c for c in chapters if not c.name.startswith("00")]
    body = "\n".join(c.read_text(encoding="utf-8") for c in body_files)
    body, toc, _ = number_document(body)
    front_html = "\n".join(c.read_text(encoding="utf-8") for c in front)
    css = (HERE / "guide.css").read_text(encoding="utf-8")
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{TITLE}: {SUBTITLE}</title>
<style>{css}</style>
<script>
window.MathJax = {{
  tex: {{ inlineMath: [['$', '$']], displayMath: [['\\\\[', '\\\\]']], processEscapes: true,
         tags: 'none' }},
  svg: {{ fontCache: 'global', scale: 0.95 }},
  startup: {{ pageReady: () => MathJax.startup.defaultPageReady().then(() => {{ window.__mathDone = true; }}) }}
}};
</script>
<script src="../vendor/tex-svg.js"></script>
</head>
<body>
{front_html.replace("<!--TOC-->", toc_html(toc, pages))}
{body}
</body></html>"""
    return doc, toc


# ------------------------------------------------------------------ printing


def find_browser() -> str:
    for b in BROWSERS:
        if Path(b).is_file() or shutil.which(b):
            return b if Path(b).is_file() else shutil.which(b)
    raise SystemExit("Chrome or Edge not found")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _print(url: str, pdf_path: Path, browser: str) -> None:
    import websockets

    port = free_port()
    profile = tempfile.mkdtemp(prefix="guide_chrome_")
    proc = subprocess.Popen([browser, "--headless=new", "--disable-gpu", f"--remote-debugging-port={port}",
                             f"--user-data-dir={profile}", "--allow-file-access-from-files",
                             "--no-first-run", "--no-default-browser-check", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                targets = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json").read())
                page = next(t for t in targets if t.get("type") == "page")
                break
            except Exception:
                time.sleep(0.1)
        else:
            raise RuntimeError("browser did not start")
        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=None) as ws:
            counter = 0

            async def call(method, **params):
                nonlocal counter
                counter += 1
                my = counter
                await ws.send(json.dumps({"id": my, "method": method, "params": params}))
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("id") == my:
                        if "error" in msg:
                            raise RuntimeError(f"{method}: {msg['error']}")
                        return msg["result"]

            await call("Page.enable")
            await call("Page.navigate", url=url)
            t0 = time.time()
            while True:
                r = await call("Runtime.evaluate", expression="window.__mathDone === true", returnByValue=True)
                if r["result"].get("value"):
                    break
                if time.time() - t0 > 300:
                    raise RuntimeError("MathJax did not finish within 300 s")
                await asyncio.sleep(0.5)
            await asyncio.sleep(1.0)  # images / layout settle
            res = await call("Page.printToPDF", preferCSSPageSize=True, printBackground=True,
                             displayHeaderFooter=False, generateDocumentOutline=True, generateTaggedPDF=True)
            pdf_path.write_bytes(base64.b64decode(res["data"]))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(profile, ignore_errors=True)


def print_pdf(html_path: Path, pdf_path: Path) -> None:
    asyncio.run(_print(html_path.as_uri(), pdf_path, find_browser()))


def heading_pages(pdf_path: Path, ids: list[str]) -> dict[str, int] | None:
    try:
        from pypdf import PdfReader
    except ImportError:
        print("pypdf not installed: table of contents printed without page numbers")
        return None
    reader = PdfReader(str(pdf_path))
    pages = {}
    for name, dest in reader.named_destinations.items():
        key = name.lstrip("/")
        try:
            pages[key] = reader.get_destination_page_number(dest) + 1
        except Exception:
            pass
    found = {i: pages[i] for i in ids if i in pages}
    if len(found) < len(ids):
        print(f"WARNING: no page for {len(ids) - len(found)} headings: {[i for i in ids if i not in found][:8]}")
    return found


def main() -> None:
    OUT_HTML.parent.mkdir(exist_ok=True)
    pages = None
    for attempt in range(4):
        doc, toc = assemble(pages)
        OUT_HTML.write_text(doc, encoding="utf-8")
        t0 = time.time()
        print_pdf(OUT_HTML, OUT_PDF)
        new = heading_pages(OUT_PDF, [e["id"] for e in toc])
        print(f"pass {attempt + 1}: printed in {time.time() - t0:.0f} s")
        if new is None or new == pages:
            break
        pages = new
    try:
        from pypdf import PdfReader
        print(f"{OUT_PDF}: {len(PdfReader(str(OUT_PDF)).pages)} pages")
    except ImportError:
        print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
