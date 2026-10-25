r"""Render the ArmStrong logo (frontend/public/armstrong-logo.svg) into packaging/ArmStrong.ico for the exe.

    backend\.venv\Scripts\python.exe packaging\make_icon.py

Headless Chrome or Edge rasterizes the SVG and Pillow writes a multi-size icon. build-exe.ps1 uses the committed
icon, so this only needs to run again when the logo changes.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
SVG = HERE.parent / "frontend" / "public" / "armstrong-logo.svg"
OUT = HERE / "ArmStrong.ico"
BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]
SIZE = 512
ICON_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]


def main() -> None:
    browser = next((b for b in BROWSERS if Path(b).is_file()), None)
    if browser is None:
        sys.exit("Chrome or Edge is needed to render the logo.")
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "logo.png"
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        f"--user-data-dir={Path(tmp) / 'profile'}", "--default-background-color=00000000",
                        f"--window-size={SIZE},{SIZE}", f"--screenshot={png}", SVG.as_uri()],
                       check=True, timeout=90, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        img = Image.open(png).convert("RGBA")
    if img.size != (SIZE, SIZE):
        img = img.resize((SIZE, SIZE), Image.LANCZOS)
    # Rounded corners as in the SVG (rx 30 of 128), antialiased by drawing the mask large and shrinking it.
    big = Image.new("L", (4 * SIZE, 4 * SIZE), 0)
    ImageDraw.Draw(big).rounded_rectangle((0, 0, 4 * SIZE - 1, 4 * SIZE - 1), radius=4 * SIZE * 30 // 128, fill=255)
    img.putalpha(big.resize((SIZE, SIZE), Image.LANCZOS))
    img.resize((256, 256), Image.LANCZOS).save(OUT, sizes=ICON_SIZES)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
