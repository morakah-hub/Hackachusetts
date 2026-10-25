"""Draw the Armstrong Studio logo (robot arm placing a block) as launcher/logo.png and launcher/logo.ico."""
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
S = 1024  # drawn large, then downsampled for smooth edges


def draw() -> Image.Image:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # background: rounded square with a vertical gradient
    bg = Image.new("RGBA", (S, S))
    gd = ImageDraw.Draw(bg)
    for y in range(S):
        t = y / S
        c = (int(38 + 10 * t), int(45 - 8 * t), int(54 - 10 * t), 255)
        gd.line([(0, y), (S, y)], fill=c)
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, S - 1, S - 1], radius=220, fill=255)
    img.paste(bg, (0, 0), mask)
    teal, white, shade, orange = (46, 196, 182, 255), (236, 239, 242, 255), (150, 160, 170, 255), (240, 140, 50, 255)
    # table line and the teal work-area arc
    d.rounded_rectangle([150, 800, 874, 836], radius=18, fill=(90, 98, 108, 255))
    d.arc([170, 560, 854, 1040], start=200, end=340, fill=teal, width=34)
    # base
    d.rounded_rectangle([250, 700, 430, 800], radius=26, fill=white)
    # links: shoulder (340, 690) -> elbow (420, 330) -> wrist (700, 300)
    w = 92
    d.line([(340, 690), (420, 330)], fill=white, width=w)
    d.line([(420, 330), (700, 300)], fill=white, width=w)
    for (x, y), r in (((340, 690), 70), ((420, 330), 66), ((700, 300), 56)):
        d.ellipse([x - r, y - r, x + r, y + r], fill=white)
        d.ellipse([x - r * 0.42, y - r * 0.42, x + r * 0.42, y + r * 0.42], fill=teal)
    # hand pointing down with two fingers around an orange block
    d.rounded_rectangle([640, 350, 760, 430], radius=20, fill=white)
    d.rounded_rectangle([630, 430, 668, 560], radius=14, fill=shade)
    d.rounded_rectangle([732, 430, 770, 560], radius=14, fill=shade)
    d.rounded_rectangle([660, 470, 740, 560], radius=14, fill=orange)
    # stacked blocks on the table (the "assembly")
    d.rounded_rectangle([560, 720, 680, 800], radius=12, fill=(61, 139, 253, 255))
    d.rounded_rectangle([700, 720, 820, 800], radius=12, fill=(61, 139, 253, 255))
    d.rounded_rectangle([540, 680, 840, 724], radius=12, fill=(242, 177, 52, 255))
    return img


def main() -> None:
    big = draw()
    big.resize((512, 512), Image.LANCZOS).save(HERE / "logo.png")
    big.resize((256, 256), Image.LANCZOS).save(HERE / "logo.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48),
                                                                         (64, 64), (128, 128), (256, 256)])
    print("wrote", HERE / "logo.png", "and", HERE / "logo.ico")


if __name__ == "__main__":
    main()
