"""Render a saved `pandas-doctor run` transcript into the README GIF and a still card.

The transcript is real output (see docs/hero-run.txt); this only draws it like a
terminal, revealing it line by line, so the demo can't drift from what the tool prints.

    uv run --with pillow python docs/make_assets.py docs/hero-run.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BG = (24, 26, 33)
FG = (220, 223, 228)
DIM = (130, 136, 150)
GREEN = (126, 211, 132)
CYAN = (110, 190, 230)
YELLOW = (229, 192, 123)
BOLD = (255, 255, 255)
PAD, LINE_H, FONT_SIZE, WIDTH = 28, 26, 17, 1180

FONT_CANDIDATES = [
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/Library/Fonts/JetBrainsMono-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]


def font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, FONT_SIZE)
    return ImageFont.load_default()


def colour(line: str) -> tuple[int, int, int]:
    s = line.strip()
    if s.startswith("$"):
        return BOLD
    if "faster" in s and ("after" in s or "once" in s):
        return GREEN
    if s.startswith("result identical"):
        return GREEN
    if line.startswith("       ") and not s.startswith("("):
        return CYAN
    if s.startswith(("(", "Run again", "note:")):
        return DIM
    if (
        s.startswith(("pandas-doctor run", "Rewrites", "Where the time"))
        or s[:2].rstrip(".").isdigit()
    ):
        return BOLD
    if "slower" in s or "not faster" in s:
        return YELLOW
    return FG


def frame(
    lines: list[str], total_rows: int, f: ImageFont.FreeTypeFont | ImageFont.ImageFont, width: int
) -> Image.Image:
    height = PAD * 2 + LINE_H * total_rows + 30
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        draw.ellipse((PAD + i * 22, 14, PAD + i * 22 + 12, 26), fill=c)
    for row, line in enumerate(lines):
        draw.text((PAD, PAD + 22 + row * LINE_H), line, font=f, fill=colour(line))
    return img


def main() -> None:
    transcript = Path(sys.argv[1]).read_text().rstrip("\n").splitlines()
    out_dir = Path(__file__).resolve().parent
    f = font()
    width = max(WIDTH, int(max(f.getlength(line) for line in transcript)) + 2 * PAD)
    frames, durations = [], []
    for n in range(1, len(transcript) + 1):
        frames.append(frame(transcript[:n], len(transcript), f, width))
        durations.append(900 if transcript[n - 1].startswith("$") else 140)
    durations[-1] = 4000
    frames[0].save(
        out_dir / "demo.gif",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    frames[-1].save(out_dir / "card.png")
    print(f"wrote {out_dir / 'demo.gif'} ({len(frames)} frames) and {out_dir / 'card.png'}")


if __name__ == "__main__":
    main()
