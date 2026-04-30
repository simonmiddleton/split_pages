#!/usr/bin/env python3
"""Rotate, split and reorder a section of disassembled book scans.

The `scans/` directory contains 12 JPEGs (`Scan.jpeg`, `Scan 1.jpeg` … `Scan 11.jpeg`).
Each scan is portrait, with two book pages stacked top-to-bottom and the text
rotated 90° counter-clockwise (so the image must be rotated 90° clockwise to
read it).

The scans are of one folded signature of 12 leaves (24 pages, numbered
iii–xxvi). For scan index k (0-based) the top page is the lower-numbered page
(k + 3) and the bottom page is the higher-numbered one (26 − k). After a 90°
clockwise rotation, the original "top" page becomes the right-hand page and
the "bottom" page becomes the left-hand page.

Output: `pages/page_03.jpg` … `pages/page_26.jpg`, in reading order.
"""

from pathlib import Path
from PIL import Image

SCANS_DIR = Path("scans")
OUT_DIR = Path("pages")
NUM_SCANS = 12          # Scan.jpeg + Scan 1.jpeg … Scan 11.jpeg
FIRST_PAGE = 3          # page iii
LAST_PAGE = 26          # page xxvi


def scan_path(index: int) -> Path:
    """Return the file path for scan index `index` (0-based)."""
    return SCANS_DIR / ("Scan.jpeg" if index == 0 else f"Scan {index}.jpeg")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    for k in range(NUM_SCANS):
        src = scan_path(k)
        low_page = FIRST_PAGE + k       # lower-numbered page (was on top)
        high_page = LAST_PAGE - k       # higher-numbered page (was on bottom)

        with Image.open(src) as img:
            # Rotate 90° clockwise so the text is upright.
            rotated = img.rotate(-90, expand=True)
            w, h = rotated.size
            mid = w // 2

            left = rotated.crop((0, 0, mid, h))     # higher-numbered page
            right = rotated.crop((mid, 0, w, h))    # lower-numbered page

            right.save(OUT_DIR / f"page_{low_page:02d}.jpg",
                       "JPEG", quality=95)
            left.save(OUT_DIR / f"page_{high_page:02d}.jpg",
                      "JPEG", quality=95)

        print(f"{src.name}: page {low_page:02d} (right) + page {high_page:02d} (left)")


if __name__ == "__main__":
    main()
