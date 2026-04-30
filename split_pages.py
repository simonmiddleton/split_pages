#!/usr/bin/env python3
"""Rotate, split and reorder a section of disassembled book scans.

The `scans/` directory contains 12 JPEGs (`Scan.jpeg`, `Scan 1.jpeg` … `Scan 11.jpeg`).
Each scan is portrait, with two book pages stacked top-to-bottom and the text
rotated 90° counter-clockwise (so the image must be rotated 90° clockwise to
read it).

The scans are of one folded signature of 12 leaves (24 pages, numbered
iii–xxvi). For scan index k (0-based) the top page is the lower-numbered page
(k + 3) and the bottom page is the higher-numbered one (26 − k).

Even-indexed scans (outer faces of each folded sheet) were placed on the
scanner rotated 90° counter-clockwise, so they need to be rotated 90°
clockwise to read. Odd-indexed scans (inner faces, where the sheet was
flipped over) were placed rotated 90° clockwise, so they need to be rotated
90° counter-clockwise. After the appropriate rotation, the original "top"
page is the lower-numbered page; for even scans it ends up on the right of
the rotated image and for odd scans on the left.

Each scan also has a ~130-pixel-tall scanner artifact along its bottom edge.
This strip is cropped from the source image before rotation and splitting so
that the two halves are equal width and contain full page content.

Output: `pages/page_03.jpg` … `pages/page_26.jpg`, in reading order.
"""

from pathlib import Path
from PIL import Image

SCANS_DIR = Path("scans")
OUT_DIR = Path("pages")
#NUM_SCANS = 12          # Scan.jpeg + Scan 1.jpeg … Scan 11.jpeg
#FIRST_PAGE = 3          # page iii
#LAST_PAGE = 26          # page xxvi
BOTTOM_CROP = 230       # pixels to trim off the bottom of each source scan


def scan_section(section_path, first_page) -> int:

    def scan_path(index: int) -> Path:
        """Return the file path for scan index `index` (0-based)."""
        return section_path / ("Scan.jpeg" if index == 0 else f"Scan {index}.jpeg")

    num_scans = len(list(section_path.glob('*.jpeg')))
    page_count = num_scans * 2

    for k in range(num_scans):
        src = scan_path(k)
        low_page = first_page + k       # lower-numbered page (was on top)
        high_page = first_page + page_count - 1 - k       # higher-numbered page (was on bottom)

        with Image.open(src) as img:
            # Trim the scanner artifact along the bottom edge so the page
            # content is centred and the halves are equal width.
            w0, h0 = img.size
            trimmed = img.crop((0, 0, w0, h0 - BOTTOM_CROP))

            # Even scans need a 90° CW rotation; odd scans were placed on the
            # scanner the other way up and need a 90° CCW rotation.
            if k % 2 == 0:
                rotated = trimmed.rotate(-90, expand=True)
                # CW: top-of-original → right of rotated.
                low_side, high_side = "right", "left"
            else:
                rotated = trimmed.rotate(90, expand=True)
                # CCW: top-of-original → left of rotated.
                low_side, high_side = "left", "right"

            w, h = rotated.size
            mid = w // 2
            left = rotated.crop((0, 0, mid, h))
            right = rotated.crop((mid, 0, w, h))

            low_img = right if low_side == "right" else left
            high_img = left if high_side == "left" else right

            low_img.save(OUT_DIR / f"page_{low_page:03d}.jpg",
                         "JPEG", quality=95)
            high_img.save(OUT_DIR / f"page_{high_page:03d}.jpg",
                          "JPEG", quality=95)

        print(f"{src.name}: page {low_page:03d} ({low_side}) + "
              f"page {high_page:03d} ({high_side})")

    return page_count

def scan_sections(sections_path):
    first_page = 1
    for section in sorted(list(sections_path.iterdir())):
        first_page += scan_section(section, first_page)

def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    scan_sections(SCANS_DIR)


if __name__ == "__main__":
    main()
