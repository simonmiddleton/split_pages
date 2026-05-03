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
import io
import shutil

import numpy as np
from PIL import Image
import fitz  # PyMuPDF

import ocr_av

SCANS_DIR = Path("scans")
OUT_DIR = Path("pages")
PDF_PATH = Path("book.pdf")
TXT_PATH = Path("book.txt")
BOTTOM_CROP = 230       # pixels to trim off the bottom of each source scan

XTRIM = 60


def estimate_skew(
    img: Image.Image,
    angle_range: float = 3.0,
    coarse_step: float = 0.5,
    fine_step: float = 0.05,
    sample_max_dim: int = 1000,
    min_confidence: float = 1.05,
) -> float:
    """Estimate skew angle (degrees) of a primarily-text page using a
    horizontal-projection variance score. Negative means the image is rotated
    clockwise from upright; rotating by ``+angle`` straightens it.

    The score metric is the variance of horizontal pixel sums after rotating
    by the candidate angle: when text rows align with the image rows the
    sums alternate sharply between text-row peaks and white-row troughs,
    maximising variance.

    `angle_range` caps the search; pages on a flatbed scanner are rarely
    off by more than a couple of degrees and a tight cap protects against
    pathological pages (e.g. a fold-out map with diagonal street labels)
    hijacking the metric. `min_confidence` is the minimum ratio of the
    best score to the score at 0°; if the search can't find a clearly
    better angle the page is treated as already straight.
    """
    gray = img.convert("L")
    scale = sample_max_dim / max(gray.size)
    if scale < 1.0:
        gray = gray.resize(
            (int(gray.width * scale), int(gray.height * scale)),
            Image.BILINEAR,
        )
    arr = np.asarray(gray)
    # Binarise: text dark on light.
    binary = (arr < arr.mean() - 10).astype(np.uint8) * 255
    bin_img = Image.fromarray(binary)

    def score(angle: float) -> float:
        rot = bin_img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
        rows = np.asarray(rot).sum(axis=1, dtype=np.int64)
        return float(np.var(rows))

    coarse = np.arange(-angle_range, angle_range + coarse_step, coarse_step)
    best = max(coarse, key=score)
    lo = max(-angle_range, best - coarse_step)
    hi = min(angle_range, best + coarse_step)
    fine = np.arange(lo, hi + fine_step, fine_step)
    best_angle = float(max(fine, key=score))

    s0 = score(0.0)
    sb = score(best_angle)
    if s0 > 0 and sb / s0 < min_confidence:
        return 0.0
    return best_angle


def deskew(img: Image.Image, threshold: float = 0.1) -> tuple[Image.Image, float]:
    """Return a deskewed copy of `img` and the angle applied (degrees)."""
    angle = estimate_skew(img)
    if abs(angle) < threshold:
        return img, 0.0
    rotated = img.rotate(
        angle, resample=Image.BICUBIC, fillcolor="white", expand=False,
    )
    return rotated, angle


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

            # Pages were never placed perfectly square on the scanner — apply
            # a small projection-profile deskew to straighten the text lines.
            rotated, skew_angle = deskew(rotated)

            w, h = rotated.size
            mid = w // 2
            left = rotated.crop((XTRIM, 0, mid - XTRIM, h))
            right = rotated.crop((mid + XTRIM, 0, w - XTRIM, h))

            low_img = right if low_side == "right" else left
            high_img = left if high_side == "left" else right

            low_img.save(OUT_DIR / f"page_{low_page:03d}.jpg",
                         "JPEG", quality=95)
            high_img.save(OUT_DIR / f"page_{high_page:03d}.jpg",
                          "JPEG", quality=95)

        print(f"{src.name}: page {low_page:03d} ({low_side}) + "
              f"page {high_page:03d} ({high_side})  skew {skew_angle:+.2f}°")

    page_num = first_page + page_count
    for page in sorted(list(section_path.glob('*.jpg'))):
        shutil.copy(page, OUT_DIR / f"page_{page_num:03d}.jpg")
        page_num += 1

    return page_num

def scan_sections(sections_path):
    OUT_DIR.mkdir(exist_ok=True)

    first_page = 1
    for section in sorted(list(sections_path.iterdir())):
        first_page = scan_section(section, first_page)

def build_searchable_pdf(
    pages_dir: Path = OUT_DIR,
    pdf_path: Path = PDF_PATH,
    txt_path: Path = TXT_PATH,
    jpeg_quality: int | None = 80,
    reading_order: str = "columns",
) -> Path:
    """OCR every page JPEG with Apple Vision and assemble a searchable PDF.

    Each page consists of the original JPEG plus an invisible text layer
    (PDF text-rendering mode 3) positioned at the bounding boxes returned
    by VNRecognizeTextRequest. A plain-text sidecar of every page is
    also written to `txt_path`.

    `jpeg_quality` re-encodes each page image at the given JPEG quality
    before embedding it (the source pages are stored at quality 95). Pass
    ``None`` to embed the original JPEG bytes unchanged.

    `reading_order` controls how the sidecar text is stitched together:
    ``"columns"`` reads each column top-to-bottom (best for the book's
    flowing two-column directory pages); ``"rows"`` clusters fragments
    by y and reads each row left-to-right (best for tabular layouts).
    """
    if reading_order == "columns":
        order_fn = ocr_av.columns_from_annots
    elif reading_order == "rows":
        order_fn = ocr_av.rows_from_annots
    else:
        raise ValueError(
            f"reading_order must be 'columns' or 'rows', got {reading_order!r}"
        )
    page_files = sorted(pages_dir.glob("page_*.jpg"))
    if not page_files:
        raise RuntimeError(f"No pages found in {pages_dir}")

    doc = fitz.open()
    txt_chunks: list[str] = []

    for page_file in page_files:
        with Image.open(page_file) as img:
            w, h = img.size
            if jpeg_quality is None:
                img_bytes = page_file.read_bytes()
            else:
                buf = io.BytesIO()
                img.convert("RGB").save(
                    buf, "JPEG", quality=jpeg_quality, optimize=True
                )
                img_bytes = buf.getvalue()

        # 1pt = 1px so PDF page dimensions match the source image.
        page = doc.new_page(width=w, height=h)
        page.insert_image(fitz.Rect(0, 0, w, h), stream=img_bytes)

        annots = ocr_av.annotate(page_file)
        for text, _conf, (fx, fy, fw, fh) in annots:
            # Apple Vision: fractional coords, origin bottom-left.
            # PyMuPDF page: points, origin top-left, y grows downward.
            x_left = fx * w
            box_h_pt = fh * h
            baseline_y = (1.0 - fy) * h    # bottom of bbox ≈ text baseline
            fontsize = max(box_h_pt * 0.85, 1.0)
            page.insert_text(
                (x_left, baseline_y),
                text,
                fontname="helv",
                fontsize=fontsize,
                render_mode=3,             # invisible (searchable) text
            )

        txt_chunks.append(f"=== {page_file.name} ===")
        txt_chunks.append(order_fn(annots))
        txt_chunks.append("")
        print(f"OCR'd {page_file.name}: {len(annots)} fragments")

    doc.save(str(pdf_path), deflate=True, garbage=4, clean=True)
    doc.close()

    txt_path.write_text("\n".join(txt_chunks), encoding="utf-8")

    print(f"Wrote {pdf_path} ({len(page_files)} pages, "
          f"{pdf_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Wrote {txt_path} ({txt_path.stat().st_size / 1e3:.0f} KB)")
    return pdf_path


def main() -> None:
    scan_sections(SCANS_DIR)
    build_searchable_pdf()

if __name__ == "__main__":
    main()
