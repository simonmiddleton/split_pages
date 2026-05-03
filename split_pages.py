#!/usr/bin/env python3
"""Rebuild the book from its disassembled scans.

End-to-end pipeline:

  1. **Split** every scanned signature in `scans/` into single pages
     (`scan_sections` → `scan_section`).
  2. **OCR** each page with Apple Vision and **build a searchable PDF**
     plus a plain-text sidecar (`build_searchable_pdf`).

Layout of the source material
-----------------------------
The book was disassembled into folded signatures, each scanned as a
sequence of JPEGs (`Scan.jpeg`, `Scan 1.jpeg`, …) inside its own
sub-directory of `scans/` (`section 01`, `section 02`, …). For each
section, every JPEG is portrait with **two book pages stacked
top-to-bottom** and the text running sideways.

For scan index *k* (0-based) within a section of *N* scans, the top page
is the lower-numbered page (`first_page + k`) and the bottom page is the
higher-numbered one (`first_page + 2N − 1 − k`).

Even-indexed scans (outer faces of each folded sheet) were placed on the
scanner rotated 90° counter-clockwise, so they need to be rotated 90°
clockwise to read. Odd-indexed scans (inner faces, where the sheet was
flipped over) were placed rotated 90° clockwise, so they need to be
rotated 90° counter-clockwise. After the appropriate rotation, the
original "top" page is the lower-numbered page; for even scans it ends
up on the right of the rotated image and for odd scans on the left.

Pre-processing per scan
-----------------------
Each scan also has a ~230-pixel-tall scanner artifact along its bottom
edge (`BOTTOM_CROP`). This strip is cropped before rotation so the two
halves are equal width and contain full page content.

Pages were never placed perfectly square on the scanner, so after the
90° rotation each scan is **deskewed** with a small horizontal-projection
variance search (`estimate_skew` / `deskew`) before being split into
left/right halves.

Some sections also contain post-processed JPEGs (`*.jpg`) that are
already single pages — these are appended verbatim to the page sequence
after the split scans.

Output
------
* `pages/page_NNN.jpg` — single-page JPEGs in reading order
* `book.pdf` — searchable PDF (image + invisible Apple Vision text layer)
* `book.txt` — plain-text sidecar in column-aware reading order
"""

from pathlib import Path
import io
import shutil

import numpy as np
from PIL import Image
import fitz  # PyMuPDF

import ocr_av

from itertools import groupby

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
    (PDF text-rendering mode 3) with one text run per WORD positioned
    at its per-word bounding box (via Vision's
    ``VNRecognizedText.boundingBoxForRange:``). A plain-text sidecar of
    every page is also written to `txt_path`, with words regrouped back
    into their parent Vision lines in the same reading order.

    `jpeg_quality` re-encodes each page image at the given JPEG quality
    before embedding it (the source pages are stored at quality 95). Pass
    ``None`` to embed the original JPEG bytes unchanged.

    `reading_order` controls how the sidecar text and the PDF text
    stream are ordered: ``"columns"`` reads each column top-to-bottom
    (best for the book's flowing two-column directory pages); ``"rows"``
    clusters lines by y and reads each row left-to-right (best for
    tabular layouts). Column / row clustering is done per LINE so words
    from the same line stay together.
    """
    if reading_order == "columns":
        def order(annots):
            ordered = ocr_av.sort_columns_words(annots)
            return ordered, ocr_av.join_word_lines(ordered)
    elif reading_order == "rows":
        def order(annots):
            rows = ocr_av.sort_rows_words(annots)
            ordered = [a for row in rows for a in row]
            # Within each row, words from the same Vision line are
            # joined with a single space; lines within a row are
            # separated by two spaces (preserving the previous tabular
            # feel for pages where one visual row contains multiple
            # logical fragments).
            text = "\n".join(
                "  ".join(
                    " ".join(a[0] for a in grp)
                    for _, grp in groupby(row, key=lambda a: a[3])
                )
                for row in rows
            )
            return ordered, text
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

        annots = ocr_av.annotate_words(page_file)
        ordered, page_text = order(annots)
        # Insert one invisible text run per word, in the chosen reading
        # order, so the PDF's text stream matches the sidecar.
        # Helvetica font metrics (relative to em). Used to size and
        # position each word so its selection box matches the Vision
        # bbox: the bbox spans ascender-top to descender-bottom =
        # (ASCENT+DESCENT) em, with the baseline DESCENT em above the
        # bottom of the bbox.
        HELV_ASCENT = 0.718
        HELV_DESCENT = 0.207
        for a in ordered:
            text, _conf = a[0], a[1]
            fx, fy, fw, fh = a[2]
            # Apple Vision: fractional coords, origin bottom-left.
            # PyMuPDF page: points, origin top-left, y grows downward.
            x_left = fx * w
            box_h_pt = fh * h
            fontsize = max(box_h_pt / (HELV_ASCENT + HELV_DESCENT), 1.0)
            # `(1 - fy) * h` is the bottom of the Vision bbox in PDF coords;
            # the baseline sits DESCENT em above that.
            baseline_y = (1.0 - fy) * h - HELV_DESCENT * fontsize
            page.insert_text(
                (x_left, baseline_y),
                text,
                fontname="helv",
                fontsize=fontsize,
                render_mode=3,             # invisible (searchable) text
            )

        txt_chunks.append(f"=== {page_file.name} ===")
        txt_chunks.append(page_text)
        txt_chunks.append("")
        print(f"OCR'd {page_file.name}: {len(annots)} words")

    doc.save(str(pdf_path), deflate=True, garbage=4, clean=True)
    doc.close()

    txt_path.write_text("\n".join(txt_chunks), encoding="utf-8")

    print(f"Wrote {pdf_path} ({len(page_files)} pages, "
          f"{pdf_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Wrote {txt_path} ({txt_path.stat().st_size / 1e3:.0f} KB)")
    return pdf_path


def main() -> None:
    scan_sections(SCANS_DIR)
    build_searchable_pdf(jpeg_quality=50)

if __name__ == "__main__":
    main()
