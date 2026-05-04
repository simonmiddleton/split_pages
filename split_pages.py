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

Subcommands
-----------
  images   Split and deskew scans into page JPEGs
  ocr      OCR page JPEGs and build searchable PDF
  toc      Add table of contents to PDF
  all      Run images → ocr → toc in sequence
"""

from pathlib import Path
import argparse

import image_processing
import ocr_pdf
import add_toc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild a book from disassembled scans.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── images ────────────────────────────────────────────────────────────────
    p_img = sub.add_parser(
        "images",
        help="Split and deskew scans into page JPEGs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_img.add_argument("--scans-dir",   default="scans", metavar="DIR",
                       help="directory containing scan section subdirectories")
    p_img.add_argument("--pages-dir",   default="pages", metavar="DIR",
                       help="output directory for single-page JPEGs")
    p_img.add_argument("--bottom-crop", default=230,     type=int, metavar="PX",
                       help="pixels to trim from bottom of each source scan")
    p_img.add_argument("--xtrim",       default=60,      type=int, metavar="PX",
                       help="pixels to trim from each side after splitting")

    # ── ocr ───────────────────────────────────────────────────────────────────
    p_ocr = sub.add_parser(
        "ocr",
        help="OCR page JPEGs and build searchable PDF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_ocr.add_argument("--pages-dir",     default="pages",    metavar="DIR",
                       help="directory containing page_NNN.jpg files")
    p_ocr.add_argument("--pdf",           default="book.pdf", metavar="FILE",
                       help="output PDF path")
    p_ocr.add_argument("--txt",           default="book.txt", metavar="FILE",
                       help="output plain-text sidecar path")
    p_ocr.add_argument("--jpeg-quality",  default=50,         type=int, metavar="Q",
                       help="JPEG quality for PDF embedding (None to embed unchanged)")
    p_ocr.add_argument("--reading-order", default="columns",
                       choices=["columns", "rows"],
                       help="text ordering: columns (flowing) or rows (tabular)")

    # ── toc ───────────────────────────────────────────────────────────────────
    p_toc = sub.add_parser(
        "toc",
        help="Add table of contents to PDF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_toc.add_argument("--pdf",     default="book.pdf",     metavar="FILE",
                       help="input PDF path")
    p_toc.add_argument("--out-pdf", default="book_toc.pdf", metavar="FILE",
                       help="output PDF path with TOC bookmarks")

    # ── all ───────────────────────────────────────────────────────────────────
    p_all = sub.add_parser(
        "all",
        help="Run images → ocr → toc in sequence",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_all.add_argument("--scans-dir",    default="scans",        metavar="DIR")
    p_all.add_argument("--pages-dir",    default="pages",        metavar="DIR")
    p_all.add_argument("--pdf",          default="book.pdf",     metavar="FILE")
    p_all.add_argument("--txt",          default="book.txt",     metavar="FILE")
    p_all.add_argument("--bottom-crop",  default=230,            type=int, metavar="PX")
    p_all.add_argument("--xtrim",        default=60,             type=int, metavar="PX")
    p_all.add_argument("--jpeg-quality", default=50,             type=int, metavar="Q")
    p_all.add_argument("--reading-order",default="columns",
                       choices=["columns", "rows"])
    p_all.add_argument("--out-pdf",      default="book_toc.pdf", metavar="FILE",
                       help="output PDF path with TOC bookmarks")

    args = parser.parse_args()

    if args.command == "images":
        image_processing.OUT_DIR     = Path(args.pages_dir)
        image_processing.BOTTOM_CROP = args.bottom_crop
        image_processing.XTRIM       = args.xtrim
        image_processing.scan_sections(Path(args.scans_dir))

    elif args.command == "ocr":
        ocr_pdf.build_searchable_pdf(
            pages_dir=Path(args.pages_dir),
            pdf_path=Path(args.pdf),
            txt_path=Path(args.txt),
            jpeg_quality=args.jpeg_quality,
            reading_order=args.reading_order,
        )

    elif args.command == "toc":
        add_toc.IN_PDF  = Path(args.pdf)
        add_toc.OUT_PDF = Path(args.out_pdf)
        add_toc.main()

    elif args.command == "all":
        image_processing.OUT_DIR     = Path(args.pages_dir)
        image_processing.BOTTOM_CROP = args.bottom_crop
        image_processing.XTRIM       = args.xtrim
        image_processing.scan_sections(Path(args.scans_dir))

        ocr_pdf.build_searchable_pdf(
            pages_dir=Path(args.pages_dir),
            pdf_path=Path(args.pdf),
            txt_path=Path(args.txt),
            jpeg_quality=args.jpeg_quality,
            reading_order=args.reading_order,
        )

        add_toc.IN_PDF  = Path(args.pdf)
        add_toc.OUT_PDF = Path(args.out_pdf)
        add_toc.main()


if __name__ == "__main__":
    main()
