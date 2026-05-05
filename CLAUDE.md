# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv run python split_pages.py all                     # full pipeline: images → ocr → toc
uv run python split_pages.py images                  # stage 1: split/deskew scans → pages/
uv run python split_pages.py ocr                     # stage 2: OCR pages/ → book.pdf + book.txt
uv run python split_pages.py toc                     # stage 3: add bookmarks → book_toc.pdf
uv run python split_pages.py ocr --reading-order rows  # row-order instead of column-order
uv run python split_pages.py <subcommand> --help     # full option list for any stage
```

There are no automated tests — validation is done by inspecting output files.

Always use `uv run python` (never bare `python3`).

## Pipeline architecture

Three stages, each in its own module, driven by `split_pages.py` (argparse CLI):

```
scans/section NN/*.jpeg  →  [image_processing.py]  →  pages/page_NNN.jpg
pages/page_NNN.jpg       →  [ocr_pdf.py]           →  book.pdf + book.txt
book.pdf                 →  [add_toc.py]            →  book_toc.pdf
```

### Stage 1 — `image_processing.py`

Each source scan is a portrait JPEG containing **two book pages stacked vertically**, text rotated 90°. Processing per scan:
1. Rotate 90° CW (even index) or CCW (odd index) to restore upright text
2. Crop `--bottom-crop` pixels (default 230) to remove scanner platen artefact
3. Deskew: coarse 0.5° sweep ±3°, then fine 0.05° around best; scored by variance of horizontal projection sums. Suppressed if improvement < 5% (protects fold-out map pages with diagonal text)
4. Split into top/bottom halves; trim `--xtrim` pixels from inner gutter

### Stage 2 — `ocr_pdf.py` + `ocr_av.py`

`ocr_av.py` wraps Apple Vision (via `ocrmac`/pyobjc). Key detail: `ocrmac` exposes only line-level bboxes; `annotate_words()` calls `VNRecognizedText.boundingBoxForRange:` via pyobjc to get **per-word** bboxes. Each word gets its own invisible PDF text run (render mode 3) positioned over its image location — this is what enables accurate search highlighting.

`_fix_word()` in `ocr_pdf.py` corrects OCR digit confusions before embedding. `fix_digits()` in `add_toc.py` is the canonical substitution table (imported by `ocr_pdf.py`): `I/l→1`, `O/o→0`, Cyrillic `о/О→0`, `б→6`.

Two reading-order strategies in `ocr_av.py`:
- `columns` (default): clusters word x-centres by line into columns (gap threshold 0.06), reads each column top-to-bottom — correct for the two-column directory pages
- `rows`: groups lines by y into rows (tolerance 0.012), reads left-to-right — correct for tabular pages

### Stage 3 — `add_toc.py`

Scans the PDF text layer for section headers (`GENERAL INDEX`, `INDEX TO PLACES`, etc.) to discover page offsets, then parses the Index to Places using `column_lines()` (groups words by x-centre into N columns, clusters into visual lines by y within 4pt) to extract `{street → book_page}` and build hierarchical PDF bookmarks.

## Key design trade-off

Text runs are positioned at real word locations, so **search highlighting lands on the correct word** in the image. The cost: Preview/Acrobat copy-paste on multi-column pages interleaves columns line-by-line. `book.txt` and `pdftotext -raw` / `fitz.get_text(sort=False)` give correct reading order. This is intentional — do not change it without reading the README's "PDF text reading-order trade-off" section.

## Dependencies

Managed by `uv` (`pyproject.toml`). Python ≥ 3.14. Requires macOS (Apple Vision is OS-level). Tesseract is optional, only needed for `ocr_bench.py`.
