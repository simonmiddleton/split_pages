# bookscans

Tooling that reassembles a disassembled cased-bound book — a copy of
*Spalding's Cambridge Directory 1931–32* — from flatbed scans of its
folded signatures into a clean, searchable PDF and a plain-text
sidecar.

## Pipeline

The whole flow lives in [`split_pages.py`](split_pages.py) and runs in
two stages:

1. **`scan_sections` → `scan_section`** — turn each scanned signature
   in `scans/section NN/` into single-page JPEGs in `pages/`.
2. **`build_searchable_pdf`** — OCR every page with Apple Vision and
   produce a searchable [`book.pdf`](#outputs) plus a row- or
   column-ordered [`book.txt`](#outputs) sidecar.

Run end-to-end:

```bash
uv run python split_pages.py
```

### 1. Splitting and de-skewing scans

Each section is a folded signature. Every JPEG inside it is portrait,
contains two book pages stacked top-to-bottom, and has its text running
sideways because the leaves were placed long-edge-down on the scanner.
For scan index *k* (0-based) within a section of *N* scans:

* the **top page** of the scan is the lower-numbered page
  (`first_page + k`),
* the **bottom page** is the higher-numbered one
  (`first_page + 2N − 1 − k`).

Even-indexed scans (outer faces of each folded sheet) were placed on
the scanner rotated 90° counter-clockwise and need a 90° clockwise
rotation to read; odd-indexed scans (inner faces, after the sheet was
flipped) were placed 90° clockwise and need a 90° counter-clockwise
rotation.

After the appropriate 90° rotation, three further fixes are applied
before the scan is split into two halves:

* **Bottom artefact crop** — `BOTTOM_CROP = 230` pixels are trimmed off
  the bottom of the source scan to remove a flatbed scanner artefact
  along the platen edge.
* **De-skew** — pages were never placed perfectly square. `estimate_skew`
  does a coarse 0.5° sweep across ±3°, then a 0.05° fine sweep around
  the best candidate, scoring each angle by the variance of the
  horizontal pixel-sum projection (text rows align ⇒ sums alternate
  sharply ⇒ variance is maximal). A confidence guard
  (`best_score / score(0) ≥ 1.05`) suppresses noisy results on pages
  without good horizontal text structure (e.g. the fold-out map with
  diagonal street labels), so those pages are left unrotated rather
  than spuriously skewed by several degrees.
* **Edge trim** — `XTRIM = 60` pixels are dropped from the inner gutter
  on the split, removing residual binding shadow.

The `lo`/`hi` halves of the rotated scan are saved as
`pages/page_NNN.jpg` at JPEG quality 95.

After processing the JPEGs, any pre-existing `*.jpg` files inside the
section are appended verbatim, in sorted order, to the page sequence —
this lets you drop a manually-fixed page into the section and have it
slot in automatically.

### 2. OCR and searchable-PDF assembly

`build_searchable_pdf` walks `pages/page_*.jpg` in sorted order. For
each page it:

1. Re-encodes the JPEG at quality 80 (configurable via
   `jpeg_quality`) and embeds it into a new PDF page sized in PDF
   points to match the JPEG's pixel dimensions (1pt = 1px).
2. Runs Apple Vision via `ocr_av.annotate_words` to get a list of
   per-**word** annotations
   `(text, confidence, [x, y, w, h], line_id)` with fractional,
   bottom-left origin coordinates. Vision returns line-level
   observations; per-word bounding boxes come from
   `VNRecognizedText.boundingBoxForRange:error:`, called via pyobjc
   (`ocrmac` only exposes line bboxes). `line_id` is the index of the
   parent Vision line so the sidecar can regroup words back into lines.
3. Sorts those word annots according to `reading_order` using the
   word-aware variants `sort_columns_words` / `sort_rows_words`. These
   compute the column / row clustering on each line's union bbox so
   words from the same line stay together; words within a line keep
   their natural left-to-right order.
4. Inserts each **word** as its own **invisible text run** (PDF
   text-rendering mode 3) at its own bounding box on the PDF page.
   This is what makes the PDF searchable; because the box is per-word,
   selection highlights and copy-of-partial-line selections in
   Preview/Acrobat hug individual words rather than spanning the whole
   line.
5. Joins the same sorted words back into lines (via `ocr_av.join_word_lines`)
   in the `book.txt` sidecar so the plain text and the PDF text stream
   are produced in the same order.

Two reading-order strategies are exposed via `ocr_av.py`:

| Order      | Helper                | Use when                                                   |
|------------|-----------------------|------------------------------------------------------------|
| `columns`  | `sort_columns`        | Flowing two-column directory pages (most of this book).    |
| `rows`     | `sort_rows`           | Tabular pages where each visual row is one logical record. |

Each helper has a sibling that returns just the joined string
(`columns_from_annots`, `rows_from_annots`) — these are kept for ad-hoc
text rendering of a single page.

## Outputs

| File         | Contents                                                                              |
|--------------|----------------------------------------------------------------------------------------|
| `pages/`     | Single-page JPEGs in reading order (`page_001.jpg` … `page_473.jpg`).                 |
| `book.pdf`   | Searchable PDF: image + invisible Apple Vision text layer.                            |
| `book.txt`   | Plain-text sidecar in `reading_order` order. Pages are separated by `=== page_NNN.jpg ===` headers. |

`scans/`, `pages/`, and any `*.pdf` are gitignored.

## OCR engine choice

We benchmarked **Tesseract 5.5** (both `eng` and `eng_best` traineddata,
PSM 3 and PSM 6) and **Apple Vision** (via `ocrmac`) on three
representative pages — see [`ocr_bench.py`](ocr_bench.py) and the
outputs in `ocr_bench/`.

Apple Vision wins decisively on this material:

* On a typical two-column directory page (e.g. `page_200`), Tesseract
  silently dropped *the entire right-hand column* (~30 entries from
  "15 Baldry, Geo." through "24 Wright, W.") whereas Apple Vision
  caught all of it.
* On the cover, Tesseract emitted ~150 characters of garbled noise
  hallucinated from the decorative border; Apple Vision returned just
  the real text (`CAMBRIDGE 1931-32`).
* It's about 2× faster (~0.4–0.7 s/page vs 1–2.5 s for Tesseract
  `eng_best`) and runs entirely on-device via macOS' VisionKit.

The trade-off is that Apple Vision occasionally misreads
single-character forms — `1` ↔ `I` in numerals, `T` ↔ `I` mid-word
(`Terrace` → `Ierrace`), etc. These are tractable in post-processing
and are far less destructive than missing whole columns.

## ⚠️ PDF text reading-order trade-off

This is the subtle bit, and worth understanding before relying on
copy-paste from the PDF.

`build_searchable_pdf` inserts the invisible text in the chosen
`reading_order`, and that's exactly what you'll get from any tool that
reads the **PDF content stream in raw order**:

```bash
pdftotext -raw book.pdf -                  # ✓ matches book.txt
```

```python
import fitz
fitz.open("book.pdf")[199].get_text(sort=False)  # ✓ matches book.txt
```

But most consumer tools — including **macOS Preview**, **Adobe
Acrobat** selection-copy, **`pdftotext`** without `-raw`, and most
browser PDF viewers — re-run their own *spatial* reading-order
heuristic on the glyph positions. Because every fragment is positioned
exactly over the visible word it represents (so search-highlight
rectangles land on the right text), those heuristics see two columns
of similarly-positioned text and **interleave them line-by-line** —
i.e. one row from the left column, then one row from the right column,
and so on. Copy-paste from Preview will therefore *not* match
`book.txt` on multi-column pages.

There are three ways to resolve this. They each cost something:

| Approach                                                    | Search highlight on visible word | Preview/Acrobat copy order |
|-------------------------------------------------------------|----------------------------------|----------------------------|
| **Current** (text positioned per-word, in reading order)     | ✅                                | ❌ (interleaved)            |
| **Linearised layout** (text stacked vertically by reading rank) | ❌ (highlights at synthetic y)    | ✅ (clean)                  |
| **Tagged PDF** with `/StructTreeRoot`                        | ✅                                | ❌ (Preview/Acrobat ignore tags for ordinary copy — tags are for accessibility/reflow) |

We deliberately keep the **current** behaviour because:

1. Search highlighting on the visible word is the highest-value feature
   of a searchable archive PDF — clicking a search hit should land on
   the actual word in the image.
2. The `book.txt` sidecar already provides clean reading-order text
   for full-text export.
3. `pdftotext -raw` (and PyMuPDF's `get_text(sort=False)`) honour the
   stream order if you need it programmatically.
4. Tagged PDF would not help the consumer tools that motivate the
   request, and is considerably more work.

If you ever do need correct copy order in Preview/Acrobat, add a
`text_layout="linear"` mode that reassigns each fragment's `y` to its
rank in the reading order and accept that search highlights will
appear at synthetic positions.

## Repository layout

```
split_pages.py     # main pipeline (scan → split → deskew → OCR → searchable PDF + txt)
ocr_av.py          # Apple Vision wrapper + column/row sort helpers
ocr_bench.py       # benchmarking harness (Tesseract vs Apple Vision)
ocr_bench/         # benchmark output (gitignored)
pyproject.toml     # uv-managed project (Python ≥ 3.14)
uv.lock
scans/             # source scans (gitignored)
pages/             # split single-page JPEGs (gitignored)
book.pdf           # searchable output (gitignored)
book.txt           # sidecar text
```

## Dependencies

Managed via `uv`. Runtime dependencies:

* `pillow` — image I/O and rotation/cropping
* `pymupdf` — PDF assembly with invisible text overlay
* `ocrmac` — `ocrmac` wraps Apple's `VNRecognizeTextRequest`
* `numpy` — projection-profile arithmetic for de-skew

External: macOS (Apple Vision is OS-level), and `tesseract` 5.x is
optional and only needed if you want to run [`ocr_bench.py`](ocr_bench.py)
to compare engines.
