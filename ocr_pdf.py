from pathlib import Path
import io
from itertools import groupby

from PIL import Image
import fitz  # PyMuPDF

import re

import ocr_av
from add_toc import fix_digits

# Characters that Vision commonly returns instead of digits.  A token made
# up entirely of these (plus real digits) is treated as a number and passed
# through fix_digits; ordinary words are left untouched.
_NUMBER_LIKE = re.compile(r'^[0-9IloоОб]+$')  # I=capital i, l=lowercase L, o=latin o, о/О=cyrillic o, б=cyrillic б


def _fix_word(w: str) -> str:
    return fix_digits(w) if _NUMBER_LIKE.match(w) else w

OUT_DIR  = Path("pages")
PDF_PATH = Path("book.pdf")
TXT_PATH = Path("book.txt")


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
            return ordered, ocr_av.join_word_lines(ordered, row_tol=0.012)
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
        annots = [(_fix_word(a[0]),) + a[1:] for a in annots]
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
                text + ('' if text[-1] == '' else ' '),
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
