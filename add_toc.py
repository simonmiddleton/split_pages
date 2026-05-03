#!/usr/bin/env python3
"""Build a PDF table of contents for book.pdf.

Top-level entries come from the General Index (page 21).
Street-level sub-entries are extracted from the Index to Places (pages 25-29)
using column-aware word-position extraction from the OCR text layer, then
mapped to PDF page numbers via the fixed book-page offset.

Usage:
    uv run python add_toc.py
Output: book_toc.pdf
"""

from pathlib import Path
import re
import string
from collections import defaultdict
import fitz  # PyMuPDF

IN_PDF  = Path("book.pdf")
OUT_PDF = Path("book_toc.pdf")

# PDF page 69 == book page 1 of the Streets section.
OFFSET = 68          # pdf_page = book_page + OFFSET
PLACES_PAGES = range(25, 30)   # PDF pages 25-29 (Index to Places)


# ── OCR digit normalisation ──────────────────────────────────────────────────

def fix_digits(s: str) -> str:
    """Repair common OCR digit-substitutions in a page-number string."""
    s = re.sub(r'\bI(?=\d)', '1', s)    # I76  → 176
    s = re.sub(r'(?<=\d)I\b', '1', s)   # 15I  → 151
    s = re.sub(r'\bI\b', '1', s)        # lone I → 1
    return (s
            .replace('б', '6')
            .replace('о', '0')
            .replace('О', '0')
            .replace('l', '1'))


def parse_nums(s: str) -> list[int]:
    """Parse page numbers, handling OCR splits like 'I 39' → 139.

    Commas separate genuinely distinct page numbers; spaces within a
    comma-group are OCR artefacts (e.g. 'I 39' = 139, '1 52,257' = 152,257).
    """
    fixed = fix_digits(s)
    result = []
    for part in re.split(r',', fixed):
        # Join space-separated digit tokens within one comma-group
        joined = ''.join(re.split(r'\s+', part.strip()))
        if re.fullmatch(r'\d+', joined) and 1 <= int(joined) <= 500:
            result.append(int(joined))
    return result


def trailing_nums(line: str) -> tuple[str, list[int]]:
    """Split a line into (name_prefix, page_numbers) by finding trailing digits.

    Handles:
      "Street name 42"          → ("Street name", [42])
      "Street name ... 42, 73"  → ("Street name", [42, 73])
      "••. IOI"                  → ("", [101])
      "Street name"             → ("Street name", [])
    """
    # Capture anything that ends with OCR-digit tokens after optional separators
    m = re.match(
        r'^(.*?)\s*[.•·\s]*\s*([0-9I]\S*(?:[,\s]+[0-9I]\S*)*)$',
        line.rstrip(),
    )
    if m:
        raw_nums = m.group(2)
        fixed = fix_digits(raw_nums)
        if re.fullmatch(r'[\d,\s]+', fixed.strip()):
            nums = parse_nums(raw_nums)
            if nums:
                return m.group(1).strip(), nums
    return line, []


# ── Column-aware line extraction from a PDF page ────────────────────────────

def column_lines(page: fitz.Page, n_cols: int = 3) -> list[list[str]]:
    """Return n_cols lists of text lines, each extracted from one visual column.

    Words are grouped into columns by x-centre position, then within each
    column sorted top-to-bottom and assembled into text lines by clustering
    words whose top edges are within 4 pt of each other.
    """
    raw = page.get_text("words")    # (x0,y0,x1,y1, text, block,line,word)
    if not raw:
        return [[] for _ in range(n_cols)]

    pw = page.rect.width
    cw = pw / n_cols

    buckets: dict[int, list] = defaultdict(list)
    for w in raw:
        col = min(int(((w[0] + w[2]) / 2) / cw), n_cols - 1)
        buckets[col].append(w)

    result: list[list[str]] = []
    for ci in range(n_cols):
        words = sorted(buckets[ci], key=lambda w: (w[1], w[0]))
        lines: list[str] = []
        cur: list = []
        for w in words:
            if cur and w[1] - cur[0][1] > 4:
                lines.append(' '.join(x[4] for x in sorted(cur, key=lambda x: x[0])))
                cur = []
            cur.append(w)
        if cur:
            lines.append(' '.join(x[4] for x in sorted(cur, key=lambda x: x[0])))
        result.append(lines)
    return result


# ── Noise filter ─────────────────────────────────────────────────────────────

_HEADER = re.compile(
    r'^('
    r'INDEX\s+TO\s+PLACES\.?'
    r'|PAGE'
    r'|CAMBRIDGE\s+DIRECTORY.*'
    r'|[xivlc]+\)?'            # Roman numeral page header
    r')$',
    re.I,
)

_SEPARATOR = re.compile(r'^[.•·\s\-]+$')


# ── Main parser ──────────────────────────────────────────────────────────────

def parse_places(doc: fitz.Document) -> dict[str, int]:
    """Return {place_name: first_book_page} from the Index to Places pages."""
    entries: dict[str, int] = {}

    for pnum in PLACES_PAGES:
        page = doc[pnum - 1]   # fitz uses 0-based indexing
        for col in column_lines(page):
            pending: str | None = None

            for raw in col:
                line = raw.strip()
                if not line:
                    continue

                # Dot/bullet separator lines: preserve the pending name so the
                # page number that follows can still be associated with it.
                if _SEPARATOR.match(line):
                    continue

                # Header/footer lines reset context.
                if _HEADER.match(line):
                    pending = None
                    continue

                prefix, nums = trailing_nums(line)

                if nums:
                    # Either a complete "Name ... 42" entry, or a number line
                    # following a pending name
                    name = prefix if prefix else pending
                    if name:
                        name = name.rstrip('.,•· ')
                        # Require the name to start with a capital letter and
                        # contain only word characters (skip garbled lines)
                        if re.match(r'^[A-Z][a-zA-Z]', name):
                            entries.setdefault(name, nums[0])
                    pending = None

                elif re.match(r'^[A-Z][a-zA-Z]', line) and not re.search(r'\d', line):
                    # Looks like a place name with no page number yet
                    pending = line.rstrip('.,•· ')

                else:
                    # Unrecognised line — discard any pending name
                    pending = None

    return entries


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    doc = fitz.open(str(IN_PDF))
    n_pages = len(doc)

    print(f"Parsing Index to Places (PDF pages 25–29) from {IN_PDF} …")
    streets = parse_places(doc)
    print(f"  {len(streets)} entries found")

    sample = sorted(streets.items())[:15]
    print("  First 15:")
    for name, bp in sample:
        print(f"    {name!r:42s}  book p.{bp:3d}  →  PDF p.{bp + OFFSET}")

    # Build TOC in document order
    toc: list[list] = [
        [1, "General Index",  21],
        [1, "Index to Places", 25],
    ]
    out_of_range = 0
    for name, bp in sorted(streets.items()):
        pdf = bp + OFFSET
        if 1 <= pdf <= n_pages:
            toc.append([2, string.capwords(name), pdf])
        else:
            out_of_range += 1

    toc += [
        [1, "Streets in Alphabetical Order", 69],
        [1, "Private Residents",             341],
        [1, "Trades and Professions",        423],
    ]

    if out_of_range:
        print(f"  ({out_of_range} entries skipped — page number out of range)")

    doc.set_toc(toc)
    doc.save(str(OUT_PDF), deflate=True, garbage=4)
    doc.close()

    n_streets = sum(1 for t in toc if t[0] == 2)
    print(f"\nWrote {OUT_PDF}  ({n_pages} pages, {len(toc)} bookmarks, "
          f"{n_streets} street entries)")
    print("\nVerification hints:")
    print("  Open book_toc.pdf, check bookmark panel for 'Index to Places'")
    print("  Click 'Abbey Road' → should be PDF page 70")
    print("  Click 'Streets in Alphabetical Order' → should be PDF page 69")


if __name__ == "__main__":
    main()
