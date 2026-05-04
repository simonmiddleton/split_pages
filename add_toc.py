#!/usr/bin/env python3
"""Build a PDF table of contents for book.pdf.

Top-level entries come from the General Index.
Street-level sub-entries are extracted from the Index to Places using
column-aware word-position extraction from the OCR text layer, then
mapped to PDF page numbers via the derived book-page offset.

Section page numbers and the book-page offset are discovered automatically
by scanning the PDF for characteristic text patterns.

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

_MIN_PLACES_ENTRIES = 10   # pages with fewer entries are not Index to Places


# ── OCR digit normalisation ──────────────────────────────────────────────────

def fix_digits(s: str) -> str:
    """Repair common OCR digit-substitutions in a page-number string."""
    return (s
            .replace('I', '1')
            .replace('l', '1')
            .replace('б', '6')
            .replace('o', '0')
            .replace('о', '0')
            .replace('О', '0'))


def parse_nums(s: str) -> list[int]:
    """Parse page numbers, handling OCR splits like 'I 39' → 139.

    Commas separate genuinely distinct page numbers; spaces within a
    comma-group are OCR artefacts (e.g. 'I 39' = 139, '1 52,257' = 152,257).
    """
    fixed = fix_digits(s)
    result = []
    for part in re.split(r',', fixed):
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


# ── Places-entry counter (used for auto-detecting the section range) ──────────

def _count_places_entries(page: fitz.Page) -> int:
    """Count valid Index to Places entries on a single page."""
    count = 0
    for col in column_lines(page):
        pending: str | None = None
        for raw in col:
            line = raw.strip()
            if not line or _SEPARATOR.match(line):
                continue
            if _HEADER.match(line):
                pending = None
                continue
            prefix, nums = trailing_nums(line)
            if nums:
                name = prefix if prefix else pending
                if name and re.match(r'^[A-Z][a-zA-Z]', name.rstrip('.,•· ')):
                    count += 1
                pending = None
            elif re.match(r'^[A-Z][a-zA-Z]', line) and not re.search(r'\d', line):
                pending = line.rstrip('.,•· ')
            else:
                pending = None
    return count


# ── Section discovery ────────────────────────────────────────────────────────

def discover_sections(doc: fitz.Document) -> dict[str, int]:
    """Scan doc once and return {section_name: first_pdf_page_1based}.

    Sections found:
      general_index       – page with "GENERAL INDEX" heading
      index_to_places     – first page with "INDEX TO PLACES" heading
      streets             – page with "Street Directory" heading
      private_residents   – page that opens with "Private Residents"
      trades              – page that opens with "advertisement pages"
    """
    sections: dict[str, int] = {}
    needed = {'general_index', 'index_to_places', 'streets', 'private_residents', 'trades'}

    for i, page in enumerate(doc):
        if not needed:
            break
        pdf_page = i + 1
        text = page.get_text()
        head = text[:100]   # page opening — used for section-start detection

        if 'general_index' in needed and re.search(r'GENERAL\s*\nINDEX', text, re.I):
            sections['general_index'] = pdf_page
            needed.discard('general_index')

        if 'index_to_places' in needed and re.search(r'INDEX\s+TO\s+PLACES', text, re.I):
            sections['index_to_places'] = pdf_page
            needed.discard('index_to_places')

        if 'streets' in needed and re.search(r'Street\s+Directory', text, re.I):
            sections['streets'] = pdf_page
            needed.discard('streets')

        # Only match when "Private Residents" is the very start of the page,
        # not when it appears as a cross-reference in earlier content.
        if 'private_residents' in needed and re.search(r'Private\s+Residents', head, re.I):
            sections['private_residents'] = pdf_page
            needed.discard('private_residents')

        # The Trades section opens with "advertisement pages." as a section title.
        if 'trades' in needed and re.search(r'advertisement\s+pages', head, re.I):
            sections['trades'] = pdf_page
            needed.discard('trades')

    return sections


def find_places_pages(doc: fitz.Document, start: int) -> list[int]:
    """Return 1-based PDF page numbers for the Index to Places section.

    Starts at start and extends forward while pages contain enough entries,
    stopping at the first page that falls below _MIN_PLACES_ENTRIES.
    """
    pages: list[int] = []
    for i in range(start - 1, min(len(doc), start + 30)):
        if _count_places_entries(doc[i]) >= _MIN_PLACES_ENTRIES:
            pages.append(i + 1)
        elif pages:   # below threshold after accumulating at least one page
            break
    return pages


# ── Main parser ──────────────────────────────────────────────────────────────

def parse_places(doc: fitz.Document, pages: list[int]) -> dict[str, int]:
    """Return {place_name: first_book_page} from the Index to Places pages."""
    entries: dict[str, int] = {}

    for pnum in pages:
        page = doc[pnum - 1]   # fitz uses 0-based indexing
        for col in column_lines(page):
            pending: str | None = None

            for raw in col:
                line = raw.strip()
                if not line:
                    continue

                if _SEPARATOR.match(line):
                    continue

                if _HEADER.match(line):
                    pending = None
                    continue

                prefix, nums = trailing_nums(line)

                if nums:
                    name = prefix if prefix else pending
                    if name:
                        name = name.rstrip('.,•· ')
                        if re.match(r'^[A-Z][a-zA-Z]', name):
                            entries.setdefault(name, nums[0])
                    pending = None

                elif re.match(r'^[A-Z][a-zA-Z]', line) and not re.search(r'\d', line):
                    pending = line.rstrip('.,•· ')

                else:
                    pending = None

    return entries


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    doc = fitz.open(str(IN_PDF))
    n_pages = len(doc)

    print(f"Discovering section structure in {IN_PDF} …")
    sections = discover_sections(doc)

    missing = [s for s in ('general_index', 'index_to_places', 'streets',
                           'private_residents', 'trades')
               if s not in sections]
    if missing:
        raise SystemExit(f"Could not locate sections: {', '.join(missing)}")

    places_pages = find_places_pages(doc, sections['index_to_places'])
    if not places_pages:
        raise SystemExit("Could not identify any Index to Places pages")

    offset = sections['streets'] - 1   # pdf_page = book_page + offset

    print(f"  General Index:          PDF p.{sections['general_index']}")
    print(f"  Index to Places:        PDF p.{places_pages[0]}–{places_pages[-1]}"
          f"  ({len(places_pages)} pages)")
    print(f"  Street Directory:       PDF p.{sections['streets']}"
          f"  (offset = {offset})")
    print(f"  Private Residents:      PDF p.{sections['private_residents']}")
    print(f"  Trades and Professions: PDF p.{sections['trades']}")

    print(f"\nParsing Index to Places …")
    streets = parse_places(doc, places_pages)
    print(f"  {len(streets)} entries found")

    sample = sorted(streets.items())[:15]
    print("  First 15:")
    for name, bp in sample:
        print(f"    {name!r:42s}  book p.{bp:3d}  →  PDF p.{bp + offset}")

    # Build TOC in document order
    toc: list[list] = [
        [1, "General Index",   sections['general_index']],
        [1, "Index to Places", places_pages[0]],
    ]
    out_of_range = 0
    for name, bp in sorted(streets.items()):
        pdf = bp + offset
        if 1 <= pdf <= n_pages:
            toc.append([2, string.capwords(name), pdf])
        else:
            out_of_range += 1

    toc += [
        [1, "Streets in Alphabetical Order", sections['streets']],
        [1, "Private Residents",             sections['private_residents']],
        [1, "Trades and Professions",        sections['trades']],
    ]

    if out_of_range:
        print(f"  ({out_of_range} entries skipped — page number out of range)")

    doc.set_toc(toc)
    doc.save(str(OUT_PDF), deflate=True, garbage=4)
    doc.close()

    n_street_entries = sum(1 for t in toc if t[0] == 2)
    print(f"\nWrote {OUT_PDF}  ({n_pages} pages, {len(toc)} bookmarks, "
          f"{n_street_entries} street entries)")
    first_street = sorted(streets.items())[0] if streets else None
    print("\nVerification hints:")
    print(f"  Open {OUT_PDF}, check bookmark panel for 'Index to Places'")
    if first_street:
        name, bp = first_street
        print(f"  Click '{string.capwords(name)}' → should be PDF page {bp + offset}")
    print(f"  Click 'Streets in Alphabetical Order' → should be PDF page {sections['streets']}")


if __name__ == "__main__":
    main()
