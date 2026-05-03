"""Apple Vision OCR helpers.

`ocrmac` (which wraps macOS' VNRecognizeTextRequest) returns text fragments
with bounding boxes (fractional, origin bottom-left). This module exposes:

  * `annotate(jpeg)` — raw fragments
  * `recognise_rows(jpeg)` — row-aware reading order (best for tables &
    aligned multi-column layouts)
  * `recognise(jpeg)` — column-aware reading order (best for true
    multi-column flow)
"""
from __future__ import annotations

from pathlib import Path
from statistics import median


def annotate(jpeg_path: str | Path):
    """Return Apple Vision OCR annotations: list of (text, conf, [x,y,w,h]).
    Coordinates are fractional with origin bottom-left.
    """
    from ocrmac import ocrmac
    return ocrmac.OCR(str(jpeg_path), recognition_level="accurate").recognize()


def _cluster_columns(centres: list[float], gap: float = 0.10) -> list[float]:
    """Return sorted column-centre estimates by 1-D clustering of x-centres.

    `gap` is the minimum normalised distance between adjacent column centres.
    """
    if not centres:
        return []
    xs = sorted(centres)
    cols: list[list[float]] = [[xs[0]]]
    for x in xs[1:]:
        if x - cols[-1][-1] > gap:
            cols.append([x])
        else:
            cols[-1].append(x)
    return [median(c) for c in cols]


def _assign_column(x_centre: float, col_centres: list[float]) -> int:
    return min(range(len(col_centres)), key=lambda i: abs(col_centres[i] - x_centre))


def sort_rows(annots, *, row_tol: float = 0.012):
    """Group annots into top-to-bottom rows; each row is sorted left-to-right.

    Returns ``list[list[annot]]``.
    """
    if not annots:
        return []

    # Sort by y descending (top of image first since origin is bottom-left).
    items = sorted(annots, key=lambda a: -(a[2][1] + a[2][3]))

    rows: list[list] = []
    for a in items:
        y_top = a[2][1] + a[2][3]
        if rows and abs((rows[-1][0][2][1] + rows[-1][0][2][3]) - y_top) < row_tol:
            rows[-1].append(a)
        else:
            rows.append([a])

    for row in rows:
        row.sort(key=lambda a: a[2][0])
    return rows


def sort_columns(annots, *, gap: float = 0.10):
    """Sort annots into column-aware reading order: each column read
    top-to-bottom in turn. Returns a flat ``list[annot]``.
    """
    if not annots:
        return []

    # Use the centre x of each fragment, ignoring very short fragments (likely
    # noise like a stray punctuation mark) so column detection is stable.
    centres = [b[0] + b[2] / 2 for _, _, b in annots if b[2] > 0.02]
    col_centres = _cluster_columns(centres, gap=gap)

    keyed = []
    for a in annots:
        x, y, w, h = a[2]
        cx = x + w / 2
        col = _assign_column(cx, col_centres) if col_centres else 0
        # y is bottom-edge fraction with origin bottom-left — invert so larger
        # values correspond to further down the page for natural sort order.
        keyed.append(((col, -(y + h), x), a))

    keyed.sort(key=lambda kv: kv[0])
    return [a for _, a in keyed]


def rows_from_annots(annots, *, row_tol: float = 0.012) -> str:
    """Row-aware text rendering: each row's fragments joined by two spaces,
    rows joined by newlines. Best for tables / aligned multi-column data
    where each physical row is meant to be read as one record.
    """
    rows = sort_rows(annots, row_tol=row_tol)
    return "\n".join("  ".join(a[0] for a in row) for row in rows)


def columns_from_annots(annots, *, gap: float = 0.10) -> str:
    """Column-aware text rendering: each column read top-to-bottom in turn,
    one fragment per line. Best for flowing multi-column text where each
    column is a continuation of reading order.
    """
    return "\n".join(a[0] for a in sort_columns(annots, gap=gap))


def recognise_rows(jpeg_path: str | Path, *, row_tol: float = 0.012) -> str:
    return rows_from_annots(annotate(jpeg_path), row_tol=row_tol)


def recognise(jpeg_path: str | Path, *, gap: float = 0.10) -> str:
    return columns_from_annots(annotate(jpeg_path), gap=gap)


if __name__ == "__main__":
    import sys
    print(recognise(sys.argv[1]))
