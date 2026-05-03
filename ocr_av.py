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


def rows_from_annots(annots, *, row_tol: float = 0.012) -> str:
    """Row-aware reader: cluster fragments by y into rows, then read each row
    left-to-right. Best for tables / aligned multi-column data where each
    physical row is meant to be read as one record.
    """
    if not annots:
        return ""

    items = [(text, x, y, w, h) for text, _conf, (x, y, w, h) in annots]
    # Sort by y descending (top of image first since origin is bottom-left).
    items.sort(key=lambda it: -(it[2] + it[4]))

    rows: list[list[tuple[str, float, float, float, float]]] = []
    for it in items:
        y_top = it[2] + it[4]
        if rows and abs((rows[-1][0][2] + rows[-1][0][4]) - y_top) < row_tol:
            rows[-1].append(it)
        else:
            rows.append([it])

    out_lines: list[str] = []
    for row in rows:
        row.sort(key=lambda it: it[1])  # left-to-right
        # Pad gaps between cells with two spaces; tabs would be confusing in
        # a plain-text comparison.
        out_lines.append("  ".join(it[0] for it in row))
    return "\n".join(out_lines)


def columns_from_annots(annots, *, gap: float = 0.10) -> str:
    """Column-aware reader: detect column boundaries by clustering x-centres
    and read each column top-to-bottom. Best for flowing multi-column text
    where each column is a continuation of reading order.
    """
    if not annots:
        return ""

    # Use the centre x of each fragment, ignoring very short fragments (likely
    # noise like a stray punctuation mark) so column detection is stable.
    centres = [b[0] + b[2] / 2 for _, _, b in annots if b[2] > 0.02]
    col_centres = _cluster_columns(centres, gap=gap)

    rows = []
    for text, _conf, (x, y, w, h) in annots:
        cx = x + w / 2
        col = _assign_column(cx, col_centres) if col_centres else 0
        # y is bottom-edge fraction with origin bottom-left — invert so larger
        # values correspond to further down the page for natural sort order.
        rows.append((col, -(y + h), x, text))

    rows.sort()
    return "\n".join(r[3] for r in rows)


def recognise_rows(jpeg_path: str | Path, *, row_tol: float = 0.012) -> str:
    return rows_from_annots(annotate(jpeg_path), row_tol=row_tol)


def recognise(jpeg_path: str | Path, *, gap: float = 0.10) -> str:
    return columns_from_annots(annotate(jpeg_path), gap=gap)


if __name__ == "__main__":
    import sys
    print(recognise(sys.argv[1]))
