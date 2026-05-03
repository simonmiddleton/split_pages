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
    Coordinates are fractional with origin bottom-left. One entry per
    Vision-recognised line.
    """
    from ocrmac import ocrmac
    return ocrmac.OCR(str(jpeg_path), recognition_level="accurate").recognize()


def annotate_words(jpeg_path: str | Path):
    """Return Apple Vision OCR annotations split to one entry per word.

    Each entry is ``(text, conf, [x, y, w, h], line_id)`` where the bbox
    is fractional / origin bottom-left (same convention as ``annotate``)
    and ``line_id`` is the index of the parent Vision line observation
    (so callers can regroup words back into lines for sidecar text).

    Vision returns line-level observations; per-word bounding boxes come
    from ``VNRecognizedText.boundingBoxForRange:error:``, which we call
    directly via pyobjc since ``ocrmac`` doesn't expose it. Each word's
    bbox is the axis-aligned bounding rect of the (possibly rotated)
    quad that Vision returns.

    Words are emitted in left-to-right order within each line and in
    Vision's natural top-to-bottom line order, so the existing
    ``sort_rows`` / ``sort_columns`` sort keys (y-then-x) still produce
    sensible reading order; words on the same line share roughly the
    same y and end up adjacent after sorting.
    """
    import io
    import re
    import objc
    import Vision  # noqa: F401  (provided by pyobjc-framework-vision)
    from PIL import Image

    img = Image.open(str(jpeg_path))
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    out: list[tuple[str, float, list[float], int]] = []
    with objc.autorelease_pool():
        req = Vision.VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLevel_(0)  # 0 = accurate, 1 = fast
        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(
            buf.getvalue(), None
        )
        handler.performRequests_error_([req], None)
        for line_id, obs in enumerate(req.results() or []):
            confidence = obs.confidence()
            cands = obs.topCandidates_(1)
            if not cands:
                continue
            rec = cands[0]
            full = rec.string()
            for m in re.finditer(r"\S+", full):
                word = m.group()
                rng = (m.start(), m.end() - m.start())
                res = rec.boundingBoxForRange_error_(rng, None)
                quad = res[0] if isinstance(res, tuple) else res
                if quad is None:
                    continue
                pts = (quad.topLeft(), quad.topRight(),
                       quad.bottomLeft(), quad.bottomRight())
                xs = [p.x for p in pts]
                ys = [p.y for p in pts]
                x = min(xs)
                y = min(ys)
                w = max(xs) - x
                h = max(ys) - y
                if w <= 0 or h <= 0:
                    continue
                out.append((word, confidence, [x, y, w, h], line_id))
    return out


def join_word_lines(annots) -> str:
    """Regroup per-word annots back into lines for sidecar text.

    Walks ``annots`` (already in reading order) and joins consecutive
    words that share the same ``line_id`` with a single space; lines
    are separated by ``\\n``. Falls back to one fragment per line for
    annots without a line_id (i.e. the line-level shape from
    ``annotate``).
    """
    lines: list[str] = []
    current: list[str] = []
    current_id: object = object()  # sentinel — first word starts a new line
    for a in annots:
        line_id = a[3] if len(a) > 3 else id(a)  # unique → no grouping
        if line_id != current_id and current:
            lines.append(" ".join(current))
            current = []
        current.append(a[0])
        current_id = line_id
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


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


def _group_words_by_line(word_annots):
    """Group consecutive word annots by line_id (preserves word order)."""
    from itertools import groupby
    return [(lid, list(grp))
            for lid, grp in groupby(word_annots, key=lambda a: a[3])]


def _line_bbox(words):
    """Union bbox of the words in a line: [x, y, w, h]."""
    xs = [w[2][0] for w in words]
    ys = [w[2][1] for w in words]
    rs = [w[2][0] + w[2][2] for w in words]
    ts = [w[2][1] + w[2][3] for w in words]
    x, y = min(xs), min(ys)
    return [x, y, max(rs) - x, max(ts) - y]


def sort_columns_words(word_annots, *, gap: float = 0.10):
    """Column-aware reading order over word-level annots.

    Column assignment is done per LINE (using each line's union bbox)
    so all words on the same line stay together; words within a line
    are emitted in their original left-to-right order.
    """
    if not word_annots:
        return []
    groups = _group_words_by_line(word_annots)
    # Build line annots that share the same shape as word annots so the
    # existing sort_columns can sort them; line_id is preserved at [3].
    line_annots = [
        (words[0][0], words[0][1], _line_bbox(words), lid)
        for lid, words in groups
    ]
    by_lid = {lid: words for lid, words in groups}
    out = []
    for la in sort_columns(line_annots, gap=gap):
        out.extend(by_lid[la[3]])
    return out


def sort_rows_words(word_annots, *, row_tol: float = 0.012):
    """Row-aware grouping over word-level annots.

    Row clustering is done per LINE; each returned row is a flat list
    of words from all lines in that row, in left-to-right order
    (lines within a row sorted by x, words within each line in their
    original order).
    """
    if not word_annots:
        return []
    groups = _group_words_by_line(word_annots)
    line_annots = [
        (words[0][0], words[0][1], _line_bbox(words), lid)
        for lid, words in groups
    ]
    by_lid = {lid: words for lid, words in groups}
    out_rows = []
    for row in sort_rows(line_annots, row_tol=row_tol):
        row_words = []
        for la in row:
            row_words.extend(by_lid[la[3]])
        out_rows.append(row_words)
    return out_rows


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
    centres = [a[2][0] + a[2][2] / 2 for a in annots if a[2][2] > 0.02]
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
