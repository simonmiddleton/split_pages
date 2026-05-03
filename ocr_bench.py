#!/usr/bin/env python3
"""Quick OCR benchmark on a few representative pages.

Outputs plain-text recognition results for several engines/configurations
into ./ocr_bench/<page>__<engine>.txt so they can be compared side-by-side.
"""
from __future__ import annotations

from pathlib import Path
import os
import subprocess
import time

from PIL import Image

PAGES = ["page_001.jpg", "page_050.jpg", "page_200.jpg"]
SRC = Path("pages")
OUT = Path("ocr_bench")
OUT.mkdir(exist_ok=True)

TESSDATA_DEFAULT = "/opt/homebrew/share/tessdata"
TESSDATA_BEST = "/tmp/tessdata_best"


def run_tess(jpeg: Path, tessdata: str, psm: int, tag: str) -> tuple[str, float]:
    out_base = OUT / f"{jpeg.stem}__{tag}"
    env = {**os.environ, "TESSDATA_PREFIX": tessdata}
    t0 = time.perf_counter()
    subprocess.run(
        [
            "tesseract", str(jpeg), str(out_base),
            "-l", "eng",
            "--dpi", "300",
            "--psm", str(psm),
        ],
        check=True, env=env, capture_output=True,
    )
    elapsed = time.perf_counter() - t0
    return out_base.with_suffix(".txt").read_text(), elapsed


def run_apple_vision(jpeg: Path, tag: str) -> tuple[str, float]:
    from ocrmac import ocrmac
    t0 = time.perf_counter()
    annots = ocrmac.OCR(str(jpeg), recognition_level="accurate").recognize()
    elapsed = time.perf_counter() - t0
    # Naive sort: top-to-bottom, then left-to-right, ignoring columns.
    rows = sorted(annots, key=lambda a: (-a[2][1] - a[2][3], a[2][0]))
    text = "\n".join(t for t, _, _ in rows)
    out = OUT / f"{jpeg.stem}__{tag}.txt"
    out.write_text(text)
    return text, elapsed


def run_apple_vision_columns(jpeg: Path, tag: str) -> tuple[str, float]:
    import ocr_av
    t0 = time.perf_counter()
    text = ocr_av.recognise(jpeg)
    elapsed = time.perf_counter() - t0
    out = OUT / f"{jpeg.stem}__{tag}.txt"
    out.write_text(text)
    return text, elapsed


def run_apple_vision_rows(jpeg: Path, tag: str) -> tuple[str, float]:
    import ocr_av
    t0 = time.perf_counter()
    text = ocr_av.recognise_rows(jpeg)
    elapsed = time.perf_counter() - t0
    out = OUT / f"{jpeg.stem}__{tag}.txt"
    out.write_text(text)
    return text, elapsed


def main() -> None:
    print(f"{'page':>14} {'engine':>30} {'chars':>7} {'secs':>7}")
    for name in PAGES:
        jpeg = SRC / name
        configs = [
            ("tess_eng_psm3",  lambda j: run_tess(j, TESSDATA_DEFAULT, 3, "tess_eng_psm3")),
            ("tess_best_psm3", lambda j: run_tess(j, TESSDATA_BEST,    3, "tess_best_psm3")),
            ("tess_best_psm6", lambda j: run_tess(j, TESSDATA_BEST,    6, "tess_best_psm6")),
            ("apple_vision",   lambda j: run_apple_vision(j, "apple_vision")),
            ("apple_vision_cols", lambda j: run_apple_vision_columns(j, "apple_vision_cols")),
            ("apple_vision_rows", lambda j: run_apple_vision_rows(j, "apple_vision_rows")),
        ]
        for tag, fn in configs:
            try:
                text, secs = fn(jpeg)
                print(f"{name:>14} {tag:>30} {len(text):>7} {secs:>7.2f}")
            except Exception as e:  # noqa: BLE001
                print(f"{name:>14} {tag:>30}  ERROR: {e}")


if __name__ == "__main__":
    main()
