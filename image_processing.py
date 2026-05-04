from pathlib import Path
import shutil

import numpy as np
from PIL import Image

OUT_DIR     = Path("pages")
BOTTOM_CROP = 230
XTRIM       = 60


def estimate_skew(
    img: Image.Image,
    angle_range: float = 3.0,
    coarse_step: float = 0.5,
    fine_step: float = 0.05,
    sample_max_dim: int = 1000,
    min_confidence: float = 1.05,
) -> float:
    """Estimate skew angle (degrees) of a primarily-text page using a
    horizontal-projection variance score. Negative means the image is rotated
    clockwise from upright; rotating by ``+angle`` straightens it.

    The score metric is the variance of horizontal pixel sums after rotating
    by the candidate angle: when text rows align with the image rows the
    sums alternate sharply between text-row peaks and white-row troughs,
    maximising variance.

    `angle_range` caps the search; pages on a flatbed scanner are rarely
    off by more than a couple of degrees and a tight cap protects against
    pathological pages (e.g. a fold-out map with diagonal street labels)
    hijacking the metric. `min_confidence` is the minimum ratio of the
    best score to the score at 0°; if the search can't find a clearly
    better angle the page is treated as already straight.
    """
    gray = img.convert("L")
    scale = sample_max_dim / max(gray.size)
    if scale < 1.0:
        gray = gray.resize(
            (int(gray.width * scale), int(gray.height * scale)),
            Image.BILINEAR,
        )
    arr = np.asarray(gray)
    # Binarise: text dark on light.
    binary = (arr < arr.mean() - 10).astype(np.uint8) * 255
    bin_img = Image.fromarray(binary)

    def score(angle: float) -> float:
        rot = bin_img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
        rows = np.asarray(rot).sum(axis=1, dtype=np.int64)
        return float(np.var(rows))

    coarse = np.arange(-angle_range, angle_range + coarse_step, coarse_step)
    best = max(coarse, key=score)
    lo = max(-angle_range, best - coarse_step)
    hi = min(angle_range, best + coarse_step)
    fine = np.arange(lo, hi + fine_step, fine_step)
    best_angle = float(max(fine, key=score))

    s0 = score(0.0)
    sb = score(best_angle)
    if s0 > 0 and sb / s0 < min_confidence:
        return 0.0
    return best_angle


def deskew(img: Image.Image, threshold: float = 0.1) -> tuple[Image.Image, float]:
    """Return a deskewed copy of `img` and the angle applied (degrees)."""
    angle = estimate_skew(img)
    if abs(angle) < threshold:
        return img, 0.0
    rotated = img.rotate(
        angle, resample=Image.BICUBIC, fillcolor="white", expand=False,
    )
    return rotated, angle


def scan_section(section_path, first_page) -> int:

    def scan_path(index: int) -> Path:
        """Return the file path for scan index `index` (0-based)."""
        return section_path / ("Scan.jpeg" if index == 0 else f"Scan {index}.jpeg")

    num_scans = len(list(section_path.glob('*.jpeg')))
    page_count = num_scans * 2

    for k in range(num_scans):
        src = scan_path(k)
        low_page = first_page + k       # lower-numbered page (was on top)
        high_page = first_page + page_count - 1 - k       # higher-numbered page (was on bottom)

        with Image.open(src) as img:
            # Trim the scanner artifact along the bottom edge so the page
            # content is centred and the halves are equal width.
            w0, h0 = img.size
            trimmed = img.crop((0, 0, w0, h0 - BOTTOM_CROP))

            # Even scans need a 90° CW rotation; odd scans were placed on the
            # scanner the other way up and need a 90° CCW rotation.
            if k % 2 == 0:
                rotated = trimmed.rotate(-90, expand=True)
                # CW: top-of-original → right of rotated.
                low_side, high_side = "right", "left"
            else:
                rotated = trimmed.rotate(90, expand=True)
                # CCW: top-of-original → left of rotated.
                low_side, high_side = "left", "right"

            # Pages were never placed perfectly square on the scanner — apply
            # a small projection-profile deskew to straighten the text lines.
            rotated, skew_angle = deskew(rotated)

            w, h = rotated.size
            mid = w // 2
            left = rotated.crop((XTRIM, 0, mid - XTRIM, h))
            right = rotated.crop((mid + XTRIM, 0, w - XTRIM, h))

            low_img = right if low_side == "right" else left
            high_img = left if high_side == "left" else right

            low_img.save(OUT_DIR / f"page_{low_page:03d}.jpg",
                         "JPEG", quality=95)
            high_img.save(OUT_DIR / f"page_{high_page:03d}.jpg",
                          "JPEG", quality=95)

        print(f"{src.name}: page {low_page:03d} ({low_side}) + "
              f"page {high_page:03d} ({high_side})  skew {skew_angle:+.2f}°")

    page_num = first_page + page_count
    for page in sorted(list(section_path.glob('*.jpg'))):
        shutil.copy(page, OUT_DIR / f"page_{page_num:03d}.jpg")
        page_num += 1

    return page_num

def scan_sections(sections_path):
    OUT_DIR.mkdir(exist_ok=True)

    first_page = 1
    for section in sorted(list(sections_path.iterdir())):
        first_page = scan_section(section, first_page)
