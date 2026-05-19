"""Generate test/inputs/* — realistic test images via Pillow.

Each file targets one row of the test matrix. Run once from the
repo root::

    python test/scripts/_make_inputs.py

Produces:
    inputs/input_nominal.jpg                    12 MP, sRGB, ~1-3 MB, IPTC complete
    inputs/input_vide.jpg                       0 bytes
    inputs/input_low_mp.jpg                     1600x1250 ≈ 2 MP
    inputs/input_volumineux.jpg                 ~52 MB JPEG
    inputs/input_mauvais_format.png             PNG (not JPEG)
    inputs/input_cmyk.jpg                       CMYK colour space
    inputs/input_utf8.jpg                       IPTC with accents + emoji + I&C symbols
    inputs/input_corrompu.jpg                   truncated header
    inputs/input_no_keywords.jpg                IPTC sans keywords
    inputs/input_many_keywords.jpg              IPTC with 80 keywords
    inputs/input_no_title.jpg                   IPTC without headline/object_name
    inputs/input_brands.jpg                     IPTC keywords contain brands
    inputs/input_stuffing.jpg                   stuffing keywords (no title overlap)
    inputs/input_stuffing_in_title.jpg          stuffing keywords (in title)
    Dossier été/photo n°1.jpg                   path with spaces + accents

IPTC writing requires ExifTool. If ExifTool is missing the images
are produced but without IPTC — tests T-009/T-010/T-036/T-053 etc.
will need to be re-run after installing ExifTool.
"""

from __future__ import annotations

import io
import logging
import math
import random
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "inputs"


def _gradient_image(width: int, height: int, *, seed: int = 0, mode: str = "RGB") -> Image.Image:
    """Cheap colourful gradient. Deterministic for reproducibility."""
    random.seed(seed)
    img = Image.new(mode, (width, height))
    pixels = img.load()
    cx, cy = width / 2, height / 2
    max_dist = math.sqrt(cx * cx + cy * cy)
    base_r = random.randint(40, 200)
    base_g = random.randint(40, 200)
    base_b = random.randint(40, 200)
    for y in range(height):
        for x in range(width):
            d = math.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max_dist
            r = int(base_r + (255 - base_r) * d)
            g = int(base_g + (255 - base_g) * (1 - d))
            b = int(base_b * (1 - d))
            if mode == "RGB":
                pixels[x, y] = (r, g, b)
            elif mode == "CMYK":
                # Approximate CMYK from RGB
                c = 255 - r
                m = 255 - g
                yy = 255 - b
                k = min(c, m, yy)
                pixels[x, y] = (c - k, m - k, yy - k, k)
    # Add a title band for legibility
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", max(20, width // 60))
    except OSError:
        font = ImageFont.load_default()
    text = f"TEST {width}x{height}"
    draw.text((20, 20), text, fill="white", font=font)
    return img


def _write_jpeg(img: Image.Image, path: Path, *, quality: int = 92) -> None:
    if img.mode == "CMYK":
        img.save(path, "JPEG", quality=quality)
    else:
        img.save(path, "JPEG", quality=quality, optimize=True)


def _exiftool() -> str | None:
    candidates = [
        shutil.which("exiftool"),
        "C:\\Program Files\\ExifTool\\exiftool.exe",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    return None


def _write_iptc(path: Path, *, headline: str = "", caption: str = "",
                keywords: list[str] | None = None,
                supplemental: list[str] | None = None,
                byline: str = "Test Suite") -> bool:
    """Write IPTC via ExifTool. Returns True on success."""
    exe = _exiftool()
    if exe is None:
        return False
    args = [exe, "-overwrite_original", "-charset", "iptc=utf8"]
    if headline:
        args += [f"-IPTC:Headline={headline}", f"-IPTC:ObjectName={headline[:64]}"]
    if caption:
        args += [f"-IPTC:Caption-Abstract={caption}"]
    if keywords:
        for kw in keywords:
            args += [f"-IPTC:Keywords={kw}"]
    if supplemental:
        for sc in supplemental:
            args += [f"-IPTC:SupplementalCategories={sc}"]
    if byline:
        args += [f"-IPTC:By-line={byline}"]
    args.append(str(path))
    res = subprocess.run(args, capture_output=True, text=True)
    if res.returncode != 0:
        logger.warning("exiftool failed on %s: %s", path, res.stderr)
        return False
    return True


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def make_nominal() -> Path:
    """12 MP JPEG, sRGB, IPTC complete."""
    img = _gradient_image(4000, 3000, seed=1)
    p = OUT / "input_nominal.jpg"
    _write_jpeg(img, p)
    _write_iptc(p,
                headline="Vibrant business team brainstorming in modern office",
                caption="Diverse colleagues collaborate on a digital marketing strategy.",
                keywords=["business", "team", "office", "brainstorm", "modern",
                          "diverse", "marketing", "strategy", "collaboration",
                          "meeting", "professional", "corporate"],
                supplemental=["Business/Finance"])
    return p


def make_empty() -> Path:
    p = OUT / "input_vide.jpg"
    p.write_bytes(b"")
    return p


def make_low_mp() -> Path:
    img = _gradient_image(1600, 1250, seed=2)  # ~2 MP
    p = OUT / "input_low_mp.jpg"
    _write_jpeg(img, p)
    _write_iptc(p, headline="Low resolution test image",
                keywords=["nature", "test", "small", "landscape", "scene",
                          "outdoor", "background"])
    return p


def make_large() -> Path:
    """JPEG > 50 MB. Strategy: 10000x8000 with quality=100 and no optimize."""
    img = _gradient_image(10000, 8000, seed=3)
    p = OUT / "input_volumineux.jpg"
    # Quality 100 + no optimize keeps the file huge.
    img.save(p, "JPEG", quality=100, optimize=False, subsampling=0)
    size_mb = p.stat().st_size / (1024 * 1024)
    print(f"  volumineux = {size_mb:.1f} MB")
    if size_mb < 45:
        # Pad with arbitrary bytes at the end to push past 50 MB. JPEG
        # decoders ignore trailing data after the EOI marker (FF D9).
        padding = int((52 * 1024 * 1024) - p.stat().st_size)
        if padding > 0:
            with p.open("ab") as fh:
                fh.write(b"\x00" * padding)
        print(f"  padded to {p.stat().st_size/(1024*1024):.1f} MB")
    return p


def make_png() -> Path:
    img = _gradient_image(3000, 2000, seed=4)
    p = OUT / "input_mauvais_format.png"
    img.save(p, "PNG")
    return p


def make_cmyk() -> Path:
    img = _gradient_image(3000, 2000, seed=5, mode="CMYK")
    p = OUT / "input_cmyk.jpg"
    _write_jpeg(img, p)
    return p


def make_utf8() -> Path:
    img = _gradient_image(3000, 2000, seed=6)
    p = OUT / "input_utf8.jpg"
    _write_jpeg(img, p)
    _write_iptc(
        p,
        headline="Capteur de température ±0.5°C — caméra μmétrique Ω 🎨",
        caption="Test caractères spéciaux : Çà éü öß äÖ — symboles : ° ± μ Ω ∞",
        keywords=["température", "caméra", "précision", "instrument",
                  "mesure", "industriel", "technique", "métrologie"],
        supplemental=["Industrial"],
    )
    return p


def make_corrupt() -> Path:
    img = _gradient_image(3000, 2000, seed=7)
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    truncated = buf.getvalue()[:200]  # only header
    p = OUT / "input_corrompu.jpg"
    p.write_bytes(truncated)
    return p


def make_no_keywords() -> Path:
    img = _gradient_image(3000, 2000, seed=8)
    p = OUT / "input_no_keywords.jpg"
    _write_jpeg(img, p)
    _write_iptc(p, headline="Image without keywords",
                caption="Used to test the missing-keywords blocker.")
    return p


def make_many_keywords() -> Path:
    img = _gradient_image(3000, 2000, seed=9)
    p = OUT / "input_many_keywords.jpg"
    _write_jpeg(img, p)
    keywords = [f"keyword{i:02d}" for i in range(80)]
    _write_iptc(p, headline="Image with 80 keywords (should cap at 50)",
                keywords=keywords)
    return p


def make_no_title() -> Path:
    img = _gradient_image(3000, 2000, seed=10)
    p = OUT / "input_no_title.jpg"
    _write_jpeg(img, p)
    _write_iptc(p, caption="Description only, no title.",
                keywords=["scene", "background", "abstract", "design",
                          "minimal", "clean", "neutral"])
    return p


def make_brands() -> Path:
    img = _gradient_image(3000, 2000, seed=11)
    p = OUT / "input_brands.jpg"
    _write_jpeg(img, p)
    _write_iptc(p, headline="Runner training outdoor",
                caption="Sport scene with brand references in keywords.",
                keywords=["nike", "apple", "iphone", "coca-cola", "running",
                          "sport", "outdoor", "fitness", "training", "athlete"])
    return p


def make_stuffing() -> Path:
    img = _gradient_image(3000, 2000, seed=12)
    p = OUT / "input_stuffing.jpg"
    _write_jpeg(img, p)
    _write_iptc(p, headline="Sunset over alpine lake",
                caption="Calm evening light over a remote mountain lake.",
                keywords=["stock", "image", "wallpaper", "background", "photo",
                          "picture", "sunset", "lake", "mountain", "nature"])
    return p


def make_stuffing_in_title() -> Path:
    img = _gradient_image(3000, 2000, seed=13)
    p = OUT / "input_stuffing_in_title.jpg"
    _write_jpeg(img, p)
    _write_iptc(p, headline="Lake Photo at Sunset",
                caption="Lake photography practice with classic golden hour light.",
                keywords=["photo", "lake", "sunset", "landscape", "nature",
                          "outdoor", "calm", "water", "reflection"])
    return p


def make_path_with_spaces() -> Path:
    img = _gradient_image(3000, 2000, seed=14)
    subdir = OUT / "Dossier été"
    subdir.mkdir(exist_ok=True)
    p = subdir / "photo n°1.jpg"
    _write_jpeg(img, p)
    _write_iptc(p, headline="Image dans chemin avec accents et espaces",
                keywords=["test", "unicode", "chemin", "accent", "espace",
                          "robustesse", "filesystem"])
    return p


# ---------------------------------------------------------------------------


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    generators = [
        make_nominal, make_empty, make_low_mp, make_large, make_png,
        make_cmyk, make_utf8, make_corrupt, make_no_keywords,
        make_many_keywords, make_no_title, make_brands, make_stuffing,
        make_stuffing_in_title, make_path_with_spaces,
    ]
    print(f"Generating {len(generators)} test inputs into {OUT} …")
    for gen in generators:
        p = gen()
        size = p.stat().st_size if p.exists() else 0
        print(f"  {p.relative_to(ROOT)}  ({size:,} bytes)")
    print("Done.")


if __name__ == "__main__":
    main()
