#!/usr/bin/env python3
"""Generate the BCD AppIcon set from the three tier SVGs in this folder.

Renders each SVG to a 1024 master with macOS `qlmanage`, flattens RGBA->RGB
(App Store icons must not carry an alpha channel), then Lanczos-downscales to
every AppIcon size. Size picks the tier so the mark simplifies as it shrinks:

  >= 120px   three vessels  (icon-hero.svg)
  76..119px  bottle + draft (icon-med.svg)
  <= 64px    single bottle  (icon-small.svg)

Requires Pillow:  python3 -m pip install pillow
Run from repo root:  python3 brand/generate_appicon.py
"""
import os
import subprocess
import tempfile
from PIL import Image

BRAND = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BRAND)
OUT = os.path.join(REPO, "ios/BCDApp/Resources/Assets.xcassets/AppIcon.appiconset")
TIERS = {"hero": "icon-hero.svg", "med": "icon-med.svg", "small": "icon-small.svg"}
# Every distinct pixel size referenced by Contents.json (18 slots share these 12 files).
SIZES = [20, 29, 40, 58, 60, 80, 87, 120, 152, 167, 180, 1024]


def tier_for(size):
    if size <= 64:
        return "small"
    if size < 120:
        return "med"
    return "hero"


def main():
    os.makedirs(OUT, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="bcd-icons-")
    masters = {}
    for tier, svg in TIERS.items():
        subprocess.run(
            ["qlmanage", "-t", "-s", "1024", "-o", tmp, os.path.join(BRAND, svg)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        im = Image.open(os.path.join(tmp, svg + ".png")).convert("RGB")
        if im.size != (1024, 1024):
            im = im.resize((1024, 1024), Image.LANCZOS)
        masters[tier] = im
    for s in SIZES:
        dst = os.path.join(OUT, f"icon_{s}.png")
        masters[tier_for(s)].resize((s, s), Image.LANCZOS).save(dst, "PNG")
    print(f"wrote {len(SIZES)} PNGs to {OUT}")


if __name__ == "__main__":
    main()
