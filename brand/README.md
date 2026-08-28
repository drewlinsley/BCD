# BCD brand assets

The app identity: **Reticle × Vessels** — an amber camera scan-frame locking onto
the three formats BCD reads (bottle, can, draft), on a dark HUD ground.

## Palette

| Role        | Hex       |
|-------------|-----------|
| Ink (ground)| `#17161D` (radial `#24222F` → `#100F16`) |
| Amber       | `#E9A23A` |
| Gold (meniscus) | `#F6C979` |
| Cream (glass)   | `#F3EADB` |
| Can lid (shade) | `#D9CBB0` |

## Icon system — graduated reduction

The mark simplifies as the icon shrinks, so it stays legible at every size:

| Tier   | Source              | Used at   | Shows                       |
|--------|---------------------|-----------|-----------------------------|
| hero   | `icon-hero.svg`     | ≥ 120 px  | bottle · can · draft, filled |
| med    | `icon-med.svg`      | 76–119 px | bottle · draft, filled       |
| small  | `icon-small.svg`    | ≤ 64 px   | single bottle, filled        |

These SVGs are the source of truth — edit them, then regenerate the PNGs.

## Wordmark & lockup

The logo lockup pairs the app-icon mark with the `BCD` wordmark and the
`Bottles · Cans · Draft` tagline (amber `·` separators).

| File | Use |
|------|-----|
| `wordmark-horizontal-dark.svg`  | horizontal lockup for dark grounds (cream text) |
| `wordmark-horizontal-light.svg` | horizontal lockup for light grounds (ink text) |
| `wordmark-stacked-dark.svg`     | stacked lockup for dark grounds |

Backgrounds are transparent so each lockup drops onto any surface. The `BCD`
wordmark and tagline are **outlined vector paths** — extracted from SF Pro
Display 800 and SF Pro Text 500 — so the SVGs render identically anywhere with
no font dependency (browsers, print RIPs, any OS). To restyle the wordmark, edit
the paths in a vector tool, or ask and I'll regenerate it.

## Regenerate the AppIcon set

```bash
python3 -m pip install pillow   # one-time
python3 brand/generate_appicon.py
```

Output lands in `ios/BCDApp/Resources/Assets.xcassets/AppIcon.appiconset/`
(12 PNGs, no alpha channel). `Contents.json` there maps the 18 iPhone/iPad/
marketing slots onto those files. The catalog is wired to the app via
`ASSETCATALOG_COMPILER_APPICON_NAME: AppIcon` in `ios/project.yml`; run
`make ios-gen` to regenerate the Xcode project after any change.

Rasterization uses macOS `qlmanage` (WebKit SVG) + Pillow for the RGB flatten
and Lanczos downscales.
