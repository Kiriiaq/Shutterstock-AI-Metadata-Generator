# Media assets to produce

> Visual assets needed to finish the v2.1.0 launch (pivot Pro =
> évaluation qualité). None are blocking for code shipping, but they're
> required for the README, the GitHub release page, and the Gumroad
> listing. Notably : at least one screenshot **must show the Community
> quota banner + upsell screen** so the freemium pitch is legible at
> a glance.

---

## Priority P0 — README

| Asset | Target file | Format | Specs | Status |
|---|---|---|---|---|
| **Hero GIF** | `docs/media/demo_v2.gif` | GIF | 1100×720, ≤ 8 Mo, 12 fps, 25-30 s | 🟡 TODO |
| **Workspace screenshot** | `docs/media/workspace.png` | PNG | 1400×900, light theme | 🟡 TODO |
| **Workspace dark** | `docs/media/workspace_dark.png` | PNG | 1400×900, dark theme | 🟡 TODO |
| **Export Batch modal** (Community : « Les deux 🔒 Pro » + IA verrouillée) | `docs/media/export_batch.png` | PNG | 1100×780 modal capture | 🟡 TODO |
| **Expert Report modal** (bandeau quota « 🎁 1/2 ») | `docs/media/expert_report.png` | PNG | 1100×780 modal capture | 🟡 TODO |
| **Upsell screen** (rapport expert, quota épuisé) | `docs/media/expert_report_upsell.png` | PNG | 1100×780 modal capture | 🟡 TODO |

**Hero GIF storyboard** (25-30 s) :

1. `0–4 s` — fenêtre app vide, scan d'un dossier de 12 photos
2. `4–8 s` — DataTable se remplit, multi-sélection (5 fichiers)
3. `8–12 s` — clic « 📤 Exporter… » → modale s'ouvre
4. `12–18 s` — choix « Adobe + Shutterstock », case IPTC laissée OFF, lancer
5. `18–25 s` — barre de progression, badges ⏳ → ✅ par fichier, toast succès
6. `25–30 s` — fenêtre Explorer Windows ouverte sur les 2 CSV produits

Outil suggéré : **ScreenToGif** (gratuit, Windows, export optimisé).

---

## Priority P1 — Release page / Product Hunt

| Asset | Target file | Format | Specs | Status |
|---|---|---|---|---|
| **Icône Windows** | `assets/icons/icone.ico` | ICO | 256, 128, 64, 32, 16 | ✅ existe |
| **PNG carré social** | `docs/media/social_square.png` | PNG | 1080×1080, fond accent | 🟡 TODO |
| **Bannière header** | `docs/media/banner_1920.png` | PNG | 1920×400, fond gradient | 🟡 TODO |
| **Logo SVG vectoriel** | `assets/logo.svg` | SVG | 512×512 source | 🟡 TODO |
| **Vidéo démo longue** | `docs/media/demo_long.mp4` | MP4 (H.264) | 1080p, 60-90 s | 🟡 TODO |

---

## Priority P2 — Communication LinkedIn / X

| Asset | Format | Specs | Usage |
|---|---|---|---|
| **Carousel 8 slides** | PNG ×8 | 1080×1350 | Post storytelling, Phase 7 |
| **Slide hook** | PNG | 1080×1350 | Slide 1 du carousel |
| **Slide problème** | PNG | 1080×1350 | Slide 2 |
| **Slide architecture** | PNG | 1080×1350 | Slide 6 |
| **Code snippet hero** | PNG | 1080×1080 | Visible dans le pitch tech |
| **Avant/Après timing** | PNG | 1080×1080 | 5-10 min vs 3 s (heuristique) |

---

## Scripts / commandes utiles

### Capture rapide d'une fenêtre

```powershell
# ScreenToGif : https://www.screentogif.com/
# Ou via PowerShell + .NET pour une capture immédiate :
Add-Type -AssemblyName System.Drawing
$bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$bmp = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
$gfx = [System.Drawing.Graphics]::FromImage($bmp)
$gfx.CopyFromScreen($bounds.Location, [Drawing.Point]::Empty, $bounds.Size)
$bmp.Save("$env:USERPROFILE\Desktop\screen.png")
```

### Optimisation GIF post-capture

```powershell
# Avec ffmpeg installé :
ffmpeg -i raw.gif -vf "fps=12,scale=1100:-1:flags=lanczos" -loop 0 demo_v2.gif

# Compression supplémentaire avec gifsicle :
gifsicle -O3 --lossy=80 demo_v2.gif -o demo_v2_opt.gif
```

### Optimisation PNG

```powershell
# Avec pngquant :
pngquant --quality=80-95 --strip workspace.png

# Avec optipng :
optipng -o7 workspace.png
```

---

## Dataset de captures pour le storyboard

Le dossier `test/inputs/` contient 15 vraies images JPEG (générées par
Pillow) prêtes à servir d'images d'illustration pour le GIF :

- `input_nominal.jpg` — 12 MP, sRGB, IPTC complet → idéal pour la démo "happy path"
- `input_brands.jpg` — IPTC avec marques (filtrées par l'app) → preuve anti-stuffing
- `input_low_mp.jpg` — 2 MP → preuve du warning lax

Utiliser ces fichiers pour le GIF garantit un dataset reproductible et
non lié à des données perso.

---

## Charte visuelle

- **Couleur principale** : `#1f4e78` (bleu-gris foncé, alignée sur `palette.primary`)
- **Couleur OK** : `#16a34a`
- **Couleur warning** : `#d97706`
- **Couleur NOK** : `#dc2626`
- **Font UI** : Segoe UI (Windows), fallback `system-ui`
- **Font code** : Cascadia Code, fallback `Consolas`, `monospace`
- **Style** : flat, peu d'ombres, beaucoup de blanc/gris-bleu doux

---

## Vérification finale avant push GitHub

- [ ] GIF hero ≤ 8 Mo (limite GitHub README pour rendu fluide)
- [ ] PNGs ≤ 500 Ko chacun (optimisés pngquant)
- [ ] Texte alternatif (`alt="..."`) sur chaque `<img>` dans README
- [ ] Liens relatifs (`docs/media/…`), pas absolus
- [ ] Mentions LinkedIn faites avec lien GitHub (vérifier l'URL canonique)

---

*Fichier vivant — mis à jour à chaque ajout d'asset.*
