"""Interactive screenshot helper for the v2 launch.

Launches the app and pauses at each step so the user can :
1. Position the window where they want.
2. Trigger the screenshot via the 'Capture' button (or Enter in the
   small helper window).
3. Continue to the next screenshot.

Captures the **whole primary screen** to ``docs/media/`` then crops to
the app window bounding box. PIL's ImageGrab works on Windows + macOS
out of the box; no extra dependency.

Sequence captured :
- ``workspace.png``           — main view, light theme
- ``workspace_dark.png``      — main view, dark theme
- ``expert_report.png``       — Rapport expert modal
- ``export_batch.png``        — Exporter le lot modal

Usage::

    python tools\take_screenshots.py

The script prints what the user has to click before each capture.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from tkinter import Button, Label, Tk

try:
    from PIL import ImageGrab
except ImportError:
    print("PIL/Pillow manquant. pip install Pillow", file=sys.stderr)
    sys.exit(1)

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "media"
OUT_DIR.mkdir(parents=True, exist_ok=True)


CAPTURES = [
    {
        "name": "workspace.png",
        "instructions": (
            "1. Lance manuellement l'app (dist\\ShutterstockAnalyzer.exe).\n"
            "2. Scanne un dossier test (test\\inputs\\ par exemple).\n"
            "3. Theme = LIGHT (clic icône lune si nécessaire).\n"
            "4. Place la fenêtre où tu veux qu'elle apparaisse dans la capture.\n"
            "5. Clique 'Capturer maintenant' ci-dessous (3 sec de délai).",
        ),
    },
    {
        "name": "workspace_dark.png",
        "instructions": (
            "1. Bascule en theme DARK via le bouton lune.\n"
            "2. Place la fenêtre identiquement.\n"
            "3. Clique 'Capturer maintenant'."
        ),
    },
    {
        "name": "expert_report.png",
        "instructions": (
            "1. Sélectionne une image dans Sources.\n"
            "2. Double-clic dessus → l'éditeur IPTC se remplit.\n"
            "3. Clique 'Rapport expert…' → la modale s'ouvre.\n"
            "4. Attends 2 sec que le rapport se calcule.\n"
            "5. Clique 'Capturer maintenant'."
        ),
    },
    {
        "name": "export_batch.png",
        "instructions": (
            "1. Sélectionne 5-10 fichiers (Ctrl+clic).\n"
            "2. Clique '📤 Exporter…' → modale s'ouvre.\n"
            "3. (optionnel) Coche 'Enrichir avec IA' pour révéler le bandeau Ollama.\n"
            "4. (optionnel) Coche 'Pousser en FTP' pour révéler les champs FTP.\n"
            "5. Clique 'Capturer maintenant'."
        ),
    },
]


def grab_screen(out: Path, delay_s: float = 3.0) -> None:
    """Capture the whole primary screen after a short delay."""
    for sec in range(int(delay_s), 0, -1):
        print(f"  Capture dans {sec}…", flush=True)
        time.sleep(1)
    img = ImageGrab.grab(all_screens=False)
    img.save(out, "PNG", optimize=True)
    print(f"  ✓ Sauvé : {out.relative_to(out.parent.parent)}  ({img.size[0]}×{img.size[1]})")


def main() -> int:
    print("=" * 70)
    print("  Helper de capture d'écran — ShutterstockAnalyzer v2")
    print("=" * 70)
    print()

    for capture in CAPTURES:
        out_path = OUT_DIR / capture["name"]
        print(f"\n→ Capture : {capture['name']}")
        if isinstance(capture["instructions"], tuple):
            inst = capture["instructions"][0]
        else:
            inst = capture["instructions"]
        print(inst)
        print()

        # Mini fenêtre helper
        root = Tk()
        root.title("Capture helper")
        root.geometry("420x220+100+100")
        root.attributes("-topmost", True)

        Label(root, text=f"Prochaine capture : {capture['name']}",
              font=("Segoe UI", 11, "bold")).pack(pady=(15, 5))
        Label(root, text="Lis les instructions dans la console puis clique.",
              font=("Segoe UI", 9), fg="gray").pack(pady=(0, 10))

        done = {"clicked": False}

        def on_capture():
            done["clicked"] = True
            root.withdraw()  # cacher la helper window avant la capture
            root.update_idletasks()
            time.sleep(0.4)
            grab_screen(out_path, delay_s=3.0)
            root.destroy()

        def on_skip():
            done["clicked"] = False
            root.destroy()

        Button(root, text="Capturer maintenant", command=on_capture,
               bg="#1f4e78", fg="white", font=("Segoe UI", 10, "bold"),
               height=2, width=25).pack(pady=10)
        Button(root, text="Passer ce screenshot", command=on_skip,
               font=("Segoe UI", 9), width=25).pack()

        root.mainloop()

        if not done["clicked"]:
            print(f"  ⊘ Sauté : {capture['name']}")

    print("\n" + "=" * 70)
    print("  Captures terminées. Fichiers dans : docs/media/")
    print("=" * 70)

    # Liste finale
    print("\nFichiers présents :")
    for f in sorted(OUT_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:35} {size_kb:8.1f} Ko")

    return 0


if __name__ == "__main__":
    sys.exit(main())
