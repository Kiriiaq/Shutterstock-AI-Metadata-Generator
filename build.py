#!/usr/bin/env python3
"""
Script de build pour Shutterstock AI Metadata Generator
Crée des exécutables Windows avec PyInstaller

Usage:
    python build.py          # Build version release
    python build.py --debug  # Build version debug (avec console)
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path

# Configuration
APP_NAME = "ShutterstockAI-MetadataGenerator"
VERSION = "1.0.0"
MAIN_SCRIPT = "shutterstock_analyzer_unified.py"

# Dossiers
BUILD_DIR = Path("build")
DIST_DIR = Path("dist")


def check_pyinstaller():
    """Vérifie si PyInstaller est installé"""
    try:
        import PyInstaller
        print(f"PyInstaller {PyInstaller.__version__} trouvé")
        return True
    except ImportError:
        print("PyInstaller non trouvé. Installation...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"])
        return True


def clean_build():
    """Nettoie les dossiers de build"""
    for folder in [BUILD_DIR, DIST_DIR]:
        if folder.exists():
            print(f"Nettoyage de {folder}...")
            shutil.rmtree(folder)


def build_executable(debug: bool = False):
    """Construit l'exécutable"""

    suffix = "_debug" if debug else ""
    exe_name = f"{APP_NAME}-v{VERSION}{suffix}"

    print(f"\n{'='*60}")
    print(f"Construction de {exe_name}...")
    print(f"Mode: {'DEBUG (avec console)' if debug else 'RELEASE (sans console)'}")
    print(f"{'='*60}\n")

    # Options PyInstaller
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", exe_name,
        "--onefile",  # Un seul fichier .exe
        "--clean",    # Nettoie le cache
    ]

    # Mode fenêtre ou console
    if debug:
        cmd.append("--console")  # Affiche la console pour debug
    else:
        cmd.append("--windowed")  # Pas de console

    # Imports cachés nécessaires
    hidden_imports = [
        "PIL",
        "PIL.Image",
        "PIL.ImageOps",
        "PIL.ExifTags",
        "piexif",
        "ollama",
        "pydantic",
        "pydantic.fields",
        "customtkinter",
        "CTkToolTip",
        "requests",
        "urllib3",
        "tkinter",
        "tkinter.messagebox",
        "tkinter.filedialog",
    ]

    for imp in hidden_imports:
        cmd.extend(["--hidden-import", imp])

    # Exclure les modules inutiles pour réduire la taille
    excludes = [
        "matplotlib",
        "numpy",
        "pandas",
        "scipy",
        "tensorflow",
        "torch",
        "pytest",
        "IPython",
        "notebook",
        "jupyter",
    ]

    for exc in excludes:
        cmd.extend(["--exclude-module", exc])

    # Script principal
    cmd.append(MAIN_SCRIPT)

    # Exécuter PyInstaller
    print("Commande:", " ".join(cmd[:10]), "...")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        exe_path = DIST_DIR / f"{exe_name}.exe"
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print(f"\n{'='*60}")
            print(f"BUILD RÉUSSI!")
            print(f"Exécutable: {exe_path}")
            print(f"Taille: {size_mb:.1f} MB")
            print(f"{'='*60}\n")
            return True

    print(f"\nERREUR: Le build a échoué")
    return False


def main():
    """Point d'entrée principal"""

    print(f"\n{'='*60}")
    print(f"  {APP_NAME} v{VERSION} - Build Script")
    print(f"{'='*60}\n")

    # Vérifier le script principal
    if not Path(MAIN_SCRIPT).exists():
        print(f"ERREUR: {MAIN_SCRIPT} non trouvé!")
        sys.exit(1)

    # Vérifier PyInstaller
    check_pyinstaller()

    # Parser les arguments
    debug_mode = "--debug" in sys.argv

    # Nettoyer
    if "--no-clean" not in sys.argv:
        clean_build()

    # Build
    success = build_executable(debug=debug_mode)

    if success:
        print("\n" + "="*60)
        print("BUILD TERMINÉ AVEC SUCCÈS")
        print(f"Exécutable dans: {DIST_DIR.absolute()}")
        print("="*60)
    else:
        print("\nLe build a échoué")
        sys.exit(1)


if __name__ == "__main__":
    main()
