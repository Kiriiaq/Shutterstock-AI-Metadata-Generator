#!/usr/bin/env python3
"""Build ShutterstockAnalyzer executable with PyInstaller.

Usage:
    python build.py           # Full build (all libs embedded)
    python build.py --light   # Light build (no libs, needs Python + deps installed)
"""

import subprocess
import sys
import os
import shutil
import argparse
from pathlib import Path

APP_NAME = "ShutterstockAnalyzer"
VERSION = "1.0.0"
ICON = "assets/icons/icone.ico"

HIDDEN_IMPORTS = [
    "customtkinter", "darkdetect",
    "CTkToolTip", "PIL", "piexif", "ollama", "pydantic", "requests",
]

EXCLUDE_MODULES = [
    "scipy", "cv2", "dlib", "moviepy", "whisper", "oletools",
    "pandas", "numpy", "openpyxl", "fitz", "pymupdf",
    "docx", "pptx", "PyPDF2", "reportlab", "matplotlib", "seaborn", "win32com",
]

GLOBAL_EXCLUDES = [
    "unittest", "test", "tests", "pytest", "pydoc", "doctest",
    "lib2to3", "ensurepip", "venv", "distutils",
    "setuptools", "pkg_resources", "pip",
    "tkinter.test", "idlelib",
    "matplotlib.tests", "numpy.tests", "pandas.tests", "scipy.tests",
]

# All heavy libs to strip in --light mode
ALL_HEAVY_LIBS = [
    "numpy", "pandas", "scipy", "cv2", "matplotlib", "seaborn",
    "pymupdf", "fitz", "reportlab", "shapely", "dlib",
    "moviepy", "whisper", "PIL", "Pillow",
    "docx", "pptx", "openpyxl", "xlrd", "xlsxwriter",
    "PyPDF2", "oletools", "win32com", "pythoncom", "pywintypes",
    "customtkinter", "darkdetect", "requests", "pydantic",
    "ollama", "rawpy", "imageio", "pillow_heif",
    "tqdm", "exifread", "piexif", "tinydb", "edge_tts",
    "flask", "aiohttp", "openai", "gtts", "praw", "bs4",
    "CTkMessagebox", "CTkToolTip", "easygui",
    "chardet", "unidecode", "send2trash", "yaml",
    "tomli", "tomli_w", "pdfplumber",
]


def build(light=False):
    project_dir = Path(__file__).parent
    suffix = "-light" if light else ""
    output_name = f"{APP_NAME}-{VERSION}{suffix}"
    dist_dir = project_dir / "dist"
    build_dir = project_dir / "build"

    # Clean build artifacts
    if build_dir.exists():
        shutil.rmtree(build_dir)
    for spec in project_dir.glob("*.spec"):
        spec.unlink()
    dist_dir.mkdir(exist_ok=True)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile", "--console",
        "--name", output_name,
        "--distpath", str(dist_dir),
        "--workpath", str(build_dir),
        "--specpath", str(project_dir),
        "--noconfirm",
    ]

    icon_path = project_dir / ICON
    if icon_path.exists():
        cmd.extend(["--icon", str(icon_path)])

    if light:
        # Light: exclude heavy libs but KEEP modules needed by the app
        keep = set(HIDDEN_IMPORTS)
        for mod in ALL_HEAVY_LIBS:
            if mod not in keep:
                cmd.extend(["--exclude-module", mod])
        for hi in HIDDEN_IMPORTS:
            cmd.extend(["--hidden-import", hi])
        for mod in GLOBAL_EXCLUDES:
            cmd.extend(["--exclude-module", mod])
    else:
        # Full: include needed, exclude unneeded
        for hi in HIDDEN_IMPORTS:
            cmd.extend(["--hidden-import", hi])
        for mod in EXCLUDE_MODULES + GLOBAL_EXCLUDES:
            cmd.extend(["--exclude-module", mod])

    # Bundle assets and src
    for data_dir in ["assets", "src"]:
        src_path = project_dir / data_dir
        if src_path.exists():
            cmd.extend(["--add-data", f"{src_path}{os.pathsep}{data_dir}"])

    cmd.append(str(project_dir / "main.py"))

    mode = "light (no libs)" if light else "full"
    print(f"Building {output_name}.exe ({mode})...")
    result = subprocess.run(cmd, cwd=str(project_dir))

    # Cleanup
    if build_dir.exists():
        shutil.rmtree(build_dir)
    for spec in project_dir.glob("*.spec"):
        spec.unlink()

    exe = dist_dir / f"{output_name}.exe"
    if result.returncode == 0 and exe.exists():
        size = exe.stat().st_size / (1024 * 1024)
        print(f"OK: {exe.name} ({size:.1f} MB)")
    else:
        print("BUILD FAILED")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Build {APP_NAME}")
    parser.add_argument("--light", action="store_true",
                        help="Light build without libraries (needs Python + deps on target)")
    args = parser.parse_args()
    build(light=args.light)
