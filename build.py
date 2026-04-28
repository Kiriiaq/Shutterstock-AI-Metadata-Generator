#!/usr/bin/env python3
"""Build ShutterstockAnalyzer executables with PyInstaller.

Usage:
    python build.py debug      # Debug profile: console, --debug=imports, --noupx
    python build.py release    # Release profile: --windowed, --noupx, optimized
    python build.py all        # Both profiles
    python build.py clean      # Remove build/, dist/, *.spec
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

APP_NAME = "ShutterstockAnalyzer"
VERSION = "2.0.0"
ICON = "assets/icons/icone.ico"
ENTRY_POINT = "main.py"

# Modules used by the v2 codebase that PyInstaller cannot auto-detect.
# Verified by grep against src/ + main.py:
#   customtkinter — main.py + every page
#   darkdetect    — transitive via customtkinter, but ctk uses string-based imports
#   PIL           — scan_page, write_page, splash_screen, validators
HIDDEN_IMPORTS = [
    "customtkinter",
    "darkdetect",
    "PIL",
]

# Modules NOT used by v2 but pulled in by transitive trees we want to strip
# from the bundle. Keeps the EXE small and reduces cold-start time.
EXCLUDE_MODULES = [
    # Heavy data/science stacks never imported by v2
    "scipy",
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    # Office formats — no Office workflow in v2
    "docx",
    "pptx",
    "openpyxl",
    "xlrd",
    "xlsxwriter",
    "oletools",
    # PDF stacks — no PDF workflow
    "PyPDF2",
    "pdfplumber",
    "fitz",
    "pymupdf",
    "reportlab",
    # Computer vision — not used in v2 (we delegate vision to Ollama via HTTP)
    "cv2",
    "dlib",
    "moviepy",
    "whisper",
    # Win32 COM — not used in v2 (ExifTool is invoked via subprocess)
    "win32com",
    "pythoncom",
    "pywintypes",
    # Deps listed in v1 but never imported by v2 src
    "piexif",
    "pydantic",
    "ollama",
    "CTkToolTip",
]

# Standard-library and tooling submodules safe to drop in any profile.
GLOBAL_EXCLUDES = [
    "unittest",
    "pytest",
    "pydoc",
    "doctest",
    "lib2to3",
    "ensurepip",
    "venv",
    "distutils",
    "setuptools",
    "pkg_resources",
    "pip",
    "tkinter.test",
    "idlelib",
]

PROJECT_DIR = Path(__file__).parent
DIST_DIR = PROJECT_DIR / "dist"
BUILD_DIR = PROJECT_DIR / "build"


def _clean_artifacts(verbose: bool = True) -> None:
    """Remove build artifacts that PyInstaller leaves behind."""
    targets = [BUILD_DIR, *PROJECT_DIR.glob("*.spec")]
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
            if verbose:
                print(f"  removed dir   {target.name}")
        elif target.exists():
            target.unlink()
            if verbose:
                print(f"  removed file  {target.name}")


def _common_args(output_name: str) -> list[str]:
    """Args shared by debug and release profiles."""
    args = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",
        "--name",
        output_name,
        "--distpath",
        str(DIST_DIR),
        "--workpath",
        str(BUILD_DIR),
        "--specpath",
        str(PROJECT_DIR),
        "--noconfirm",
        "--noupx",  # UPX triggers AV false positives on Windows
    ]

    icon_path = PROJECT_DIR / ICON
    if icon_path.exists():
        args.extend(["--icon", str(icon_path)])

    for hi in HIDDEN_IMPORTS:
        args.extend(["--hidden-import", hi])

    for mod in EXCLUDE_MODULES + GLOBAL_EXCLUDES:
        args.extend(["--exclude-module", mod])

    # Bundle assets and src/ alongside the EXE (resolved at runtime via
    # main.resource_path which honours sys._MEIPASS).
    for data_dir in ["assets", "src"]:
        src_path = PROJECT_DIR / data_dir
        if src_path.exists():
            args.extend(["--add-data", f"{src_path}{os.pathsep}{data_dir}"])

    return args


def _build(profile: str) -> Path:
    """Run PyInstaller for the given profile and return the EXE path."""
    if profile == "debug":
        output_name = f"{APP_NAME}-debug"
        cmd = _common_args(output_name) + ["--console", "--debug=imports"]
    elif profile == "release":
        output_name = APP_NAME
        cmd = _common_args(output_name) + ["--windowed", "--noconsole"]
    else:
        raise ValueError(f"Unknown profile: {profile!r}")

    cmd.append(str(PROJECT_DIR / ENTRY_POINT))

    DIST_DIR.mkdir(exist_ok=True)
    print(f"\n=== Building {output_name}.exe ({profile}) ===")
    print(f"Command: {' '.join(cmd[:8])} ... ({len(cmd)} args total)")

    result = subprocess.run(cmd, cwd=str(PROJECT_DIR))
    _clean_artifacts(verbose=False)

    exe = DIST_DIR / f"{output_name}.exe"
    if result.returncode != 0 or not exe.exists():
        print(f"BUILD FAILED ({profile})")
        sys.exit(1)

    size_mb = exe.stat().st_size / (1024 * 1024)
    print(f"OK: {exe.name} ({size_mb:.1f} MB)")
    return exe


def _smoke(exe: Path, profile: str, timeout: float = 8.0) -> None:
    """Run the EXE briefly to ensure it starts without crashing."""
    print(f"\n=== Smoke test: {exe.name} (timeout {timeout}s) ===")
    try:
        proc = subprocess.Popen(
            [str(exe)],
            cwd=str(exe.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        try:
            stdout, _ = proc.communicate(timeout=timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, _ = proc.communicate()
            rc = 0  # Timeout means it stayed alive — that's the success path

        # First few lines for context (debug profile prints to console)
        head = stdout.decode("utf-8", errors="replace").splitlines()[:10] if stdout else []
        if head:
            print("First lines of output:")
            for line in head:
                print(f"  {line}")
        print(f"  exit code: {rc}, profile: {profile}")
    except Exception as e:
        print(f"  smoke test could not run: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="build.py", description=f"Build {APP_NAME} v{VERSION}")
    parser.add_argument(
        "profile",
        choices=["debug", "release", "all", "clean"],
        help="What to build (or `clean` to remove build artifacts).",
    )
    parser.add_argument(
        "--no-smoke",
        action="store_true",
        help="Skip the post-build EXE smoke test.",
    )
    args = parser.parse_args()

    if args.profile == "clean":
        print("Cleaning build artifacts...")
        _clean_artifacts()
        if DIST_DIR.exists():
            shutil.rmtree(DIST_DIR, ignore_errors=True)
            print(f"  removed dir   {DIST_DIR.name}")
        return

    profiles = ["debug", "release"] if args.profile == "all" else [args.profile]
    built: list[tuple[Path, str]] = []
    for profile in profiles:
        exe = _build(profile)
        built.append((exe, profile))

    if not args.no_smoke:
        for exe, profile in built:
            _smoke(exe, profile)

    print("\n=== Build summary ===")
    for exe, profile in built:
        size_mb = exe.stat().st_size / (1024 * 1024)
        print(f"  {profile:8} -> {exe} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
