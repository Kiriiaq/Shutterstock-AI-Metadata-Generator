<#
.SYNOPSIS
    Release pipeline for ShutterstockAnalyzer.

.DESCRIPTION
    Runs all pre-release checks, builds the EXEs, tags the commit, and
    (optionally) pushes to GitHub + creates the GitHub Release.

    Steps :
      1. Ensure repo is clean (no uncommitted changes).
      2. Run pytest (must be green).
      3. Run ruff check (must be clean).
      4. Run E2E pipeline + diff vs reference.
      5. Build PyInstaller debug + release.
      6. Smoke test the release EXE.
      7. Create annotated git tag.
      8. (optional) Push tag and create GitHub Release with EXE assets.

.PARAMETER Version
    Version to release, e.g. "v2.1.0". Required.

.PARAMETER Push
    Push the tag to origin after creating it.

.PARAMETER Release
    Create a GitHub Release using the gh CLI. Implies -Push.

.EXAMPLE
    # Dry run — local only, no push
    .\tools\release.ps1 -Version v2.1.0

    # Full release
    .\tools\release.ps1 -Version v2.1.0 -Release
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Version,

    [switch]$Push,

    [switch]$Release
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Step($title) {
    Write-Host "`n=== $title ===" -ForegroundColor Cyan
}

function Die($msg) {
    Write-Host "FAILED: $msg" -ForegroundColor Red
    exit 1
}

# 1. Repo clean
Step "1/8  Vérification repo propre"
$dirty = git status --porcelain
if ($dirty) {
    Write-Host $dirty
    Die "Le repo a des modifications non commitées. Commit/stash avant release."
}
Write-Host "  ✓ Repo clean"

# 2. Pytest
Step "2/8  Suite de tests pytest"
python -m pytest tests/ -q --tb=short
if ($LASTEXITCODE -ne 0) { Die "Tests en échec." }
Write-Host "  ✓ Tests verts"

# 3. Ruff
Step "3/8  Lint ruff"
python -m ruff check app/ src/ main.py build.py tests/
if ($LASTEXITCODE -ne 0) { Die "Ruff a remonté des erreurs." }
Write-Host "  ✓ Ruff propre"

# 4. E2E pipeline
Step "4/8  Pipeline E2E + comparaison vs référence"
python test/scripts/run_tests.py
if ($LASTEXITCODE -ne 0) { Die "run_tests.py a échoué." }
python test/scripts/compare_outputs.py
if ($LASTEXITCODE -ne 0) { Die "Régression détectée vs outputs_reference/." }
Write-Host "  ✓ Cell-for-cell match"

# 5. Build
Step "5/8  Build PyInstaller (debug + release)"
python build.py clean
python build.py all
if ($LASTEXITCODE -ne 0) { Die "Build PyInstaller a échoué." }
$relExe = "dist\ShutterstockAnalyzer.exe"
$dbgExe = "dist\ShutterstockAnalyzer-debug.exe"
if (-not (Test-Path $relExe)) { Die "Release EXE introuvable." }
if (-not (Test-Path $dbgExe)) { Die "Debug EXE introuvable." }
$relSize = [math]::Round((Get-Item $relExe).Length / 1MB, 1)
$dbgSize = [math]::Round((Get-Item $dbgExe).Length / 1MB, 1)
Write-Host "  ✓ Release : $relExe ($relSize Mo)"
Write-Host "  ✓ Debug   : $dbgExe ($dbgSize Mo)"

# 6. Smoke (déjà fait par build.py mais on revérifie taille minimum)
Step "6/8  Smoke test taille EXE"
if ($relSize -lt 20 -or $relSize -gt 50) {
    Die "Taille release EXE hors fourchette attendue 20-50 Mo (=$relSize Mo)"
}
Write-Host "  ✓ Taille OK"

# 7. Tag
Step "7/8  Création du tag $Version"
$existing = git tag --list $Version
if ($existing) {
    Die "Le tag $Version existe déjà. Supprime-le (git tag -d $Version) ou choisis une autre version."
}
$msg = "Release $Version`n`nSee CHANGELOG.md for details."
git tag -a $Version -m $msg
Write-Host "  ✓ Tag créé localement"

# 8. Push + Release (optionnel)
if ($Push -or $Release) {
    Step "8/8  Push du tag vers origin"
    git push origin $Version
    if ($LASTEXITCODE -ne 0) { Die "Push tag échoué." }
    Write-Host "  ✓ Tag pushé"

    if ($Release) {
        Write-Host "`n=== Création GitHub Release ===" -ForegroundColor Cyan
        $ghCmd = Get-Command gh -ErrorAction SilentlyContinue
        if (-not $ghCmd) {
            Write-Host "ATTENTION: 'gh' CLI absent. Installe depuis https://cli.github.com/" -ForegroundColor Yellow
            Write-Host "Crée la release manuellement sur https://github.com/Kiriiaq/Shutterstock-AI-Metadata-Generator/releases/new"
            exit 0
        }
        gh release create $Version $relExe $dbgExe `
            --title "$Version" `
            --notes-file CHANGELOG.md `
            --verify-tag
        if ($LASTEXITCODE -ne 0) { Die "gh release create a échoué." }
        Write-Host "  ✓ Release GitHub créée : https://github.com/Kiriiaq/Shutterstock-AI-Metadata-Generator/releases/tag/$Version"
    }
} else {
    Step "8/8  Push + Release (sautés — pass -Push ou -Release pour publier)"
    Write-Host "  Pour push le tag :  git push origin $Version"
    Write-Host "  Pour la release   :  gh release create $Version dist\*.exe --notes-file CHANGELOG.md"
}

Write-Host "`n✅ RELEASE PIPELINE COMPLETE — $Version" -ForegroundColor Green
