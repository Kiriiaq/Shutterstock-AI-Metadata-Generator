"""CSV exporters for Adobe Stock and Shutterstock contributor portals.

Both writers consume :class:`ExpertMetadataReport` and emit a CSV
matching the portal's import template. UTF-8 with BOM is used because
both portals' Excel-based templates expect it.

Column layouts:

Adobe Stock (5 columns)
    Filename, Title, Keywords, Category, Releases

Shutterstock (7 columns)
    Filename, Description, Keywords, Categories, Editorial, Mature, Illustration

The ``export_double_csv`` helper writes both files side-by-side in a
single call — the common case after a batch analysis.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from ..models.metadata_models import ExpertMetadataReport

logger = logging.getLogger(__name__)


ADOBE_COLUMNS = ["Filename", "Title", "Keywords", "Category", "Releases"]
SHUTTERSTOCK_COLUMNS = [
    "Filename",
    "Description",
    "Keywords",
    "Categories",
    "Editorial",
    "Mature",
    "Illustration",
]


@dataclass
class ExportResult:
    """Return value of an export call."""

    adobe_csv: Optional[Path] = None
    shutterstock_csv: Optional[Path] = None
    row_count: int = 0
    skipped: int = 0


def write_adobe_csv(
    reports: Iterable[ExpertMetadataReport],
    output_path: Path,
) -> int:
    """Write the Adobe Stock contributor CSV. Returns row count."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    # utf-8-sig writes the BOM Excel expects; ``newline=""`` is the
    # csv-module idiom on Windows to avoid blank lines.
    with output_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=ADOBE_COLUMNS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for report in reports:
            row = report.to_adobe_csv_row()
            writer.writerow(_select_columns(row, ADOBE_COLUMNS))
            count += 1
    logger.info("Adobe CSV written: %s (%d rows)", output_path, count)
    return count


def write_shutterstock_csv(
    reports: Iterable[ExpertMetadataReport],
    output_path: Path,
) -> int:
    """Write the Shutterstock contributor CSV. Returns row count."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SHUTTERSTOCK_COLUMNS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for report in reports:
            row = report.to_shutterstock_csv_row()
            writer.writerow(_select_columns(row, SHUTTERSTOCK_COLUMNS))
            count += 1
    logger.info("Shutterstock CSV written: %s (%d rows)", output_path, count)
    return count


def export_double_csv(
    reports: Iterable[ExpertMetadataReport],
    output_dir: Path,
    *,
    basename: str = "metadata",
) -> ExportResult:
    """Write both Adobe and Shutterstock CSVs side by side.

    Files are named ``{basename}_adobe.csv`` and
    ``{basename}_shutterstock.csv`` inside ``output_dir``.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports_list: List[ExpertMetadataReport] = list(reports)

    adobe_path = out_dir / f"{basename}_adobe.csv"
    sh_path = out_dir / f"{basename}_shutterstock.csv"

    adobe_count = write_adobe_csv(reports_list, adobe_path)
    sh_count = write_shutterstock_csv(reports_list, sh_path)

    return ExportResult(
        adobe_csv=adobe_path,
        shutterstock_csv=sh_path,
        row_count=max(adobe_count, sh_count),
        skipped=0,
    )


def _select_columns(row: dict, columns: List[str]) -> dict:
    """Project ``row`` onto ``columns``, filling missing keys with ''."""
    return {col: row.get(col, "") for col in columns}
