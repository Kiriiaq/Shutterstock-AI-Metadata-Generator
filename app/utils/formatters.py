"""French-locale formatters for numbers, dates, durations, file sizes."""

from __future__ import annotations

from datetime import datetime

NBSP = " "  # non-breaking space — French thousand separator


def fmt_int(n: int) -> str:
    """``1234567`` → ``"1 234 567"`` with non-breaking spaces."""
    return f"{n:,}".replace(",", NBSP)


def fmt_float(x: float, decimals: int = 2) -> str:
    """French float: ``1234.5`` → ``"1 234,50"`` with NBSP and comma."""
    return f"{x:,.{decimals}f}".replace(",", "\x00").replace(".", ",").replace("\x00", NBSP)


def fmt_date(dt: datetime) -> str:
    """``JJ/MM/AAAA``."""
    return dt.strftime("%d/%m/%Y")


def fmt_datetime(dt: datetime) -> str:
    """``JJ/MM/AAAA HH:MM``."""
    return dt.strftime("%d/%m/%Y %H:%M")


def fmt_size(bytes_count: int) -> str:
    """``1234`` → ``"1,2 Ko"``, ``1500000`` → ``"1,4 Mo"``."""
    units = ("o", "Ko", "Mo", "Go", "To")
    size = float(bytes_count)
    for unit in units:
        if abs(size) < 1024.0:
            if unit == "o":
                return f"{int(size)}{NBSP}{unit}"
            return fmt_float(size, decimals=1) + NBSP + unit
        size /= 1024.0
    return f"{fmt_float(size, decimals=1)}{NBSP}Po"


def fmt_duration_ms(ms: int) -> str:
    """``42`` → ``"42 ms"``, ``1234`` → ``"1,2 s"``, ``75000`` → ``"1 min 15 s"``."""
    if ms < 1000:
        return f"{ms}{NBSP}ms"
    seconds = ms / 1000.0
    if seconds < 60:
        return f"{fmt_float(seconds, 1)}{NBSP}s"
    minutes = int(seconds // 60)
    rem = int(seconds % 60)
    return f"{minutes}{NBSP}min{NBSP}{rem:02d}{NBSP}s"
