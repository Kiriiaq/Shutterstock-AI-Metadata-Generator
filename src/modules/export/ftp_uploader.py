"""FTP / FTPS uploader for stock platform contributor portals.

Adobe Stock and Shutterstock both accept contributor uploads via FTP.
This module wraps the stdlib :mod:`ftplib` to give the rest of the
app a simple ``upload_files()`` call without leaking ``ftplib``
quirks (passive mode, encoding, TLS variants…).

Design notes
------------
- **No external dependency**: ftplib is stdlib. Keeps the EXE small.
- **TLS-first by default** (FTPS explicit / FTP_TLS). Plain FTP is
  still available via ``use_tls=False`` for legacy endpoints.
- **Streaming**: files are sent via :meth:`storbinary` with a 32 KB
  block size, so a 50 MB JPEG never sits in RAM.
- **Lax error model**: we collect failures into the
  :class:`UploadResult` instead of raising on the first failure —
  matches the rest of the project where the reviewer is the final
  gate, not the local pipeline.
"""

from __future__ import annotations

import ftplib
import logging
import socket
import ssl
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional

logger = logging.getLogger(__name__)


# Block size for storbinary. 32 KB is a good trade-off between syscall
# overhead and progress granularity (you get a callback ~30× per MB).
DEFAULT_BLOCK_SIZE = 32 * 1024


# ``ftplib.all_errors`` is itself a tuple; an ``except`` clause cannot
# nest tuples, so we flatten to one tuple at module load time.
_FTP_ERRORS: tuple = tuple(set(ftplib.all_errors) | {OSError, ssl.SSLError, socket.error})


@dataclass
class FtpConfig:
    """Connection settings — credentials are NEVER logged.

    ``host`` may include a port (``ftp.example.com:21``), or pass
    ``port`` explicitly. Default port is 21 for FTP/FTPS-explicit and
    990 for FTPS-implicit (rarely needed for stock platforms).
    """

    host: str
    user: str
    password: str
    remote_dir: str = "/"
    port: int = 21
    use_tls: bool = True  # FTPS explicit (default for Shutterstock)
    passive: bool = True
    timeout: float = 30.0
    # Verify server certificate. Set False for self-signed test
    # servers. Adobe/Shutterstock use valid certs so the default holds.
    verify_cert: bool = True

    def safe_summary(self) -> str:
        """Human-readable summary safe to log (no credentials)."""
        proto = "FTPS" if self.use_tls else "FTP"
        return f"{proto}://{self.user}@{self.host}:{self.port}{self.remote_dir}"


@dataclass
class UploadItem:
    """One file's upload outcome."""

    local_path: Path
    remote_name: str
    success: bool = False
    bytes_sent: int = 0
    duration_s: float = 0.0
    error: Optional[str] = None


@dataclass
class UploadResult:
    """Outcome of an :func:`upload_files` call."""

    config_summary: str
    items: List[UploadItem] = field(default_factory=list)
    total_bytes: int = 0
    duration_s: float = 0.0
    error: Optional[str] = None  # connection-level error, not per-file

    @property
    def success_count(self) -> int:
        return sum(1 for i in self.items if i.success)

    @property
    def failure_count(self) -> int:
        return sum(1 for i in self.items if not i.success)

    @property
    def is_complete_success(self) -> bool:
        return self.error is None and self.failure_count == 0 and bool(self.items)


# ============================================================================
# Public API
# ============================================================================


def upload_files(
    files: Iterable[Path],
    config: FtpConfig,
    *,
    progress: Optional[Callable[[Path, int, int], None]] = None,
) -> UploadResult:
    """Upload *files* to an FTP/FTPS server.

    Args:
        files: Iterable of local paths.
        config: Connection + destination settings.
        progress: Callback ``(path, bytes_sent, total_bytes)`` fired
            roughly every 32 KB. Use to drive a UI progress bar. Can
            be ``None``.

    Returns:
        :class:`UploadResult` with per-file outcomes. Never raises
        on individual file errors; only catastrophic connection
        failures populate ``result.error``.
    """
    paths = [Path(f) for f in files]
    result = UploadResult(config_summary=config.safe_summary())
    if not paths:
        logger.info("upload_files called with empty file list — no-op")
        return result

    logger.info("FTP upload to %s — %d file(s)", result.config_summary, len(paths))
    t0 = time.perf_counter()

    try:
        ftp = _connect(config)
    except _FTP_ERRORS as exc:
        result.error = f"Connexion FTP impossible : {exc}"
        result.duration_s = time.perf_counter() - t0
        logger.exception("FTP connect failed")
        return result

    try:
        _ensure_remote_dir(ftp, config.remote_dir)
        for path in paths:
            result.items.append(_upload_one(ftp, path, progress=progress))
    finally:
        try:
            ftp.quit()
        except _FTP_ERRORS:
            try:
                ftp.close()
            except OSError:
                pass

    result.total_bytes = sum(it.bytes_sent for it in result.items)
    result.duration_s = time.perf_counter() - t0
    return result


def test_connection(config: FtpConfig) -> tuple[bool, str]:
    """Quick connectivity check — used by the UI's « Tester » button.

    Returns ``(success, message)``. No file is uploaded; we just open
    the control connection, log in, list the remote dir, then quit.
    """
    summary = config.safe_summary()
    try:
        ftp = _connect(config)
    except Exception as exc:  # noqa: BLE001
        return False, f"Échec connexion {summary} : {exc}"
    try:
        _ensure_remote_dir(ftp, config.remote_dir, create_if_missing=False)
        # Listing is the canonical "I'm in" check.
        ftp.nlst()
        return True, f"Connecté : {summary}"
    except Exception as exc:  # noqa: BLE001
        return False, f"Connecté à {config.host} mais accès dossier KO : {exc}"
    finally:
        try:
            ftp.quit()
        except _FTP_ERRORS:
            try:
                ftp.close()
            except OSError:
                pass


# Pytest collects every top-level ``test_*`` callable as a test case.
# ``test_connection`` is part of the public API — flag it so the
# collector skips it. Must come AFTER the function definition.
test_connection.__test__ = False  # type: ignore[attr-defined]


# ============================================================================
# Internals
# ============================================================================


def _connect(config: FtpConfig) -> ftplib.FTP:
    """Open a control connection, log in, set passive + utf-8."""
    host = config.host
    port = config.port
    if ":" in host and host.count(":") == 1:
        h, p = host.rsplit(":", 1)
        if p.isdigit():
            host, port = h, int(p)

    if config.use_tls:
        ctx = ssl.create_default_context()
        if not config.verify_cert:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        ftp: ftplib.FTP = ftplib.FTP_TLS(timeout=config.timeout, context=ctx)
    else:
        ftp = ftplib.FTP(timeout=config.timeout)

    ftp.connect(host=host, port=port)
    ftp.login(user=config.user, passwd=config.password)

    if isinstance(ftp, ftplib.FTP_TLS):
        # Switch the data channel to TLS too. Without prot_p, the
        # control channel is encrypted but file bytes go in clear,
        # which Shutterstock's portal rejects with 550.
        try:
            ftp.prot_p()
        except ftplib.all_errors as exc:
            logger.warning("FTPS PROT P failed (server may not require it): %s", exc)

    ftp.set_pasv(config.passive)
    try:
        ftp.sendcmd("OPTS UTF8 ON")  # safe to ignore failures on older servers
    except ftplib.all_errors:
        pass
    return ftp


def _ensure_remote_dir(ftp: ftplib.FTP, remote_dir: str, *, create_if_missing: bool = True) -> None:
    """Navigate to *remote_dir*, creating it if needed.

    Empty / "/" / "." → noop. Otherwise we attempt a single ``cwd``,
    falling back to ``mkd`` then ``cwd`` on 550. Multi-level paths
    are walked component by component for portability.
    """
    if not remote_dir or remote_dir in ("/", "."):
        return

    target = remote_dir.replace("\\", "/")
    # Absolute → cwd to root first.
    if target.startswith("/"):
        try:
            ftp.cwd("/")
        except ftplib.all_errors:
            pass
        target = target.lstrip("/")

    for component in (c for c in target.split("/") if c):
        try:
            ftp.cwd(component)
        except ftplib.all_errors:
            if not create_if_missing:
                raise
            ftp.mkd(component)
            ftp.cwd(component)


def _upload_one(
    ftp: ftplib.FTP,
    path: Path,
    *,
    progress: Optional[Callable[[Path, int, int], None]],
) -> UploadItem:
    """Send a single file via STOR. Per-file error → UploadItem.error."""
    item = UploadItem(local_path=path, remote_name=path.name)
    if not path.exists():
        item.error = f"Fichier introuvable : {path}"
        return item

    try:
        total = path.stat().st_size
    except OSError as exc:
        item.error = f"stat() impossible : {exc}"
        return item

    t0 = time.perf_counter()
    sent = 0

    def _on_block(_chunk: bytes) -> None:
        nonlocal sent
        sent += len(_chunk)
        if progress:
            try:
                progress(path, sent, total)
            except Exception:  # noqa: BLE001
                logger.exception("FTP progress callback raised")

    try:
        with path.open("rb") as fh:
            ftp.storbinary(f"STOR {path.name}", fh, blocksize=DEFAULT_BLOCK_SIZE,
                           callback=_on_block)
        item.success = True
        item.bytes_sent = sent
    except _FTP_ERRORS as exc:
        item.error = str(exc)
        item.bytes_sent = sent
        logger.warning("STOR failed for %s: %s", path.name, exc)

    item.duration_s = time.perf_counter() - t0
    return item
