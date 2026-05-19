"""Tests for the FTP uploader. Use the stdlib pyftpdlib? No — we
keep the dep footprint zero and rely on monkeypatching the ftplib
classes. Each test stubs FTP/FTP_TLS so no socket is ever opened.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.modules.export.ftp_uploader import (
    FtpConfig,
    UploadResult,
    test_connection,
    upload_files,
)


@pytest.fixture
def config() -> FtpConfig:
    return FtpConfig(
        host="ftp.example.com",
        user="u",
        password="p",  # noqa: S106 — fixture value, not real secret
        remote_dir="/inbox",
        use_tls=False,  # plain FTP keeps the stub simple
    )


@pytest.fixture
def sample_files(tmp_path) -> list[Path]:
    out = []
    for i in range(3):
        p = tmp_path / f"file_{i}.csv"
        p.write_bytes(f"col1,col2\n{i},foo".encode())
        out.append(p)
    return out


class TestFtpConfig:
    def test_safe_summary_hides_password(self, config):
        s = config.safe_summary()
        assert "p" not in s.split(":")  # password not embedded
        assert "u@" in s
        assert "ftp.example.com" in s


class TestUploadFiles:
    def test_empty_list_returns_empty_result(self, config):
        result = upload_files([], config)
        assert isinstance(result, UploadResult)
        assert result.items == []
        assert result.error is None

    def test_successful_upload_all_files(self, config, sample_files):
        fake = MagicMock()
        with patch("src.modules.export.ftp_uploader.ftplib.FTP",
                   return_value=fake):
            result = upload_files(sample_files, config)

        assert result.error is None
        assert result.success_count == 3
        assert result.failure_count == 0
        assert all(it.success for it in result.items)
        # storbinary called once per file
        assert fake.storbinary.call_count == 3

    def test_partial_failure_continues(self, config, sample_files):
        from src.modules.export.ftp_uploader import ftplib

        fake = MagicMock()

        # 2nd file: storbinary raises, others succeed.
        def storbinary_side_effect(cmd, fh, blocksize=None, callback=None):
            if "file_1" in cmd:
                raise ftplib.error_perm("550 access denied")
            # simulate streaming for the callback path
            if callback:
                while True:
                    chunk = fh.read(blocksize or 32 * 1024)
                    if not chunk:
                        break
                    callback(chunk)

        fake.storbinary.side_effect = storbinary_side_effect
        with patch("src.modules.export.ftp_uploader.ftplib.FTP",
                   return_value=fake):
            result = upload_files(sample_files, config)

        assert result.success_count == 2
        assert result.failure_count == 1
        bad = next(it for it in result.items if not it.success)
        assert "550" in bad.error

    def test_connection_error_populates_result_error(self, config, sample_files):
        with patch("src.modules.export.ftp_uploader.ftplib.FTP",
                   side_effect=OSError("Network unreachable")):
            result = upload_files(sample_files, config)
        assert result.error and "Connexion FTP" in result.error
        assert result.items == []  # nothing was attempted

    def test_remote_dir_navigation_creates_when_missing(self, config, sample_files):
        from src.modules.export.ftp_uploader import ftplib

        fake = MagicMock()
        # cwd raises 550 on first attempt → mkd then cwd succeeds
        cwd_calls = []

        def cwd_side_effect(path):
            cwd_calls.append(path)
            if path == "inbox" and cwd_calls.count("inbox") == 1:
                raise ftplib.error_perm("550 No such dir")

        fake.cwd.side_effect = cwd_side_effect
        with patch("src.modules.export.ftp_uploader.ftplib.FTP",
                   return_value=fake):
            upload_files(sample_files, config)

        # mkd should have been called for the missing dir
        fake.mkd.assert_called_with("inbox")


class TestTestConnection:
    def test_successful_probe(self, config):
        fake = MagicMock()
        fake.nlst.return_value = ["file1", "file2"]
        with patch("src.modules.export.ftp_uploader.ftplib.FTP",
                   return_value=fake):
            ok, msg = test_connection(config)
        assert ok is True
        assert "Connecté" in msg

    def test_login_failure(self, config):
        from src.modules.export.ftp_uploader import ftplib

        fake = MagicMock()
        fake.login.side_effect = ftplib.error_perm("530 login incorrect")
        with patch("src.modules.export.ftp_uploader.ftplib.FTP",
                   return_value=fake):
            ok, msg = test_connection(config)
        assert ok is False
        assert "Échec" in msg or "KO" in msg
