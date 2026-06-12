"""Tests for the Ollama-related methods on the facade.

The OllamaClient itself talks HTTP, so we stub it at the integration
boundary — these tests verify the facade's defensive wrapping
(auto-init, lazy probe, error swallowing).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.modules.ai.ollama_client import ModelInfo
from src.modules.integration import ShutterstockAIv2


@pytest.fixture
def facade(tmp_path):
    """Bare ShutterstockAIv2 with an in-memory DB. ExifTool may be
    absent — irrelevant for these tests since we stub ollama_client."""
    db_path = tmp_path / "test.db"
    instance = ShutterstockAIv2(db_path=db_path)
    yield instance
    instance.close()


def _fake_vision_models(names):
    """Build ModelInfo stubs that look like vision models."""
    out = []
    for n in names:
        m = ModelInfo(name=n, size=0)
        out.append(m)
    return out


class TestListVisionModels:
    def test_returns_empty_when_no_ollama(self, facade):
        # No ollama_client attribute → list returns [] without raising
        with patch.object(facade, "init_ai", side_effect=Exception("offline")):
            result = facade.list_vision_models()
        assert result == []

    def test_returns_names_from_client(self, facade):
        fake_client = MagicMock()
        fake_client.list_vision_models.return_value = _fake_vision_models(
            ["llama3.2-vision:11b", "llava:7b"]
        )
        facade.ollama_client = fake_client
        result = facade.list_vision_models()
        assert result == ["llama3.2-vision:11b", "llava:7b"]

    def test_refresh_forwards_to_client(self, facade):
        fake_client = MagicMock()
        fake_client.list_vision_models.return_value = []
        facade.ollama_client = fake_client
        facade.list_vision_models(refresh=True)
        fake_client.list_models.assert_called_once_with(refresh=True)


class TestPreloadModel:
    def test_rejects_empty_name(self, facade):
        ok, msg = facade.preload_model("")
        assert ok is False
        assert "modèle" in msg.lower()

    def test_loads_and_persists(self, facade):
        fake_client = MagicMock()
        fake_client.load_model.return_value = True
        fake_analyzer = MagicMock()
        facade.ollama_client = fake_client
        facade.vision_analyzer = fake_analyzer

        ok, msg = facade.preload_model("llama3.2-vision:11b")
        assert ok is True
        assert "llama3.2-vision:11b" in msg
        fake_client.load_model.assert_called_once_with("llama3.2-vision:11b")
        # vision_analyzer.model should be updated
        assert fake_analyzer.model == "llama3.2-vision:11b"
        # Setting persisted
        assert facade.get_setting("ollama_model") == "llama3.2-vision:11b"

    def test_load_failure_returns_false(self, facade):
        fake_client = MagicMock()
        fake_client.load_model.return_value = False
        facade.ollama_client = fake_client
        ok, msg = facade.preload_model("nope:0")
        assert ok is False
        assert "Échec" in msg

    def test_swallows_exceptions(self, facade):
        fake_client = MagicMock()
        fake_client.load_model.side_effect = RuntimeError("server gone")
        facade.ollama_client = fake_client
        ok, msg = facade.preload_model("x")
        assert ok is False
        assert "server gone" in msg


class TestGetCurrentModel:
    def test_returns_none_when_not_initialized(self, facade):
        assert facade.get_current_model() is None

    def test_returns_client_current_model(self, facade):
        fake_client = MagicMock()
        fake_client.current_model = "llava:7b"
        facade.ollama_client = fake_client
        assert facade.get_current_model() == "llava:7b"


def _fake_metadata(*, headline, keywords):
    """ImageMetadata stub matching the _has_complete_iptc criterion."""
    meta = MagicMock()
    meta.has_iptc = bool(headline or keywords)
    meta.iptc.headline = headline
    meta.iptc.keywords = keywords
    return meta


def _install_fake_analyzer(facade, analyzed_paths):
    """Stub vision_analyzer whose analyze_batch records the paths it gets
    and emits one completed result per path (mirrors the real callback
    contract used by analyze_batch_ai)."""
    fake_analyzer = MagicMock()

    def fake_batch(paths, **kwargs):
        analyzed_paths.extend(paths)
        for p in paths:
            res = MagicMock()
            res.status.value = "completed"
            res.is_successful = True
            res.to_dict.return_value = {"file_path": str(p), "status": "completed"}
            kwargs["on_result"](res)
        return []

    fake_analyzer.analyze_batch.side_effect = fake_batch
    facade.vision_analyzer = fake_analyzer
    facade.ollama_client = MagicMock()
    return fake_analyzer


class TestAnalyzeBatchSkipPreFilter:
    """Audit B-01 — « Ignorer si méta » must actually skip in batch mode.

    VisionAnalyzer.analyze_image only honours skip_if_has_metadata when
    existing_metadata is supplied, and the batch path never supplies it,
    so the facade must pre-filter. These tests pin that contract.
    """

    def test_files_with_complete_iptc_are_skipped(self, facade, tmp_path):
        file_a = tmp_path / "a.jpg"
        file_b = tmp_path / "b.jpg"
        file_a.write_bytes(b"x")
        file_b.write_bytes(b"x")

        facade.metadata_reader = MagicMock()  # enables the pre-filter
        full = _fake_metadata(headline="Sunset", keywords=["sky"])
        facade.read_metadata = MagicMock(
            side_effect=lambda p: full if p.name == "a.jpg" else None
        )

        analyzed: list = []
        _install_fake_analyzer(facade, analyzed)

        seen_results = []
        summary = facade.analyze_batch_ai(
            [file_a, file_b],
            skip_if_has_metadata=True,
            on_result=seen_results.append,
        )

        assert summary["skipped"] == 1
        assert summary["completed"] == 1
        assert [p.name for p in analyzed] == ["b.jpg"]
        skipped = [r for r in seen_results if r["status"] == "skipped"]
        assert len(skipped) == 1 and skipped[0]["file_path"].endswith("a.jpg")

    def test_flag_off_analyzes_everything(self, facade, tmp_path):
        file_a = tmp_path / "a.jpg"
        file_a.write_bytes(b"x")

        facade.metadata_reader = MagicMock()
        facade.read_metadata = MagicMock(
            return_value=_fake_metadata(headline="T", keywords=["k"])
        )

        analyzed: list = []
        _install_fake_analyzer(facade, analyzed)

        summary = facade.analyze_batch_ai([file_a], skip_if_has_metadata=False)

        assert summary["skipped"] == 0
        assert summary["completed"] == 1
        assert [p.name for p in analyzed] == ["a.jpg"]
        facade.read_metadata.assert_not_called()

    def test_partial_iptc_is_not_skipped(self, facade, tmp_path):
        # Keywords without headline → incomplete → must go through AI.
        file_a = tmp_path / "a.jpg"
        file_a.write_bytes(b"x")

        facade.metadata_reader = MagicMock()
        facade.read_metadata = MagicMock(
            return_value=_fake_metadata(headline=None, keywords=["k"])
        )

        analyzed: list = []
        _install_fake_analyzer(facade, analyzed)

        summary = facade.analyze_batch_ai([file_a], skip_if_has_metadata=True)

        assert summary["skipped"] == 0
        assert summary["completed"] == 1
        assert [p.name for p in analyzed] == ["a.jpg"]
