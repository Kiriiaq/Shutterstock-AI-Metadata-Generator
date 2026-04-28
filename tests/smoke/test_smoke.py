"""Smoke baseline — captures behavior that must keep working through the audit.

If any of these fails after a refactor, that's a regression. They cover the
surface area that is actually wired up today (ShutterstockAIv2 + DB + workers
+ engines + utils). UI is exercised by tests/ui/ with headless instantiation.
"""


def test_database_crud(tmp_path):
    from src.modules.storage.database import ActionType, Database

    db = Database(tmp_path / "smoke.db")
    db.set_setting("audit_key", "audit_value")
    assert db.get_setting("audit_key") == "audit_value"

    log_id = db.log_action(
        ActionType.METADATA_READ,
        file_path="/tmp/x.jpg",
        success=True,
        duration_ms=42,
    )
    assert log_id > 0

    logs = db.get_audit_logs(limit=10)
    assert len(logs) == 1
    assert logs[0].file_path == "/tmp/x.jpg"
    assert logs[0].action_type is ActionType.METADATA_READ
    db.close()


def test_database_batch_lifecycle(tmp_path):
    from src.modules.storage.database import Database

    db = Database(tmp_path / "batch.db")
    db.create_batch(batch_id="b1", source_folder="/tmp", total_files=3)
    db.update_batch_progress("b1", processed=2, failed=1)
    db.complete_batch("b1", status="completed")
    stats = db.get_statistics()
    assert stats["total_batches"] == 1
    db.close()


def test_database_set_file_flags(tmp_path):
    """set_file_flags must update boolean flags without requiring hash/size."""
    from src.modules.storage.database import Database

    db = Database(tmp_path / "flags.db")
    db.set_file_flags("/tmp/img1.jpg", has_ai_analysis=True)
    row = db.get_file_status("/tmp/img1.jpg")
    assert row is not None
    assert row["has_ai_analysis"] == 1
    assert row["has_metadata"] == 0

    db.set_file_flags("/tmp/img1.jpg", has_metadata=True)
    row = db.get_file_status("/tmp/img1.jpg")
    assert row["has_metadata"] == 1
    assert row["has_ai_analysis"] == 1

    # No-op when both kwargs are None
    db.set_file_flags("/tmp/img1.jpg")
    db.close()


def test_iptc_engine_templates():
    from src.modules.engines.iptc_engine import IPTCEngine

    engine = IPTCEngine()
    templates = engine.list_templates()
    assert isinstance(templates, list)


def test_iptc_fields_roundtrip_scalars():
    """Scalar IPTCFields fields roundtrip via to_dict/from_dict."""
    from src.modules.models.metadata_models import IPTCFields

    fields = IPTCFields(
        headline="Mountain at sunset",
        caption="A scenic view of mountains.",
        copyright_notice="(c) 2026",
        country_code="FRA",
    )
    data = fields.to_dict()
    restored = IPTCFields.from_dict(data)
    assert restored.headline == "Mountain at sunset"
    assert restored.caption == "A scenic view of mountains."
    assert restored.copyright_notice == "(c) 2026"
    assert restored.country_code == "FRA"


def test_iptc_fields_roundtrip_lists():
    """List fields (keywords, supplemental_categories) roundtrip via to/from_dict.

    Regression guard for B-16: previously dropped because the from_dict
    filter used hasattr(cls, key) which returns False for fields declared
    with field(default_factory=...).
    """
    from src.modules.models.metadata_models import IPTCFields

    src_fields = IPTCFields(
        headline="x",
        keywords=["mountain", "sunset"],
        supplemental_categories=["Nature", "Landscape"],
    )
    restored = IPTCFields.from_dict(src_fields.to_dict())
    assert restored.keywords == ["mountain", "sunset"]
    assert restored.supplemental_categories == ["Nature", "Landscape"]


def test_collect_image_files(tmp_path):
    from src.modules.workers.worker_pool import collect_image_files

    (tmp_path / "a.jpg").touch()
    (tmp_path / "b.PNG").touch()
    (tmp_path / "ignore.txt").touch()
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.tiff").touch()

    flat = collect_image_files(tmp_path, recursive=False)
    assert {p.name.lower() for p in flat} == {"a.jpg", "b.png"}

    deep = collect_image_files(tmp_path, recursive=True)
    assert {p.name.lower() for p in deep} == {"a.jpg", "b.png", "c.tiff"}


def test_clean_keywords_basic():
    from src.modules.workers.worker_pool import clean_keywords_advanced

    raw = ["mountain", "the", "Mountain", "x", "  trees  ", "photo"]
    cleaned = clean_keywords_advanced(raw)
    assert "the" not in cleaned
    assert "photo" not in cleaned
    assert cleaned.count("mountain") == 1
    assert "trees" in cleaned
    assert all(len(k) >= 2 for k in cleaned)


def test_worker_pool_start_stop():
    from src.modules.workers.worker_pool import WorkerPool

    pool = WorkerPool(max_workers=2)
    pool.start()
    assert pool._running is True
    pool.stop(wait=True)
    assert pool._running is False


def test_worker_pool_executes_handler(tmp_path):
    from src.modules.workers.worker_pool import Task, TaskPriority, WorkerPool

    pool = WorkerPool(max_workers=2)
    pool.register_handler("noop", lambda path, params: ("ok", str(path)))
    pool.start()
    target = tmp_path / "x.jpg"
    target.touch()
    pool.submit_task(Task(task_id="t1", task_type="noop", file_path=target, priority=TaskPriority.NORMAL))
    result = pool.process_queue()
    assert result.completed_tasks == 1
    assert result.failed_tasks == 0
    pool.stop()


def test_ollama_status_enum():
    from src.modules.ai.ollama_client import OllamaStatus

    assert OllamaStatus.ONLINE.value == "online"
    assert OllamaStatus.OFFLINE.value == "offline"
    assert OllamaStatus.BUSY.value == "busy"


def test_validators_dimensions():
    from src.utils.validators import validate_image_dimensions

    ok, err = validate_image_dimensions(3000, 2000, min_megapixels=4.0)
    assert ok is True
    assert err is None

    ok, err = validate_image_dimensions(500, 500, min_megapixels=4.0)
    assert ok is False
    assert err and "low" in err.lower()


def test_shutterstock_params_serialization():
    from src.core.params import ShutterstockParams

    params = ShutterstockParams(source_folder="/tmp/photos")
    data = params.to_dict()
    assert data["source_folder"] == "/tmp/photos"
    restored = ShutterstockParams.from_dict(data)
    assert restored.source_folder == "/tmp/photos"


def test_src_package_imports():
    """Catch missing modules / circular imports across the active surface."""
    import importlib

    for mod in [
        "src.core.params",
        "src.core.config_manager",
        "src.modules.integration",
        "src.modules.storage.database",
        "src.modules.workers.worker_pool",
        "src.modules.engines.iptc_engine",
        "src.modules.engines.metadata_reader",
        "src.modules.engines.metadata_writer",
        "src.modules.ai.ollama_client",
        "src.modules.ai.vision_analyzer",
        "src.modules.ai.prompt_templates",
        "src.modules.models.metadata_models",
        "src.utils.validators",
        "src.utils.file_utils",
        "src.utils.splash_screen",
    ]:
        importlib.import_module(mod)


def test_shutterstock_ai_v2_instantiates(tmp_path, monkeypatch):
    """Smoke: top-level facade builds even without ExifTool installed."""
    from src.modules import integration as integ

    # Force ExifTool-not-found path so we cover the graceful-degradation branch.
    monkeypatch.setattr(integ, "MetadataReader", _RaiseExifToolMissing)
    api = integ.ShutterstockAIv2(db_path=tmp_path / "facade.db")
    assert api.exiftool_available is False
    assert api.metadata_reader is None
    assert api.metadata_writer is None
    api.close()


class _RaiseExifToolMissing:
    """Test helper: simulates ExifTool absence."""

    def __init__(self, *args, **kwargs):
        from src.modules.engines.metadata_reader import ExifToolNotFoundError

        raise ExifToolNotFoundError("forced for smoke test")
