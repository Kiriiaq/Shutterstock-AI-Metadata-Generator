"""Observable application state.

Wraps a dict and emits ``state.{key}.changed`` on the EventBus whenever a
value mutates. Vues subscribe via ``state.observe(key, callback)`` and
receive the new value as the only argument.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.core.events import EventBus


class AppState:
    """Central observable state. Single instance owned by ``App``."""

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._values: dict[str, Any] = {
            "source_folder": None,
            "scanned_images": [],
            "selected_paths": [],
            "current_batch_id": None,
            "ai_status": "unknown",
            "exiftool_available": False,
        }

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Update *key*. No-op if value is unchanged."""
        if self._values.get(key) == value:
            return
        self._values[key] = value
        self._bus.emit(f"state.{key}.changed", value)

    def observe(self, key: str, callback: Callable[[Any], None]) -> Callable[[], None]:
        """Subscribe to changes of *key*. Returns unsubscribe callable."""
        return self._bus.on(f"state.{key}.changed", callback)

    def snapshot(self) -> dict[str, Any]:
        """Shallow copy of the full state — for debugging only."""
        return dict(self._values)
