"""Tiny pub-sub bus.

Single thread — runs on the Tk mainloop. Subscribers are called
synchronously in registration order. Handler exceptions are logged
(``logging.exception``) so one bad subscriber doesn't break the chain.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

Handler = Callable[..., Any]


class EventBus:
    """In-process event bus."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Handler]] = {}

    def on(self, event: str, handler: Handler) -> Callable[[], None]:
        """Subscribe *handler* to *event*. Returns an unsubscribe callable."""
        self._subscribers.setdefault(event, []).append(handler)

        def _unsubscribe() -> None:
            self.off(event, handler)

        return _unsubscribe

    def off(self, event: str, handler: Handler) -> None:
        """Unsubscribe *handler* from *event*. Silent if not present."""
        try:
            self._subscribers.get(event, []).remove(handler)
        except ValueError:
            pass

    def emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Notify subscribers of *event*. Errors logged, never raised."""
        for handler in list(self._subscribers.get(event, [])):
            try:
                handler(*args, **kwargs)
            except Exception:
                logger.exception("Event handler error for %r", event)

    def clear(self, event: str | None = None) -> None:
        """Drop all subscribers (or just for *event*)."""
        if event is None:
            self._subscribers.clear()
        else:
            self._subscribers.pop(event, None)
