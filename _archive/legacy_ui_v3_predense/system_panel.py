"""SystemPanel — always-visible right column.

Shows live system state and the last few audit entries so the user
never has to click to know what the application is doing. Refreshes on
a timer and on AppState changes.

Sections:
    1. Status rows  (ExifTool, Ollama, Model, Backend, Workers)
    2. Stats row    (Total processed · with metadata · with AI · errors 24h)
    3. Live audit log tail (last N entries)
    4. Quick navigation buttons (Settings, History, AI, Validate, Upload, Help)
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import customtkinter as ctk

from app.config.theme import (
    RADIUS_MD,
    SPACE_MD,
    SPACE_SM,
    SPACE_XS,
    get_color,
    get_font,
)
from app.utils.formatters import fmt_int

if TYPE_CHECKING:
    from app.app import App

logger = logging.getLogger(__name__)

REFRESH_INTERVAL_MS = 5000  # auto-refresh stats + audit tail every 5 s
AUDIT_TAIL_SIZE = 8


class SystemPanel(ctk.CTkFrame):
    """Right-column system state view. One instance per WorkspaceView."""

    def __init__(self, master: ctk.CTkFrame, *, app: "App") -> None:
        super().__init__(
            master,
            fg_color=get_color("bg_elevated"),
            border_color=get_color("border"),
            border_width=1,
            corner_radius=RADIUS_MD,
        )
        self.app = app
        self._status_rows: dict[str, ctk.CTkLabel] = {}
        self._status_dots: dict[str, ctk.CTkLabel] = {}
        self._stat_values: dict[str, ctk.CTkLabel] = {}
        self._audit_lines: list[ctk.CTkLabel] = []
        self._after_id: str | None = None
        self._build()

    # ------------------------------------------------------------------

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self._build_status(self, row=0)
        self._build_stats(self, row=1)
        self._build_audit_tail(self, row=2)
        self._build_quick_actions(self, row=3)

    def _build_status(self, parent: ctk.CTkFrame, row: int) -> None:
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.grid(row=row, column=0, sticky="ew", padx=SPACE_MD, pady=(SPACE_MD, SPACE_SM))
        section.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            section,
            text="SYSTÈME",
            font=get_font("small"),
            text_color=get_color("fg_subtle"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w", pady=(0, SPACE_XS))

        for r, key in enumerate(["exiftool", "ollama", "model", "backend", "workers"], start=1):
            self._add_status_row(section, key, row=r)

    def _add_status_row(self, parent: ctk.CTkFrame, key: str, row: int) -> None:
        row_frame = ctk.CTkFrame(parent, fg_color="transparent")
        row_frame.grid(row=row, column=0, sticky="ew", pady=1)
        row_frame.grid_columnconfigure(2, weight=1)

        labels = {
            "exiftool": "ExifTool",
            "ollama": "Ollama",
            "model": "Modèle",
            "backend": "Backend",
            "workers": "Workers",
        }
        dot = ctk.CTkLabel(
            row_frame,
            text="●",
            font=get_font("body_strong"),
            text_color=get_color("fg_subtle"),
            width=12,
        )
        dot.grid(row=0, column=0, padx=(0, SPACE_XS))
        ctk.CTkLabel(
            row_frame,
            text=labels[key],
            font=get_font("body"),
            text_color=get_color("fg_muted"),
            anchor="w",
            width=80,
        ).grid(row=0, column=1, sticky="w")
        value = ctk.CTkLabel(
            row_frame,
            text="—",
            font=get_font("body_strong"),
            text_color=get_color("fg"),
            anchor="e",
        )
        value.grid(row=0, column=2, sticky="e")

        self._status_dots[key] = dot
        self._status_rows[key] = value

    def _build_stats(self, parent: ctk.CTkFrame, row: int) -> None:
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.grid(row=row, column=0, sticky="ew", padx=SPACE_MD, pady=(SPACE_SM, SPACE_SM))
        ctk.CTkLabel(
            section,
            text="STATISTIQUES",
            font=get_font("small"),
            text_color=get_color("fg_subtle"),
            anchor="w",
        ).pack(fill="x", pady=(0, SPACE_XS))

        grid = ctk.CTkFrame(section, fg_color="transparent")
        grid.pack(fill="x")
        for i in range(2):
            grid.grid_columnconfigure(i, weight=1, uniform="stat")

        for col, (key, label) in enumerate(
            [
                ("total_processed", "Traitées"),
                ("with_metadata", "Avec méta"),
                ("with_ai_analysis", "Avec IA"),
                ("recent_errors", "Erreurs 24 h"),
            ]
        ):
            self._stat_card(grid, key, label, col % 2, col // 2)

    def _stat_card(self, parent: ctk.CTkFrame, key: str, label: str, col: int, row: int) -> None:
        card = ctk.CTkFrame(parent, fg_color=get_color("bg"), corner_radius=RADIUS_MD)
        card.grid(row=row, column=col, sticky="ew", padx=SPACE_XS, pady=SPACE_XS)
        ctk.CTkLabel(
            card,
            text=label,
            font=get_font("small"),
            text_color=get_color("fg_muted"),
            anchor="w",
        ).pack(fill="x", padx=SPACE_SM, pady=(SPACE_XS, 0))
        value = ctk.CTkLabel(
            card,
            text="—",
            font=get_font("h3"),
            text_color=get_color("fg"),
            anchor="w",
        )
        value.pack(fill="x", padx=SPACE_SM, pady=(0, SPACE_XS))
        self._stat_values[key] = value

    def _build_audit_tail(self, parent: ctk.CTkFrame, row: int) -> None:
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.grid(row=row, column=0, sticky="nsew", padx=SPACE_MD, pady=(SPACE_SM, SPACE_SM))
        section.grid_columnconfigure(0, weight=1)
        section.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(
            section,
            text="JOURNAL LIVE",
            font=get_font("small"),
            text_color=get_color("fg_subtle"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w", pady=(0, SPACE_XS))

        self._audit_box = ctk.CTkScrollableFrame(
            section,
            fg_color=get_color("bg"),
            corner_radius=RADIUS_MD,
        )
        self._audit_box.grid(row=1, column=0, sticky="nsew")
        self._audit_box.grid_columnconfigure(0, weight=1)

        # Pre-create empty rows so layout doesn't jump.
        for i in range(AUDIT_TAIL_SIZE):
            label = ctk.CTkLabel(
                self._audit_box,
                text="",
                font=get_font("code"),
                text_color=get_color("fg_muted"),
                anchor="w",
                justify="left",
            )
            label.grid(row=i, column=0, sticky="ew", padx=SPACE_SM, pady=0)
            self._audit_lines.append(label)

    def _build_quick_actions(self, parent: ctk.CTkFrame, row: int) -> None:
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.grid(row=row, column=0, sticky="ew", padx=SPACE_MD, pady=(SPACE_SM, SPACE_MD))
        section.grid_columnconfigure((0, 1, 2), weight=1, uniform="qa")

        targets = [
            ("⚙", "Paramètres", "settings"),
            ("📜", "Historique", "audit"),
            ("🤖", "Modèle IA", "ai_control"),
            ("✓", "Valider", "validate"),
            ("↑", "Téléverser", "upload"),
            ("✎", "Édition", "editor"),
        ]
        for idx, (icon, label, view_id) in enumerate(targets):
            btn = ctk.CTkButton(
                section,
                text=f"{icon}  {label}",
                anchor="w",
                fg_color=get_color("bg"),
                hover_color=get_color("bg_hover"),
                text_color=get_color("fg"),
                font=get_font("body"),
                corner_radius=RADIUS_MD,
                height=30,
                command=lambda v=view_id: self.app.router.navigate_to(v),
            )
            btn.grid(row=idx // 3, column=idx % 3, sticky="ew", padx=SPACE_XS, pady=2)

    # ------------------------------------------------------------------
    # Public API

    def start_auto_refresh(self) -> None:
        """Begin the periodic refresh loop. Idempotent."""
        self.refresh()

    def stop_auto_refresh(self) -> None:
        """Cancel any pending after() callback."""
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def refresh(self) -> None:
        """One refresh cycle: schedules the next call."""
        self._refresh_static_status()
        self._refresh_dynamic_async()
        self._after_id = self.after(REFRESH_INTERVAL_MS, self.refresh)

    # ------------------------------------------------------------------
    # Refresh internals

    def _refresh_static_status(self) -> None:
        api = self.app.api
        # Backend
        if api is None:
            self._set_status("backend", "Indisponible", "warning")
            self._set_status("exiftool", "—", "muted")
            self._set_status("ollama", "—", "muted")
            self._set_status("model", "—", "muted")
            self._set_status("workers", "—", "muted")
            return
        self._set_status("backend", "Disponible", "success")
        self._set_status(
            "exiftool",
            "OK" if api.exiftool_available else "Absent",
            "success" if api.exiftool_available else "warning",
        )
        try:
            workers = int(api.get_setting("max_workers", 4))
        except Exception:
            workers = 4
        self._set_status("workers", str(workers), "muted")

    def _refresh_dynamic_async(self) -> None:
        api = self.app.api
        if api is None:
            return
        threading.Thread(target=self._refresh_dynamic_worker, args=(api,), daemon=True).start()

    def _refresh_dynamic_worker(self, api: Any) -> None:
        # Stats
        try:
            stats = api.get_statistics()
        except Exception:
            logger.exception("get_statistics failed")
            stats = {}
        # AI status (HTTP call)
        try:
            ai_status = api.check_ai_status() if hasattr(api, "check_ai_status") else {}
        except Exception:
            logger.exception("check_ai_status failed")
            ai_status = {"available": False, "message": "erreur"}
        # Audit tail
        try:
            since = datetime.now() - timedelta(hours=24)
            logs = api.database.get_audit_logs(start_date=since, limit=AUDIT_TAIL_SIZE)
        except Exception:
            logger.exception("audit tail fetch failed")
            logs = []

        self.after(0, lambda s=stats, a=ai_status, lg=logs: self._apply_dynamic(s, a, lg))

    def _apply_dynamic(self, stats: dict, ai_status: dict, logs: list) -> None:
        # Stats
        for key, label in self._stat_values.items():
            label.configure(text=fmt_int(int(stats.get(key, 0))))

        # Ollama + model
        if ai_status.get("available"):
            self._set_status("ollama", f"En ligne {ai_status.get('version', '')}".strip(), "success")
            current = ai_status.get("current_model")
            if current:
                self._set_status("model", current, "success")
            else:
                models_count = ai_status.get("vision_models", 0)
                self._set_status(
                    "model", f"{fmt_int(int(models_count))} dispo", "muted" if not models_count else "success"
                )
        else:
            status = ai_status.get("status", "—")
            if status == "not_initialized":
                self._set_status("ollama", "Non initialisé", "muted")
                self._set_status("model", "—", "muted")
            else:
                self._set_status("ollama", ai_status.get("message", "Hors ligne"), "warning")
                self._set_status("model", "—", "muted")

        # Audit tail — newest first
        for i, label in enumerate(self._audit_lines):
            if i < len(logs):
                log = logs[i]
                ts = log.timestamp.strftime("%H:%M:%S")
                action = log.action_type.value
                fname = Path(log.file_path).name if log.file_path else "—"
                ok = "✓" if log.success else "✗"
                color = get_color("success") if log.success else get_color("error")
                # Truncate filename to keep line short.
                if len(fname) > 22:
                    fname = fname[:19] + "…"
                label.configure(
                    text=f"{ts}  {ok} {action:<14} {fname}",
                    text_color=color,
                )
            else:
                label.configure(text="", text_color=get_color("fg_muted"))

    def _set_status(self, key: str, value_text: str, color_kind: str) -> None:
        color_map = {
            "success": get_color("success"),
            "warning": get_color("warning"),
            "error": get_color("error"),
            "muted": get_color("fg_muted"),
        }
        color = color_map.get(color_kind, get_color("fg_muted"))
        if key in self._status_dots:
            self._status_dots[key].configure(text_color=color)
        if key in self._status_rows:
            self._status_rows[key].configure(text=value_text)

    def refresh_theme(self) -> None:
        self.configure(fg_color=get_color("bg_elevated"), border_color=get_color("border"))
