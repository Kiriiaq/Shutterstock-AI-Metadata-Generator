"""
Page Journal — Logs détaillés des opérations.
"""

import customtkinter as ctk
from typing import Callable
from ...core.params import ShutterstockParams


COLORS = {"text_muted": ("#64748B", "#94A3B8"), "border": ("#E2E8F0", "#334155")}


class PageJournal(ctk.CTkFrame):
    """Page de journal/logs."""

    PAGE_KEY = "journal"

    def __init__(self, parent, params: ShutterstockParams, on_change: Callable, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self.params = params
        self._on_change = on_change
        self._build_ui()

    def _build_ui(self):
        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(header, text="Journal", font=ctk.CTkFont(size=20, weight="bold")).pack(side="left")
        ctk.CTkButton(header, text="Effacer", width=80, height=28,
                      fg_color="transparent", border_width=1,
                      command=self._clear_log).pack(side="right")

        # Zone de log
        self.log_text = ctk.CTkTextbox(
            self, font=ctk.CTkFont(family="Consolas", size=10),
            state="disabled"
        )
        self.log_text.pack(fill="both", expand=True)

    def add_log(self, message: str):
        """Ajoute un message au journal."""
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def refresh(self, params: ShutterstockParams):
        self.params = params
