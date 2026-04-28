"""
Sidebar de navigation Shutterstock Analyzer.
"""

import customtkinter as ctk
from typing import Callable, Dict


PAGES = {
    "source": {"icon": "📂", "label": "Source"},
    "modele": {"icon": "🤖", "label": "Modèle IA"},
    "analyse": {"icon": "🔍", "label": "Analyse"},
    "validation": {"icon": "✅", "label": "Validation"},
    "upload": {"icon": "📤", "label": "Upload FTPS"},
    "journal": {"icon": "📋", "label": "Journal"},
}

COLORS = {
    "bg": ("#F1F5F9", "#0F172A"),
    "active": ("#E2E8F0", "#1E293B"),
    "hover": ("#F1F5F9", "#1E293B"),
    "text": ("#1E293B", "#F1F5F9"),
    "text_muted": ("#64748B", "#94A3B8"),
    "separator": ("#E2E8F0", "#334155"),
    "success": ("#10B981", "#34D399"),
    "warning": ("#F59E0B", "#FBBF24"),
    "locked": ("#9CA3AF", "#4B5563"),
}


class Sidebar(ctk.CTkFrame):
    """Sidebar de navigation."""

    def __init__(self, parent, on_navigate: Callable, **kwargs):
        super().__init__(parent, width=180, corner_radius=0, fg_color=COLORS["bg"], **kwargs)
        self.grid_propagate(False)
        self._on_navigate = on_navigate
        self._buttons: Dict[str, ctk.CTkButton] = {}
        self._indicators: Dict[str, ctk.CTkLabel] = {}
        self._active = ""
        self._build_ui()

    def _build_ui(self):
        # Logo
        logo = ctk.CTkFrame(self, fg_color="transparent")
        logo.pack(fill="x", padx=12, pady=(18, 5))
        ctk.CTkLabel(logo, text="Shutterstock",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(logo, text="AI Analyzer",
                     font=ctk.CTkFont(size=10), text_color=COLORS["text_muted"]).pack(anchor="w")

        ctk.CTkFrame(self, height=1, fg_color=COLORS["separator"]).pack(fill="x", padx=12, pady=10)

        # Boutons
        for key, info in PAGES.items():
            frame = ctk.CTkFrame(self, fg_color="transparent")
            frame.pack(fill="x", padx=6, pady=1)

            btn = ctk.CTkButton(
                frame, text=f" {info['icon']}  {info['label']}",
                command=lambda k=key: self._on_navigate(k),
                height=36, corner_radius=8, anchor="w",
                fg_color="transparent", hover_color=COLORS["hover"],
                text_color=COLORS["text"], font=ctk.CTkFont(size=11)
            )
            btn.pack(fill="x", side="left", expand=True)

            ind = ctk.CTkLabel(frame, text="○", width=18,
                               font=ctk.CTkFont(size=10), text_color=COLORS["text_muted"])
            ind.pack(side="right", padx=(0, 4))

            self._buttons[key] = btn
            self._indicators[key] = ind

        ctk.CTkFrame(self, fg_color="transparent").pack(fill="both", expand=True)

        # Switch thème
        self.theme_switch = ctk.CTkSwitch(self, text="Sombre", font=ctk.CTkFont(size=10),
                                          height=20, command=self._toggle)
        self.theme_switch.pack(padx=12, pady=(0, 15), anchor="w")
        if ctk.get_appearance_mode() == "Dark":
            self.theme_switch.select()

    def set_active(self, key: str):
        if self._active in self._buttons:
            self._buttons[self._active].configure(fg_color="transparent")
        if key in self._buttons:
            self._buttons[key].configure(fg_color=COLORS["active"])
        self._active = key

    def set_page_status(self, key: str, status: str):
        if key not in self._indicators:
            return
        status_map = {
            "done": ("✔", COLORS["success"]), "active": ("●", COLORS["warning"]),
            "pending": ("○", COLORS["text_muted"]), "locked": ("🔒", COLORS["locked"]),
        }
        text, color = status_map.get(status, ("○", COLORS["text_muted"]))
        self._indicators[key].configure(text=text, text_color=color)

    def _toggle(self):
        current = ctk.get_appearance_mode()
        ctk.set_appearance_mode("Light" if current == "Dark" else "Dark")
