"""ContextPanel — right-side slide-in panel for contextual details.

Default hidden (width 0). Calling ``open()`` expands it to 320 px and
populates the content area via the builder previously set with
``set_content``. ``close()`` retracts and clears the content.

The panel is a regular CTkFrame placed in the App grid at column 2.
The shell sets its column weight to 0 so it doesn't steal space when
hidden; ``open()`` flips the weight on the App grid via the parent.
"""

from __future__ import annotations

from collections.abc import Callable

import customtkinter as ctk

from app.config.theme import (
    RADIUS_MD,
    SPACE_MD,
    SPACE_SM,
    get_color,
    get_font,
)

ContentBuilder = Callable[[ctk.CTkFrame], None]


class ContextPanel(ctk.CTkFrame):
    """Right-side panel. Width fixed at 320 px when open, 0 when closed."""

    WIDTH = 320

    def __init__(self, master: ctk.CTk) -> None:
        super().__init__(
            master,
            width=0,
            corner_radius=0,
            fg_color=get_color("bg_elevated"),
            border_color=get_color("border"),
            border_width=0,
        )
        self.grid_propagate(False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._is_open = False
        self._title_var = ctk.StringVar(value="")
        self._build_header()

        self._content = ctk.CTkFrame(self, fg_color="transparent")
        self._content.grid(row=1, column=0, sticky="nsew", padx=SPACE_MD, pady=(0, SPACE_MD))
        self._content.grid_columnconfigure(0, weight=1)
        self._content.grid_rowconfigure(0, weight=1)

    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._is_open

    def set_content(self, title: str, builder: ContentBuilder) -> None:
        """Replace the panel content. Builder receives the inner frame."""
        self._title_var.set(title)
        for child in self._content.winfo_children():
            child.destroy()
        try:
            builder(self._content)
        except Exception:
            ctk.CTkLabel(
                self._content,
                text="Erreur de chargement du panneau.",
                text_color=get_color("error"),
                font=get_font("body"),
            ).grid(row=0, column=0, sticky="nsew")
            raise

    def open(self) -> None:
        if self._is_open:
            return
        self._is_open = True
        self.configure(width=self.WIDTH, border_width=1)

    def close(self) -> None:
        if not self._is_open:
            return
        self._is_open = False
        self.configure(width=0, border_width=0)
        self._title_var.set("")
        for child in self._content.winfo_children():
            child.destroy()

    def toggle(self) -> None:
        if self._is_open:
            self.close()
        else:
            self.open()

    def refresh_theme(self) -> None:
        self.configure(
            fg_color=get_color("bg_elevated"),
            border_color=get_color("border"),
        )

    # ------------------------------------------------------------------

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent", height=40)
        header.grid(row=0, column=0, sticky="ew", padx=SPACE_MD, pady=SPACE_SM)
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            textvariable=self._title_var,
            font=get_font("body_strong"),
            text_color=get_color("fg"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew")

        ctk.CTkButton(
            header,
            text="✕",
            width=28,
            height=28,
            corner_radius=RADIUS_MD,
            fg_color="transparent",
            hover_color=get_color("bg_hover"),
            text_color=get_color("fg_muted"),
            font=get_font("body_strong"),
            command=self.close,
        ).grid(row=0, column=1, sticky="e")
