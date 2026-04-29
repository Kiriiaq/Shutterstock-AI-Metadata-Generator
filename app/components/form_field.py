"""FormField — label + input + error zone, used in all forms.

The widget kind is configurable via *widget_factory*. Default is a
CTkEntry; helpers ``entry_factory``, ``combo_factory``, ``textbox_factory``,
``switch_factory`` cover the common cases. Required fields display a red
asterisk; the error zone reserves vertical space so the layout doesn't
jump on first error.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import customtkinter as ctk

from app.config.theme import (
    SPACE_XS,
    get_color,
    get_font,
)

WidgetFactory = Callable[[ctk.CTkFrame], ctk.CTkBaseClass]
Validator = Callable[[Any], str | None]


def entry_factory(master: ctk.CTkFrame) -> ctk.CTkEntry:
    return ctk.CTkEntry(
        master,
        font=get_font("body"),
        fg_color=get_color("bg"),
        text_color=get_color("fg"),
        border_color=get_color("border"),
        height=32,
    )


def combo_factory(values: list[str]) -> WidgetFactory:
    def factory(master: ctk.CTkFrame) -> ctk.CTkComboBox:
        return ctk.CTkComboBox(
            master,
            values=values,
            font=get_font("body"),
            fg_color=get_color("bg"),
            text_color=get_color("fg"),
            border_color=get_color("border"),
            button_color=get_color("border"),
            button_hover_color=get_color("border_strong"),
            height=32,
        )

    return factory


def textbox_factory(height: int = 80) -> WidgetFactory:
    def factory(master: ctk.CTkFrame) -> ctk.CTkTextbox:
        return ctk.CTkTextbox(
            master,
            font=get_font("body"),
            fg_color=get_color("bg"),
            text_color=get_color("fg"),
            border_color=get_color("border"),
            border_width=1,
            height=height,
        )

    return factory


def switch_factory(text: str = "") -> WidgetFactory:
    def factory(master: ctk.CTkFrame) -> ctk.CTkSwitch:
        return ctk.CTkSwitch(master, text=text, font=get_font("body"))

    return factory


class FormField(ctk.CTkFrame):
    """One labelled input with optional validator and error display."""

    def __init__(
        self,
        master: ctk.CTkFrame,
        *,
        label: str,
        required: bool = False,
        widget_factory: WidgetFactory = entry_factory,
        validator: Validator | None = None,
        helper: str | None = None,
    ) -> None:
        super().__init__(master, fg_color="transparent")
        self.grid_columnconfigure(0, weight=1)

        self._label_text = label
        self._required = required
        self._validator = validator
        self._helper = helper
        self._error_message: str | None = None

        self._build_label()
        self._widget = widget_factory(self)
        self._widget.grid(row=1, column=0, sticky="ew")

        self._error_label = ctk.CTkLabel(
            self,
            text=helper or "",
            font=get_font("small"),
            text_color=get_color("fg_subtle") if helper else get_color("error"),
            anchor="w",
            justify="left",
        )
        self._error_label.grid(row=2, column=0, sticky="ew", pady=(SPACE_XS, 0))

        # Validate on focus-out for entry-like widgets.
        try:
            self._widget.bind("<FocusOut>", lambda _e: self.validate())
        except Exception:
            pass

    # ------------------------------------------------------------------

    @property
    def widget(self) -> ctk.CTkBaseClass:
        return self._widget

    @property
    def value(self) -> Any:
        return self._read_value()

    def set_value(self, value: Any) -> None:
        self._write_value(value)

    def set_error(self, message: str | None) -> None:
        self._error_message = message
        if message:
            self._error_label.configure(text=message, text_color=get_color("error"))
            try:
                self._widget.configure(border_color=get_color("error"))
            except Exception:
                pass
        else:
            self._error_label.configure(
                text=self._helper or "",
                text_color=get_color("fg_subtle") if self._helper else get_color("error"),
            )
            try:
                self._widget.configure(border_color=get_color("border"))
            except Exception:
                pass

    def validate(self) -> bool:
        """Run the validator. Returns True if value is acceptable."""
        if self._validator is None:
            self.set_error(None)
            return True
        error = self._validator(self.value)
        self.set_error(error)
        return error is None

    # ------------------------------------------------------------------

    def _build_label(self) -> None:
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.grid(row=0, column=0, sticky="ew", pady=(0, SPACE_XS))
        ctk.CTkLabel(
            row,
            text=self._label_text,
            font=get_font("body_strong"),
            text_color=get_color("fg"),
            anchor="w",
        ).pack(side="left")
        if self._required:
            ctk.CTkLabel(
                row,
                text=" *",
                font=get_font("body_strong"),
                text_color=get_color("error"),
            ).pack(side="left")

    def _read_value(self) -> Any:
        if isinstance(self._widget, ctk.CTkEntry):
            return self._widget.get()
        if isinstance(self._widget, ctk.CTkComboBox):
            return self._widget.get()
        if isinstance(self._widget, ctk.CTkTextbox):
            return self._widget.get("1.0", "end").rstrip("\n")
        if isinstance(self._widget, ctk.CTkSwitch):
            return bool(self._widget.get())
        return None

    def _write_value(self, value: Any) -> None:
        if isinstance(self._widget, ctk.CTkEntry):
            self._widget.delete(0, "end")
            if value is not None:
                self._widget.insert(0, str(value))
        elif isinstance(self._widget, ctk.CTkComboBox):
            self._widget.set(str(value) if value is not None else "")
        elif isinstance(self._widget, ctk.CTkTextbox):
            self._widget.delete("1.0", "end")
            if value is not None:
                self._widget.insert("1.0", str(value))
        elif isinstance(self._widget, ctk.CTkSwitch):
            (self._widget.select if value else self._widget.deselect)()
