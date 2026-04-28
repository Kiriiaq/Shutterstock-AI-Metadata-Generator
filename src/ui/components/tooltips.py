"""
Tooltips module - Provides tooltip functionality for CustomTkinter widgets
"""

import customtkinter as ctk
from typing import Optional, Dict, Any
import threading


class ToolTip:
    """
    Tooltip widget for CustomTkinter
    Shows helpful text when hovering over widgets
    """

    def __init__(
        self,
        widget: ctk.CTkBaseClass,
        text: str,
        delay: int = 500,
        wrap_length: int = 300,
        bg_color: str = "#333333",
        fg_color: str = "#ffffff",
        font_size: int = 11
    ):
        """
        Create a tooltip for a widget

        Args:
            widget: The widget to attach tooltip to
            text: Tooltip text to display
            delay: Delay in ms before showing tooltip
            wrap_length: Max width before text wraps
            bg_color: Background color
            fg_color: Text color
            font_size: Font size
        """
        self.widget = widget
        self.text = text
        self.delay = delay
        self.wrap_length = wrap_length
        self.bg_color = bg_color
        self.fg_color = fg_color
        self.font_size = font_size

        self.tooltip_window: Optional[ctk.CTkToplevel] = None
        self._after_id: Optional[str] = None
        self._visible = False

        # Bind events
        self.widget.bind("<Enter>", self._on_enter, add="+")
        self.widget.bind("<Leave>", self._on_leave, add="+")
        self.widget.bind("<ButtonPress>", self._on_leave, add="+")

    def _on_enter(self, event=None):
        """Mouse entered widget"""
        self._schedule_show()

    def _on_leave(self, event=None):
        """Mouse left widget"""
        self._cancel_schedule()
        self._hide()

    def _schedule_show(self):
        """Schedule tooltip to show after delay"""
        self._cancel_schedule()
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel_schedule(self):
        """Cancel scheduled show"""
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self):
        """Show the tooltip"""
        if self._visible:
            return

        # Get widget position
        x = self.widget.winfo_rootx()
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5

        # Create tooltip window
        self.tooltip_window = ctk.CTkToplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        # Make sure it's on top
        self.tooltip_window.wm_attributes("-topmost", True)

        # Create tooltip content
        frame = ctk.CTkFrame(
            self.tooltip_window,
            fg_color=self.bg_color,
            corner_radius=6
        )
        frame.pack(fill="both", expand=True)

        label = ctk.CTkLabel(
            frame,
            text=self.text,
            text_color=self.fg_color,
            font=ctk.CTkFont(size=self.font_size),
            wraplength=self.wrap_length,
            justify="left"
        )
        label.pack(padx=10, pady=6)

        self._visible = True

    def _hide(self):
        """Hide the tooltip"""
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None
        self._visible = False

    def update_text(self, new_text: str):
        """Update tooltip text"""
        self.text = new_text


class ToolTipManager:
    """
    Manager for tooltips across the application
    Provides centralized tooltip definitions and easy attachment
    """

    # Predefined tooltips for common UI elements
    TOOLTIPS = {
        # Settings page
        "ollama_url": "URL of your Ollama server (default: http://localhost:11434)",
        "ollama_model": "Vision model to use for image analysis. llama3.2-vision:11b recommended.",
        "ollama_timeout": "Maximum time to wait for AI response in seconds",
        "max_workers": "Number of parallel processing threads. Higher = faster but more CPU usage.",
        "batch_size": "Maximum number of files per batch (Shutterstock limit: 50)",
        "min_resolution": "Minimum image resolution in megapixels. Shutterstock requires 4+ MP.",
        "default_copyright": "Default copyright notice to add to all images",
        "default_byline": "Default photographer/creator name",
        "write_iptc": "Write metadata to IPTC fields (most compatible)",
        "write_xmp": "Write metadata to XMP fields (Adobe standard)",
        "create_backup": "Create backup files before modifying (_original suffix)",
        "exiftool_path": "Path to ExifTool executable. Leave empty for auto-detection.",
        "ftps_host": "Shutterstock FTPS server address",
        "ftps_port": "FTPS port (default: 21)",
        "ftps_username": "Your Shutterstock contributor username",
        "ftps_password": "Your Shutterstock FTP password (from contributor dashboard)",
        "debug_mode": "Enable detailed logging for troubleshooting",
        "log_level": "Minimum log level to record",

        # Write page
        "iptc_title": "Short title for the image (max 64 chars). Used as ObjectName.",
        "iptc_headline": "Descriptive headline (max 256 chars). Main title field.",
        "iptc_caption": "Full description of the image (max 2000 chars). Be detailed!",
        "iptc_keywords": "Comma-separated keywords. Include 7-50 relevant terms.",
        "iptc_byline": "Photographer or creator name (max 32 chars)",
        "iptc_copyright": "Copyright notice (max 128 chars). Example: © 2024 Your Name",
        "iptc_city": "City where photo was taken (max 32 chars)",
        "iptc_state": "State or province (max 32 chars)",
        "iptc_country": "Country name (max 64 chars)",
        "iptc_country_code": "ISO 3166-1 alpha-3 country code (3 chars). Example: USA, FRA, DEU",
        "iptc_instructions": "Special instructions (max 256 chars). Use 'EDITORIAL USE ONLY' for editorial.",
        "xmp_rating": "Star rating 0-5 for internal organization",
        "xmp_label": "Color label for internal organization",
        "xmp_subject": "XMP subject/keywords (comma-separated)",
        "template_selector": "Apply preset template values to fill common fields",
        "backup_checkbox": "Create a backup before writing metadata",
        "dry_run": "Simulate writing without actually modifying files",

        # Audit page
        "action_filter": "Filter logs by action type",
        "date_filter": "Filter logs by time period",
        "success_filter": "Filter by success or failure status",
        "export_logs": "Export filtered logs to JSON or CSV file",

        # Processing
        "recursive_scan": "Include files in subdirectories",
        "exclude_folders": "Folder names to skip (e.g., _backup, thumbs)",
        "exclude_patterns": "File patterns to skip (e.g., *_thumb.*, *.bak)",

        # Validation
        "validation_score": "Overall quality score based on completeness, quality, and SEO",
        "completeness_score": "How complete the metadata is (0-100)",
        "quality_score": "Quality of title, description, and keywords (0-100)",
        "seo_score": "Search engine optimization score (0-100)"
    }

    def __init__(self):
        """Initialize tooltip manager"""
        self._tooltips: Dict[str, ToolTip] = {}

    def attach(
        self,
        widget: ctk.CTkBaseClass,
        key: str,
        custom_text: Optional[str] = None
    ) -> ToolTip:
        """
        Attach a tooltip to a widget

        Args:
            widget: Widget to attach to
            key: Tooltip key from TOOLTIPS dict
            custom_text: Custom text (overrides predefined)

        Returns:
            ToolTip instance
        """
        text = custom_text or self.TOOLTIPS.get(key, key)
        tooltip = ToolTip(widget, text)
        self._tooltips[id(widget)] = tooltip
        return tooltip

    def attach_custom(
        self,
        widget: ctk.CTkBaseClass,
        text: str,
        **kwargs
    ) -> ToolTip:
        """
        Attach a custom tooltip to a widget

        Args:
            widget: Widget to attach to
            text: Tooltip text
            **kwargs: Additional ToolTip parameters

        Returns:
            ToolTip instance
        """
        tooltip = ToolTip(widget, text, **kwargs)
        self._tooltips[id(widget)] = tooltip
        return tooltip

    def remove(self, widget: ctk.CTkBaseClass):
        """Remove tooltip from widget"""
        widget_id = id(widget)
        if widget_id in self._tooltips:
            del self._tooltips[widget_id]

    def get_tooltip_text(self, key: str) -> str:
        """Get predefined tooltip text"""
        return self.TOOLTIPS.get(key, "")


# Global tooltip manager instance
tooltip_manager = ToolTipManager()


def add_tooltip(widget: ctk.CTkBaseClass, key_or_text: str) -> ToolTip:
    """
    Quick function to add a tooltip to a widget

    Args:
        widget: Widget to attach tooltip to
        key_or_text: Either a TOOLTIPS key or custom text

    Returns:
        ToolTip instance
    """
    if key_or_text in ToolTipManager.TOOLTIPS:
        return tooltip_manager.attach(widget, key_or_text)
    else:
        return tooltip_manager.attach_custom(widget, key_or_text)


class InfoButton(ctk.CTkButton):
    """
    Small info button that shows tooltip on hover
    """

    def __init__(
        self,
        parent,
        tooltip_text: str,
        **kwargs
    ):
        """
        Create info button with tooltip

        Args:
            parent: Parent widget
            tooltip_text: Text to show in tooltip
        """
        super().__init__(
            parent,
            text="?",
            width=20,
            height=20,
            corner_radius=10,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color="gray",
            hover_color="#555555",
            **kwargs
        )

        # Attach tooltip
        self.tooltip = ToolTip(self, tooltip_text, delay=200)

    def update_tooltip(self, text: str):
        """Update tooltip text"""
        self.tooltip.update_text(text)
