"""
Settings Page - Application configuration and preferences
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import json
import logging

from ...modules.storage.database import Database
from ...modules.engines.iptc_engine import IPTCEngine, IPTCTemplate
from ..components.tooltips import add_tooltip, InfoButton, tooltip_manager

logger = logging.getLogger(__name__)


class SettingsPage(ctk.CTkFrame):
    """
    Settings page for application configuration
    """

    # Default settings
    DEFAULT_SETTINGS = {
        # General
        "theme": "dark",
        "language": "en",

        # Ollama
        "ollama_url": "http://localhost:11434",
        "ollama_model": "llama3.2-vision:11b",
        "ollama_timeout": 120,

        # Processing
        "max_workers": 4,
        "batch_size": 50,
        "min_resolution_mp": 4.0,
        "supported_formats": ["jpg", "jpeg", "tif", "tiff", "png", "eps"],

        # Metadata
        "default_copyright": "",
        "default_byline": "",
        "write_iptc": True,
        "write_xmp": True,
        "create_backup": True,

        # ExifTool
        "exiftool_path": "",

        # FTPS
        "ftps_host": "ftps.shutterstock.com",
        "ftps_port": 21,
        "ftps_username": "",
        "ftps_password": "",

        # Paths
        "default_source_folder": "",
        "default_output_folder": "",

        # Advanced
        "debug_mode": False,
        "log_level": "INFO",
    }

    def __init__(
        self,
        parent,
        database: Database,
        on_settings_changed: Optional[Callable[[Dict[str, Any]], None]] = None,
        **kwargs
    ):
        super().__init__(parent, **kwargs)

        self.database = database
        self.on_settings_changed = on_settings_changed
        self.iptc_engine = IPTCEngine()

        # Load settings
        self._settings = self._load_settings()

        # Setup UI
        self._create_widgets()

    def _load_settings(self) -> Dict[str, Any]:
        """Load settings from database"""
        settings = self.DEFAULT_SETTINGS.copy()

        try:
            stored = self.database.get_all_settings()
            settings.update(stored)
        except Exception as e:
            logger.warning(f"Failed to load settings: {e}")

        return settings

    def _save_settings(self):
        """Save settings to database"""
        try:
            for key, value in self._settings.items():
                self.database.set_setting(key, value)

            if self.on_settings_changed:
                self.on_settings_changed(self._settings)

            messagebox.showinfo("Settings", "Settings saved successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")

    def _create_widgets(self):
        """Create UI widgets"""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Main scrollable frame
        main_scroll = ctk.CTkScrollableFrame(self)
        main_scroll.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        main_scroll.grid_columnconfigure(0, weight=1)

        # ============ Ollama Section ============
        self._create_section(main_scroll, "Ollama Configuration", self._create_ollama_settings)

        # ============ Processing Section ============
        self._create_section(main_scroll, "Processing Options", self._create_processing_settings)

        # ============ Metadata Section ============
        self._create_section(main_scroll, "Metadata Defaults", self._create_metadata_settings)

        # ============ FTPS Section ============
        self._create_section(main_scroll, "FTPS Upload", self._create_ftps_settings)

        # ============ Paths Section ============
        self._create_section(main_scroll, "Default Paths", self._create_paths_settings)

        # ============ Templates Section ============
        self._create_section(main_scroll, "IPTC Templates", self._create_templates_settings)

        # ============ Advanced Section ============
        self._create_section(main_scroll, "Advanced", self._create_advanced_settings)

        # ============ Action Buttons ============
        button_frame = ctk.CTkFrame(self)
        button_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")

        ctk.CTkButton(
            button_frame,
            text="Save Settings",
            fg_color="green",
            command=self._save_all
        ).pack(side="left", padx=10, pady=10)

        ctk.CTkButton(
            button_frame,
            text="Reset to Defaults",
            fg_color="gray",
            command=self._reset_defaults
        ).pack(side="left", padx=10, pady=10)

        ctk.CTkButton(
            button_frame,
            text="Export Settings",
            command=self._export_settings
        ).pack(side="right", padx=10, pady=10)

        ctk.CTkButton(
            button_frame,
            text="Import Settings",
            command=self._import_settings
        ).pack(side="right", padx=10, pady=10)

    def _create_section(self, parent, title: str, content_func: Callable):
        """Create a collapsible settings section"""
        section = ctk.CTkFrame(parent)
        section.pack(fill="x", pady=5)
        section.grid_columnconfigure(0, weight=1)

        # Header
        header = ctk.CTkFrame(section)
        header.pack(fill="x")

        ctk.CTkLabel(
            header,
            text=title,
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(side="left", padx=10, pady=5)

        # Content
        content = ctk.CTkFrame(section)
        content.pack(fill="x", padx=10, pady=5)

        content_func(content)

    def _create_ollama_settings(self, parent):
        """Create Ollama settings"""
        # URL
        row1 = ctk.CTkFrame(parent)
        row1.pack(fill="x", pady=2)

        ctk.CTkLabel(row1, text="Ollama URL:", width=150).pack(side="left")
        self.ollama_url = ctk.CTkEntry(row1, width=300)
        self.ollama_url.insert(0, self._settings.get("ollama_url", ""))
        self.ollama_url.pack(side="left", padx=5)
        add_tooltip(self.ollama_url, "ollama_url")

        ctk.CTkButton(
            row1,
            text="Test",
            width=60,
            command=self._test_ollama
        ).pack(side="left", padx=5)

        # Model
        row2 = ctk.CTkFrame(parent)
        row2.pack(fill="x", pady=2)

        ctk.CTkLabel(row2, text="Vision Model:", width=150).pack(side="left")
        self.ollama_model = ctk.CTkComboBox(
            row2,
            values=[
                "llama3.2-vision:11b",
                "llama3.2-vision:90b",
                "llava:7b",
                "llava:13b",
                "moondream:1.8b"
            ],
            width=250
        )
        self.ollama_model.set(self._settings.get("ollama_model", "llama3.2-vision:11b"))
        self.ollama_model.pack(side="left", padx=5)
        add_tooltip(self.ollama_model, "ollama_model")

        # Timeout
        row3 = ctk.CTkFrame(parent)
        row3.pack(fill="x", pady=2)

        ctk.CTkLabel(row3, text="Timeout (seconds):", width=150).pack(side="left")
        self.ollama_timeout = ctk.CTkEntry(row3, width=80)
        self.ollama_timeout.insert(0, str(self._settings.get("ollama_timeout", 120)))
        self.ollama_timeout.pack(side="left", padx=5)
        add_tooltip(self.ollama_timeout, "ollama_timeout")

    def _create_processing_settings(self, parent):
        """Create processing settings"""
        # Max workers
        row1 = ctk.CTkFrame(parent)
        row1.pack(fill="x", pady=2)

        ctk.CTkLabel(row1, text="Max Workers:", width=150).pack(side="left")
        self.max_workers = ctk.CTkSlider(
            row1,
            from_=1,
            to=16,
            number_of_steps=15,
            width=200
        )
        self.max_workers.set(self._settings.get("max_workers", 4))
        self.max_workers.pack(side="left", padx=5)
        add_tooltip(self.max_workers, "max_workers")

        self.max_workers_label = ctk.CTkLabel(row1, text=str(int(self.max_workers.get())))
        self.max_workers_label.pack(side="left", padx=5)
        self.max_workers.configure(command=lambda v: self.max_workers_label.configure(text=str(int(v))))

        # Batch size
        row2 = ctk.CTkFrame(parent)
        row2.pack(fill="x", pady=2)

        ctk.CTkLabel(row2, text="Batch Size:", width=150).pack(side="left")
        self.batch_size = ctk.CTkEntry(row2, width=80)
        self.batch_size.insert(0, str(self._settings.get("batch_size", 50)))
        self.batch_size.pack(side="left", padx=5)
        add_tooltip(self.batch_size, "batch_size")
        ctk.CTkLabel(row2, text="(Shutterstock max: 50)", text_color="gray").pack(side="left")

        # Min resolution
        row3 = ctk.CTkFrame(parent)
        row3.pack(fill="x", pady=2)

        ctk.CTkLabel(row3, text="Min Resolution (MP):", width=150).pack(side="left")
        self.min_resolution = ctk.CTkEntry(row3, width=80)
        self.min_resolution.insert(0, str(self._settings.get("min_resolution_mp", 4.0)))
        self.min_resolution.pack(side="left", padx=5)
        add_tooltip(self.min_resolution, "min_resolution")

    def _create_metadata_settings(self, parent):
        """Create metadata default settings"""
        # Default copyright
        row1 = ctk.CTkFrame(parent)
        row1.pack(fill="x", pady=2)

        ctk.CTkLabel(row1, text="Default Copyright:", width=150).pack(side="left")
        self.default_copyright = ctk.CTkEntry(row1, width=350)
        self.default_copyright.insert(0, self._settings.get("default_copyright", ""))
        self.default_copyright.pack(side="left", padx=5)
        add_tooltip(self.default_copyright, "default_copyright")

        # Default byline
        row2 = ctk.CTkFrame(parent)
        row2.pack(fill="x", pady=2)

        ctk.CTkLabel(row2, text="Default Byline:", width=150).pack(side="left")
        self.default_byline = ctk.CTkEntry(row2, width=350)
        self.default_byline.insert(0, self._settings.get("default_byline", ""))
        self.default_byline.pack(side="left", padx=5)
        add_tooltip(self.default_byline, "default_byline")

        # Write options
        row3 = ctk.CTkFrame(parent)
        row3.pack(fill="x", pady=2)

        self.write_iptc_var = ctk.BooleanVar(value=self._settings.get("write_iptc", True))
        write_iptc_cb = ctk.CTkCheckBox(
            row3,
            text="Write IPTC",
            variable=self.write_iptc_var
        )
        write_iptc_cb.pack(side="left", padx=10)
        add_tooltip(write_iptc_cb, "write_iptc")

        self.write_xmp_var = ctk.BooleanVar(value=self._settings.get("write_xmp", True))
        write_xmp_cb = ctk.CTkCheckBox(
            row3,
            text="Write XMP",
            variable=self.write_xmp_var
        )
        write_xmp_cb.pack(side="left", padx=10)
        add_tooltip(write_xmp_cb, "write_xmp")

        self.create_backup_var = ctk.BooleanVar(value=self._settings.get("create_backup", True))
        create_backup_cb = ctk.CTkCheckBox(
            row3,
            text="Create Backup (_original)",
            variable=self.create_backup_var
        )
        create_backup_cb.pack(side="left", padx=10)
        add_tooltip(create_backup_cb, "create_backup")

        # ExifTool path
        row4 = ctk.CTkFrame(parent)
        row4.pack(fill="x", pady=2)

        ctk.CTkLabel(row4, text="ExifTool Path:", width=150).pack(side="left")
        self.exiftool_path = ctk.CTkEntry(row4, width=300)
        self.exiftool_path.insert(0, self._settings.get("exiftool_path", ""))
        self.exiftool_path.pack(side="left", padx=5)
        add_tooltip(self.exiftool_path, "exiftool_path")

        ctk.CTkButton(
            row4,
            text="Browse",
            width=60,
            command=lambda: self._browse_file(self.exiftool_path, [("Executable", "*.exe")])
        ).pack(side="left", padx=5)

    def _create_ftps_settings(self, parent):
        """Create FTPS settings"""
        # Host
        row1 = ctk.CTkFrame(parent)
        row1.pack(fill="x", pady=2)

        ctk.CTkLabel(row1, text="FTPS Host:", width=150).pack(side="left")
        self.ftps_host = ctk.CTkEntry(row1, width=250)
        self.ftps_host.insert(0, self._settings.get("ftps_host", "ftps.shutterstock.com"))
        self.ftps_host.pack(side="left", padx=5)
        add_tooltip(self.ftps_host, "ftps_host")

        ctk.CTkLabel(row1, text="Port:").pack(side="left", padx=5)
        self.ftps_port = ctk.CTkEntry(row1, width=60)
        self.ftps_port.insert(0, str(self._settings.get("ftps_port", 21)))
        self.ftps_port.pack(side="left", padx=5)
        add_tooltip(self.ftps_port, "ftps_port")

        # Username
        row2 = ctk.CTkFrame(parent)
        row2.pack(fill="x", pady=2)

        ctk.CTkLabel(row2, text="Username:", width=150).pack(side="left")
        self.ftps_username = ctk.CTkEntry(row2, width=250)
        self.ftps_username.insert(0, self._settings.get("ftps_username", ""))
        self.ftps_username.pack(side="left", padx=5)
        add_tooltip(self.ftps_username, "ftps_username")

        # Password
        row3 = ctk.CTkFrame(parent)
        row3.pack(fill="x", pady=2)

        ctk.CTkLabel(row3, text="Password:", width=150).pack(side="left")
        self.ftps_password = ctk.CTkEntry(row3, width=250, show="*")
        self.ftps_password.insert(0, self._settings.get("ftps_password", ""))
        self.ftps_password.pack(side="left", padx=5)
        add_tooltip(self.ftps_password, "ftps_password")

        ctk.CTkButton(
            row3,
            text="Test Connection",
            width=120,
            command=self._test_ftps
        ).pack(side="left", padx=10)

    def _create_paths_settings(self, parent):
        """Create default paths settings"""
        # Source folder
        row1 = ctk.CTkFrame(parent)
        row1.pack(fill="x", pady=2)

        ctk.CTkLabel(row1, text="Default Source:", width=150).pack(side="left")
        self.default_source = ctk.CTkEntry(row1, width=350)
        self.default_source.insert(0, self._settings.get("default_source_folder", ""))
        self.default_source.pack(side="left", padx=5)

        ctk.CTkButton(
            row1,
            text="Browse",
            width=60,
            command=lambda: self._browse_folder(self.default_source)
        ).pack(side="left", padx=5)

        # Output folder
        row2 = ctk.CTkFrame(parent)
        row2.pack(fill="x", pady=2)

        ctk.CTkLabel(row2, text="Default Output:", width=150).pack(side="left")
        self.default_output = ctk.CTkEntry(row2, width=350)
        self.default_output.insert(0, self._settings.get("default_output_folder", ""))
        self.default_output.pack(side="left", padx=5)

        ctk.CTkButton(
            row2,
            text="Browse",
            width=60,
            command=lambda: self._browse_folder(self.default_output)
        ).pack(side="left", padx=5)

    def _create_templates_settings(self, parent):
        """Create IPTC templates settings"""
        # Template list
        row1 = ctk.CTkFrame(parent)
        row1.pack(fill="x", pady=2)

        ctk.CTkLabel(row1, text="Available Templates:").pack(anchor="w")

        self.template_list = ctk.CTkTextbox(row1, height=100, state="disabled")
        self.template_list.pack(fill="x", pady=5)

        # Refresh template list
        self._refresh_template_list()

        # Buttons
        row2 = ctk.CTkFrame(parent)
        row2.pack(fill="x", pady=2)

        ctk.CTkButton(
            row2,
            text="Create New Template",
            command=self._create_template
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            row2,
            text="Import Templates",
            command=self._import_templates
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            row2,
            text="Export Templates",
            command=self._export_templates
        ).pack(side="left", padx=5)

    def _create_advanced_settings(self, parent):
        """Create advanced settings"""
        # Debug mode
        row1 = ctk.CTkFrame(parent)
        row1.pack(fill="x", pady=2)

        self.debug_mode_var = ctk.BooleanVar(value=self._settings.get("debug_mode", False))
        debug_cb = ctk.CTkCheckBox(
            row1,
            text="Enable Debug Mode",
            variable=self.debug_mode_var
        )
        debug_cb.pack(side="left", padx=10)
        add_tooltip(debug_cb, "debug_mode")

        # Log level
        ctk.CTkLabel(row1, text="Log Level:").pack(side="left", padx=(20, 5))
        self.log_level = ctk.CTkComboBox(
            row1,
            values=["DEBUG", "INFO", "WARNING", "ERROR"],
            width=100
        )
        self.log_level.set(self._settings.get("log_level", "INFO"))
        self.log_level.pack(side="left", padx=5)
        add_tooltip(self.log_level, "log_level")

        # Database info
        row2 = ctk.CTkFrame(parent)
        row2.pack(fill="x", pady=2)

        ctk.CTkLabel(
            row2,
            text=f"Database: {self.database.db_path}",
            text_color="gray"
        ).pack(anchor="w")

        ctk.CTkButton(
            row2,
            text="Vacuum Database",
            width=120,
            command=self._vacuum_database
        ).pack(anchor="w", pady=5)

    def _browse_file(self, entry: ctk.CTkEntry, filetypes: list):
        """Browse for a file"""
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            entry.delete(0, "end")
            entry.insert(0, path)

    def _browse_folder(self, entry: ctk.CTkEntry):
        """Browse for a folder"""
        path = filedialog.askdirectory()
        if path:
            entry.delete(0, "end")
            entry.insert(0, path)

    def _test_ollama(self):
        """Test Ollama connection"""
        import requests

        url = self.ollama_url.get().strip()
        if not url:
            messagebox.showwarning("Warning", "Please enter Ollama URL")
            return

        try:
            response = requests.get(f"{url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                messagebox.showinfo(
                    "Ollama Connected",
                    f"Connected successfully!\n\nAvailable models:\n" + "\n".join(model_names[:10])
                )
            else:
                messagebox.showerror("Error", f"Ollama returned status: {response.status_code}")
        except Exception as e:
            messagebox.showerror("Connection Failed", f"Failed to connect to Ollama:\n{e}")

    def _test_ftps(self):
        """Test FTPS connection"""
        messagebox.showinfo("FTPS Test", "FTPS connection test coming soon...")

    def _refresh_template_list(self):
        """Refresh template list display"""
        templates = self.iptc_engine.list_templates()

        self.template_list.configure(state="normal")
        self.template_list.delete("1.0", "end")

        for name in templates:
            template = self.iptc_engine.get_template(name)
            if template:
                self.template_list.insert("end", f"- {template.name}: {template.description}\n")

        self.template_list.configure(state="disabled")

    def _create_template(self):
        """Create a new IPTC template"""
        messagebox.showinfo("Templates", "Template editor coming soon...")

    def _import_templates(self):
        """Import templates from file"""
        path = filedialog.askopenfilename(
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if path:
            try:
                self.iptc_engine.load_templates(Path(path))
                self._refresh_template_list()
                messagebox.showinfo("Import", "Templates imported successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import templates: {e}")

    def _export_templates(self):
        """Export templates to file"""
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")]
        )
        if path:
            try:
                self.iptc_engine.save_templates(Path(path))
                messagebox.showinfo("Export", f"Templates exported to:\n{path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export templates: {e}")

    def _vacuum_database(self):
        """Vacuum/optimize database"""
        try:
            self.database.vacuum()
            messagebox.showinfo("Database", "Database optimized successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to vacuum database: {e}")

    def _save_all(self):
        """Save all settings"""
        # Collect values from UI
        self._settings["ollama_url"] = self.ollama_url.get().strip()
        self._settings["ollama_model"] = self.ollama_model.get()
        self._settings["ollama_timeout"] = int(self.ollama_timeout.get() or 120)

        self._settings["max_workers"] = int(self.max_workers.get())
        self._settings["batch_size"] = int(self.batch_size.get() or 50)
        self._settings["min_resolution_mp"] = float(self.min_resolution.get() or 4.0)

        self._settings["default_copyright"] = self.default_copyright.get().strip()
        self._settings["default_byline"] = self.default_byline.get().strip()
        self._settings["write_iptc"] = self.write_iptc_var.get()
        self._settings["write_xmp"] = self.write_xmp_var.get()
        self._settings["create_backup"] = self.create_backup_var.get()
        self._settings["exiftool_path"] = self.exiftool_path.get().strip()

        self._settings["ftps_host"] = self.ftps_host.get().strip()
        self._settings["ftps_port"] = int(self.ftps_port.get() or 21)
        self._settings["ftps_username"] = self.ftps_username.get().strip()
        self._settings["ftps_password"] = self.ftps_password.get()

        self._settings["default_source_folder"] = self.default_source.get().strip()
        self._settings["default_output_folder"] = self.default_output.get().strip()

        self._settings["debug_mode"] = self.debug_mode_var.get()
        self._settings["log_level"] = self.log_level.get()

        self._save_settings()

    def _reset_defaults(self):
        """Reset all settings to defaults"""
        if not messagebox.askyesno("Reset Settings", "Reset all settings to defaults?"):
            return

        self._settings = self.DEFAULT_SETTINGS.copy()
        self._save_settings()

        # Reload UI
        messagebox.showinfo("Reset", "Settings reset to defaults. Please restart the application.")

    def _export_settings(self):
        """Export settings to JSON file"""
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")]
        )
        if path:
            try:
                # Don't export sensitive data
                export_settings = {k: v for k, v in self._settings.items()
                                   if k not in ["ftps_password"]}

                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(export_settings, f, indent=2)

                messagebox.showinfo("Export", f"Settings exported to:\n{path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {e}")

    def _import_settings(self):
        """Import settings from JSON file"""
        path = filedialog.askopenfilename(
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if path:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    imported = json.load(f)

                self._settings.update(imported)
                self._save_settings()

                messagebox.showinfo("Import", "Settings imported. Please restart the application.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import: {e}")

    def get_settings(self) -> Dict[str, Any]:
        """Get current settings"""
        return self._settings.copy()
