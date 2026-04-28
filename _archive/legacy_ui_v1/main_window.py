"""
Application principale Shutterstock Analyzer — Nouvelle interface.
Intègre OllamaManager et ImageAnalyzer pour le pipeline complet.
"""

import customtkinter as ctk
import sys
import threading
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

from ..core.params import ShutterstockParams
from ..core.config_manager import ConfigManager
from .components.sidebar import Sidebar
from .components.advanced_window import AdvancedSettingsWindow
from .pages.page_source import PageSource
from .pages.page_model import PageModel
from .pages.page_analyze import PageAnalyze
from .pages.page_validation import PageValidation
from .pages.page_upload import PageUpload
from .pages.page_journal import PageJournal

# Import des classes metier existantes
# Note: These will need to be refactored into the new structure
try:
    from shutterstock_analyzer_unified import OllamaManager, ImageAnalyzer
except ImportError:
    OllamaManager = None
    ImageAnalyzer = None


COLORS = {
    "text_muted": ("#64748B", "#94A3B8"),
    "border": ("#E2E8F0", "#334155"),
    "dirty": ("#F59E0B", "#FBBF24"),
    "success": ("#10B981", "#34D399"),
    "error": ("#EF4444", "#F87171"),
}


class ShutterstockApp(ctk.CTk):
    """Application principale avec sidebar + pages + intégration métier."""

    def __init__(self):
        super().__init__()
        self.title("Shutterstock AI Analyzer")
        # 90% of screen size
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        width = int(screen_width * 0.9)
        height = int(screen_height * 0.9)
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.minsize(850, 600)

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # State
        self.config_mgr = ConfigManager()
        self.params = self.config_mgr.load_profile("default")
        self._pages: Dict[str, object] = {}
        self._current_page: Optional[str] = None
        self._dirty = False
        self._analysis_running = False
        self._analysis_stats = {"total": 0, "success": 0, "failed": 0, "invalid": 0}

        # Intégration métier
        self.ollama_manager = OllamaManager()
        self.image_analyzer: Optional[ImageAnalyzer] = None

        self._build_layout()
        self._navigate("source")
        self._update_sidebar_status()

        # Fermeture propre
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Profile bar (top)
        self._build_profile_bar()

        # Sidebar
        self.sidebar = Sidebar(self, on_navigate=self._navigate)
        self.sidebar.grid(row=1, column=0, sticky="ns")

        # Content area
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.grid(row=1, column=1, sticky="nsew", padx=20, pady=(15, 5))
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(0, weight=1)

        # Log panel (bottom)
        self._build_log_panel()

    def _build_profile_bar(self):
        bar = ctk.CTkFrame(self, height=40, fg_color=("white", "#1E293B"), corner_radius=0)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        bar.grid_propagate(False)

        inner = ctk.CTkFrame(bar, fg_color="transparent")
        inner.pack(fill="x", padx=12, pady=5)

        ctk.CTkLabel(inner, text="Profil:", font=ctk.CTkFont(size=10),
                     text_color=COLORS["text_muted"]).pack(side="left", padx=(0, 6))

        self.profile_combo = ctk.CTkComboBox(
            inner, values=self.config_mgr.list_profiles(),
            width=140, height=28, font=ctk.CTkFont(size=10),
            command=self._load_profile
        )
        self.profile_combo.set(self.config_mgr.current_profile or "default")
        self.profile_combo.pack(side="left", padx=(0, 8))

        self.dirty_label = ctk.CTkLabel(inner, text="", font=ctk.CTkFont(size=10),
                                         text_color=COLORS["dirty"])
        self.dirty_label.pack(side="left", padx=(0, 10))

        # Boutons profil
        ctk.CTkButton(inner, text="💾", width=28, height=28, fg_color="transparent",
                      border_width=1, command=self._save_profile,
                      font=ctk.CTkFont(size=12)).pack(side="left", padx=2)
        ctk.CTkButton(inner, text="📋", width=28, height=28, fg_color="transparent",
                      border_width=1, command=self._duplicate_profile,
                      font=ctk.CTkFont(size=12)).pack(side="left", padx=2)
        ctk.CTkButton(inner, text="🗑", width=28, height=28, fg_color="transparent",
                      border_width=1, command=self._delete_profile,
                      font=ctk.CTkFont(size=12)).pack(side="left", padx=2)

        # Gear icon (advanced settings)
        ctk.CTkButton(inner, text="⚙", width=28, height=28, fg_color="transparent",
                      border_width=1, command=self._open_advanced,
                      font=ctk.CTkFont(size=13)).pack(side="right")

    def _build_log_panel(self):
        """Panneau de log compact en bas de l'interface."""
        self.log_frame = ctk.CTkFrame(self, height=100, corner_radius=0)
        self.log_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.log_frame.pack_propagate(False)

        header = ctk.CTkFrame(self.log_frame, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(4, 0))
        ctk.CTkLabel(header, text="📋 Journal", font=ctk.CTkFont(size=10, weight="bold"),
                     text_color=COLORS["text_muted"]).pack(side="left")

        self.log_text = ctk.CTkTextbox(self.log_frame, height=70,
                                        font=ctk.CTkFont(family="Consolas", size=9))
        self.log_text.pack(fill="both", expand=True, padx=10, pady=(2, 5))
        self.log_text.configure(state="disabled")

    # --- Navigation ---

    def _navigate(self, page_key: str):
        if page_key == self._current_page:
            return
        self._current_page = page_key
        self.sidebar.set_active(page_key)
        self._show_page(page_key)

    def _show_page(self, key: str):
        for widget in self.content.winfo_children():
            widget.grid_forget()

        if key not in self._pages:
            self._pages[key] = self._create_page(key)

        page = self._pages[key]
        page.refresh(self.params)
        page.grid(row=0, column=0, sticky="nsew")

    def _create_page(self, key: str):
        if key == "modele":
            return PageModel(self.content, params=self.params,
                           on_change=self._on_param_change,
                           ollama_manager=self.ollama_manager)
        elif key == "analyse":
            page = PageAnalyze(self.content, params=self.params,
                             on_change=self._on_param_change)
            # Brancher le callback de lancement
            if hasattr(page, 'analyze_btn'):
                page.analyze_btn.configure(command=self._start_analysis)
            return page
        elif key == "upload":
            page = PageUpload(self.content, params=self.params,
                            on_change=self._on_param_change)
            if hasattr(page, 'upload_btn'):
                page.upload_btn.configure(command=self._start_upload)
            return page

        page_map = {
            "source": PageSource,
            "validation": PageValidation,
            "journal": PageJournal,
        }
        cls = page_map[key]
        return cls(self.content, params=self.params, on_change=self._on_param_change)

    # --- Param changes ---

    def _on_param_change(self, field: str, value):
        setattr(self.params, field, value)
        self._set_dirty(True)
        self._update_sidebar_status()

    def _set_dirty(self, dirty: bool):
        self._dirty = dirty
        self.dirty_label.configure(text="● Non sauvegardé" if dirty else "")

    def _update_sidebar_status(self):
        # Source
        if self.params.source_folder:
            p = Path(self.params.source_folder)
            if p.exists():
                self.sidebar.set_page_status("source", "done")
            else:
                self.sidebar.set_page_status("source", "active")
        else:
            self.sidebar.set_page_status("source", "pending")

        # Modèle
        if self.params.model_name and self.ollama_manager.is_serving():
            self.sidebar.set_page_status("modele", "done")
        elif self.params.model_name:
            self.sidebar.set_page_status("modele", "active")
        else:
            self.sidebar.set_page_status("modele", "pending")

        # Analyse
        if self._analysis_running:
            self.sidebar.set_page_status("analyse", "active")
        elif self._analysis_stats["success"] > 0:
            self.sidebar.set_page_status("analyse", "done")
        elif self.params.source_folder and self.params.model_name:
            self.sidebar.set_page_status("analyse", "pending")
        else:
            self.sidebar.set_page_status("analyse", "locked")

        # Validation
        if self._analysis_stats["success"] > 0:
            self.sidebar.set_page_status("validation", "pending")
        else:
            self.sidebar.set_page_status("validation", "locked")

        # Upload
        if self.params.ftps_username and self._analysis_stats["success"] > 0:
            self.sidebar.set_page_status("upload", "pending")
        else:
            self.sidebar.set_page_status("upload", "locked")

    # --- Intégration métier : Analyse ---

    def _start_analysis(self):
        """Lance l'analyse des images via ImageAnalyzer."""
        if self._analysis_running:
            self._stop_analysis()
            return

        if not self.params.source_folder:
            self.log("⚠ Sélectionnez un dossier source d'abord")
            return

        if not self.ollama_manager.is_serving():
            self.log("⚠ Ollama n'est pas démarré. Allez dans l'onglet Modèle IA.")
            return

        # Initialiser l'analyseur
        self.image_analyzer = ImageAnalyzer(
            model=self.params.model_name,
            max_workers=self.params.workers,
            cooldown_seconds=self.params.cooldown
        )
        self.image_analyzer.progress_callback = self._on_analysis_progress

        self._analysis_running = True
        self._analysis_stats = {"total": 0, "success": 0, "failed": 0, "invalid": 0}
        self._update_sidebar_status()

        # Mettre à jour le bouton
        if "analyse" in self._pages:
            page = self._pages["analyse"]
            if hasattr(page, 'analyze_btn'):
                page.analyze_btn.configure(text="⏹  Arrêter l'analyse")

        self.log(f"Démarrage de l'analyse — Dossier: {self.params.source_folder}")
        self.log(f"Modèle: {self.params.model_name} | Workers: {self.params.workers}")

        # Lancer en thread
        thread = threading.Thread(target=self._run_analysis, daemon=True)
        thread.start()

    def _run_analysis(self):
        """Thread d'analyse."""
        try:
            stats = self.image_analyzer.process_directory(
                self.params.source_folder,
                pre_filter=self.params.prefilter_enabled,
                resume=self.params.resume_mode
            )
            self._analysis_stats = stats
            self.after(0, lambda: self._on_analysis_complete(stats))
        except Exception as e:
            self.after(0, lambda: self._on_analysis_error(str(e)))

    def _on_analysis_progress(self, message: str):
        """Callback de progression depuis ImageAnalyzer."""
        self.after(0, lambda: self.log(message))
        # Mettre à jour la progress bar si disponible
        if "analyse" in self._pages:
            page = self._pages["analyse"]
            if hasattr(page, 'update_stats'):
                self.after(0, lambda: page.update_stats(self._analysis_stats))

    def _on_analysis_complete(self, stats: dict):
        """Fin de l'analyse."""
        self._analysis_running = False
        self._analysis_stats = stats
        self._update_sidebar_status()

        if "analyse" in self._pages:
            page = self._pages["analyse"]
            if hasattr(page, 'analyze_btn'):
                page.analyze_btn.configure(text="🔍  Lancer l'analyse")

        self.log(f"✔ Analyse terminée — {stats.get('success', 0)} succès, "
                 f"{stats.get('failed', 0)} échecs, {stats.get('invalid', 0)} invalides")

    def _on_analysis_error(self, error: str):
        """Erreur pendant l'analyse."""
        self._analysis_running = False
        self._update_sidebar_status()

        if "analyse" in self._pages:
            page = self._pages["analyse"]
            if hasattr(page, 'analyze_btn'):
                page.analyze_btn.configure(text="🔍  Lancer l'analyse")

        self.log(f"✖ Erreur d'analyse: {error}")

    def _stop_analysis(self):
        """Arrête l'analyse en cours."""
        if self.image_analyzer:
            self.image_analyzer._stop_requested = True
            self.log("Arrêt de l'analyse demandé...")

    # --- Intégration métier : Upload FTPS ---

    def _start_upload(self):
        """Lance l'upload FTPS."""
        if not self.params.ftps_username or not self.params.ftps_password:
            self.log("⚠ Renseignez vos identifiants FTPS d'abord")
            return

        self.log("Upload FTPS — Fonctionnalité à brancher sur le client FTPS existant")
        # TODO: Intégrer FTPSUploader depuis shutterstock_analyzer_unified.py

    # --- Profile management ---

    def _load_profile(self, name: str):
        self.params = self.config_mgr.load_profile(name)
        self._set_dirty(False)
        self._refresh_current_page()
        self._update_sidebar_status()

    def _save_profile(self):
        name = self.profile_combo.get()
        self.config_mgr.save_profile(name, self.params)
        self._set_dirty(False)
        self._refresh_profiles()

    def _duplicate_profile(self):
        from tkinter import simpledialog
        name = simpledialog.askstring("Dupliquer", "Nom du nouveau profil:", parent=self)
        if name:
            self.config_mgr.save_profile(name, self.params)
            self._refresh_profiles()
            self.profile_combo.set(name)

    def _delete_profile(self):
        name = self.profile_combo.get()
        if name == "default":
            return
        self.config_mgr.delete_profile(name)
        self._refresh_profiles()
        self._load_profile("default")
        self.profile_combo.set("default")

    def _refresh_profiles(self):
        self.profile_combo.configure(values=self.config_mgr.list_profiles())

    def _refresh_current_page(self):
        if self._current_page and self._current_page in self._pages:
            self._pages[self._current_page].refresh(self.params)

    # --- Advanced settings ---

    def _open_advanced(self):
        AdvancedSettingsWindow(self, self.params, self._on_param_change)

    # --- Logging ---

    def log(self, message: str):
        """Ajoute un message au log panel ET à la page journal."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}"

        # Log panel
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{formatted}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

        # Page journal
        if "journal" in self._pages:
            self._pages["journal"].add_log(formatted)

    # --- Fermeture ---

    def _on_close(self):
        if self._analysis_running:
            self._stop_analysis()

        if self._dirty:
            from tkinter import messagebox
            save = messagebox.askyesnocancel(
                "Profil modifié",
                "Le profil a été modifié. Enregistrer avant de quitter ?"
            )
            if save is None:
                return
            if save:
                self._save_profile()

        self.destroy()


def run():
    """Point d'entrée de l'application."""
    app = ShutterstockApp()
    app.mainloop()
