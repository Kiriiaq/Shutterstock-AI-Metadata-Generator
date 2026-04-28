"""
Shutterstock AI Metadata Generator v2.0
Main entry point with debug console
"""

import sys
import logging
from pathlib import Path


def resource_path(relative_path: str) -> Path:
    """Resolve a path that works in both source tree and PyInstaller bundle."""
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
    else:
        base = Path(__file__).parent
    return base / relative_path


# Configuration Windows pour l'icone dans la barre des taches
if sys.platform == "win32":
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "ShutterstockAnalyzer.v2.0"
        )
    except Exception:
        pass

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging for debug
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("ShutterstockAI")

def main():
    """Main application entry point with splash screen"""
    import time
    import threading
    from src.utils.splash_screen import SplashScreen

    # Show splash screen with progress bar
    splash = SplashScreen("Shutterstock AI", "2.0.0", width=450, height=220)
    splash.show()

    ctk = None
    modules_loaded = False
    error = None

    def load_modules():
        nonlocal ctk, modules_loaded, error
        try:
            splash.update_progress(10, "Loading CustomTkinter...")
            import customtkinter
            ctk = customtkinter
            time.sleep(0.1)

            splash.update_progress(30, "Loading modules...")
            from src.modules.integration import ShutterstockAIv2
            from src.ui.pages.settings_page import SettingsPage
            from src.ui.pages.audit_page import AuditPage
            from src.ui.pages.write_page import WritePage
            from src.ui.pages.ai_control_page import AIControlPage
            from src.ui.pages.scan_page import ScanPage
            time.sleep(0.1)

            splash.update_progress(50, "Modules loaded...")
            modules_loaded = True

        except ImportError as e:
            error = e

    # Load modules in thread
    load_thread = threading.Thread(target=load_modules, daemon=True)
    load_thread.start()

    # Wait for module loading
    def check_modules():
        if load_thread.is_alive():
            splash.root.after(50, check_modules)
        else:
            splash.close()

    splash.root.after(100, check_modules)
    splash.root.mainloop()

    if error:
        logger.error(f"Import error: {error}")
        logger.error("Please install required packages:")
        logger.error("  pip install customtkinter pillow requests")
        input("\nPress Enter to exit...")
        sys.exit(1)

    if not modules_loaded:
        logger.error("Modules failed to load")
        sys.exit(1)

    logger.info("=" * 50)
    logger.info("Shutterstock AI Metadata Generator v2.0")
    logger.info("=" * 50)
    logger.info("Modules imported successfully")

    # Initialize main application
    logger.info("Initializing application...")

    class App(ctk.CTk):
        def __init__(self):
            super().__init__()

            self.title("ShutterstockAnalyzer v2.0.0 - AI Metadata Generator for Stock Photography")
            self.geometry("1400x900")
            self.minsize(1200, 800)

            # Window icon (top-left corner). Uses resource_path so it works
            # in both source tree and PyInstaller --onefile bundle.
            icon_path = resource_path("assets/icons/icone.ico")
            if icon_path.exists():
                try:
                    self.iconbitmap(str(icon_path))
                except Exception as e:
                    logger.warning(f"Could not set window icon: {e}")

            # Set theme
            ctk.set_appearance_mode("dark")
            ctk.set_default_color_theme("blue")

            # Initialize v2 API
            logger.info("Initializing ShutterstockAI v2.0 API...")
            try:
                self.api = ShutterstockAIv2()
                logger.info(f"ExifTool available: {self.api.exiftool_available}")
            except Exception as e:
                logger.warning(f"API initialization warning: {e}")
                self.api = None

            # Store references to pages
            self.ai_control_page = None
            self.scan_page = None

            # Create UI
            self._create_ui()

            logger.info("Application ready!")

        def _create_ui(self):
            """Create main UI"""
            # Configure grid
            self.grid_columnconfigure(0, weight=1)
            self.grid_rowconfigure(1, weight=1)

            # Header
            header = ctk.CTkFrame(self, height=60)
            header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
            header.grid_propagate(False)

            ctk.CTkLabel(
                header,
                text="Shutterstock AI Metadata Generator v2.0",
                font=ctk.CTkFont(size=20, weight="bold")
            ).pack(side="left", padx=20, pady=10)

            # Status indicators
            status_frame = ctk.CTkFrame(header, fg_color="transparent")
            status_frame.pack(side="right", padx=20)

            # ExifTool status
            exif_text = "ExifTool: OK" if self.api and self.api.exiftool_available else "ExifTool: NOT FOUND"
            exif_color = "#22c55e" if self.api and self.api.exiftool_available else "#ef4444"

            ctk.CTkLabel(
                status_frame,
                text="●",
                text_color=exif_color,
                font=ctk.CTkFont(size=14)
            ).pack(side="left", padx=2)

            ctk.CTkLabel(
                status_frame,
                text=exif_text,
                text_color=exif_color
            ).pack(side="left", padx=(0, 15))

            # AI status indicator (will be updated by AI control page)
            self.ai_indicator = ctk.CTkLabel(
                status_frame,
                text="●",
                text_color="#6b7280",
                font=ctk.CTkFont(size=14)
            )
            self.ai_indicator.pack(side="left", padx=2)

            self.ai_status_label = ctk.CTkLabel(
                status_frame,
                text="AI: Not checked",
                text_color="#6b7280"
            )
            self.ai_status_label.pack(side="left")

            # Tab view
            self.tabview = ctk.CTkTabview(self)
            self.tabview.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

            # Add tabs in order
            self.tabview.add("AI Control")
            self.tabview.add("Scan Images")
            self.tabview.add("AI Process")
            self.tabview.add("Metadata Editor")
            self.tabview.add("Audit Log")
            self.tabview.add("Settings")

            # Configure tabs
            for tab_name in ["AI Control", "Scan Images", "AI Process", "Metadata Editor", "Audit Log", "Settings"]:
                self.tabview.tab(tab_name).grid_columnconfigure(0, weight=1)
                self.tabview.tab(tab_name).grid_rowconfigure(0, weight=1)

            # Populate tabs
            self._create_ai_control_tab()
            self._create_scan_tab()
            self._create_process_tab()
            self._create_editor_tab()
            self._create_audit_tab()
            self._create_settings_tab()

            # Footer / Status bar
            footer = ctk.CTkFrame(self, height=30)
            footer.grid(row=2, column=0, sticky="ew", padx=10, pady=(5, 10))
            footer.grid_propagate(False)

            self.status_label = ctk.CTkLabel(
                footer,
                text="Ready",
                text_color="gray"
            )
            self.status_label.pack(side="left", padx=10)

            self.progress_label = ctk.CTkLabel(
                footer,
                text="",
                text_color="gray"
            )
            self.progress_label.pack(side="right", padx=10)

        def _create_ai_control_tab(self):
            """Create AI control tab"""
            tab = self.tabview.tab("AI Control")

            settings = {}
            if self.api:
                settings = self.api.database.get_all_settings()

            try:
                self.ai_control_page = AIControlPage(
                    tab,
                    settings=settings,
                    on_status_change=self._on_ai_status_change
                )
                self.ai_control_page.grid(row=0, column=0, sticky="nsew")
            except Exception as e:
                logger.error(f"Failed to create AIControlPage: {e}")
                ctk.CTkLabel(tab, text=f"Error loading AI Control: {e}").pack(pady=20)

        def _create_scan_tab(self):
            """Create scan images tab"""
            tab = self.tabview.tab("Scan Images")

            try:
                reader = self.api.metadata_reader if self.api else None
                self.scan_page = ScanPage(
                    tab,
                    metadata_reader=reader,
                    on_images_selected=self._on_images_selected,
                    on_process_requested=self._on_process_requested
                )
                self.scan_page.grid(row=0, column=0, sticky="nsew")
            except Exception as e:
                logger.error(f"Failed to create ScanPage: {e}")
                ctk.CTkLabel(tab, text=f"Error loading Scan: {e}").pack(pady=20)

        def _create_process_tab(self):
            """Create AI processing tab"""
            tab = self.tabview.tab("AI Process")

            frame = ctk.CTkFrame(tab)
            frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
            frame.grid_columnconfigure(0, weight=1)
            frame.grid_rowconfigure(2, weight=1)

            # Header
            ctk.CTkLabel(
                frame,
                text="AI Image Processing",
                font=ctk.CTkFont(size=18, weight="bold")
            ).pack(pady=10)

            # Options frame
            options_frame = ctk.CTkFrame(frame)
            options_frame.pack(fill="x", padx=20, pady=10)

            # Options
            self.skip_existing_var = ctk.BooleanVar(value=True)
            ctk.CTkCheckBox(
                options_frame,
                text="Skip images with existing metadata",
                variable=self.skip_existing_var
            ).pack(side="left", padx=10)

            self.write_results_var = ctk.BooleanVar(value=False)
            ctk.CTkCheckBox(
                options_frame,
                text="Write AI results to files",
                variable=self.write_results_var
            ).pack(side="left", padx=10)

            # Selected count
            self.selected_count_label = ctk.CTkLabel(
                options_frame,
                text="0 images selected",
                text_color="gray"
            )
            self.selected_count_label.pack(side="right", padx=10)

            # Control buttons
            btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
            btn_frame.pack(fill="x", padx=20, pady=10)

            self.start_process_btn = ctk.CTkButton(
                btn_frame,
                text="Start AI Processing",
                width=180,
                height=40,
                fg_color="green",
                font=ctk.CTkFont(size=14, weight="bold"),
                command=self._start_ai_processing
            )
            self.start_process_btn.pack(side="left", padx=10)

            self.stop_process_btn = ctk.CTkButton(
                btn_frame,
                text="Stop",
                width=80,
                height=40,
                fg_color="red",
                state="disabled",
                command=self._stop_ai_processing
            )
            self.stop_process_btn.pack(side="left", padx=10)

            # Progress
            self.process_progress = ctk.CTkProgressBar(frame, width=600)
            self.process_progress.pack(pady=10)
            self.process_progress.set(0)

            self.process_status = ctk.CTkLabel(
                frame,
                text="Select images in Scan tab, then click Start",
                text_color="gray"
            )
            self.process_status.pack(pady=5)

            # Results area
            self.process_results = ctk.CTkTextbox(frame, height=400)
            self.process_results.pack(fill="both", expand=True, padx=20, pady=10)
            self.process_results.insert("1.0", "AI processing results will appear here...\n")

        def _create_editor_tab(self):
            """Create metadata editor tab"""
            tab = self.tabview.tab("Metadata Editor")

            if self.api:
                try:
                    write_page = WritePage(
                        tab,
                        database=self.api.database,
                        metadata_reader=self.api.metadata_reader,
                        metadata_writer=self.api.metadata_writer,
                    )
                    write_page.grid(row=0, column=0, sticky="nsew")
                except Exception as e:
                    logger.error(f"Failed to create WritePage: {e}")
                    ctk.CTkLabel(tab, text=f"Error loading editor: {e}").pack(pady=20)
            else:
                ctk.CTkLabel(tab, text="API not available").pack(pady=20)

        def _create_audit_tab(self):
            """Create audit log tab"""
            tab = self.tabview.tab("Audit Log")

            if self.api:
                try:
                    audit_page = AuditPage(tab, self.api.database)
                    audit_page.grid(row=0, column=0, sticky="nsew")
                except Exception as e:
                    logger.error(f"Failed to create AuditPage: {e}")
                    ctk.CTkLabel(tab, text=f"Error loading audit: {e}").pack(pady=20)
            else:
                ctk.CTkLabel(tab, text="API not available").pack(pady=20)

        def _create_settings_tab(self):
            """Create settings tab"""
            tab = self.tabview.tab("Settings")

            if self.api:
                try:
                    settings_page = SettingsPage(tab, self.api.database)
                    settings_page.grid(row=0, column=0, sticky="nsew")
                except Exception as e:
                    logger.error(f"Failed to create SettingsPage: {e}")
                    ctk.CTkLabel(tab, text=f"Error loading settings: {e}").pack(pady=20)
            else:
                ctk.CTkLabel(tab, text="API not available").pack(pady=20)

        # ==================== Callbacks ====================

        def _on_ai_status_change(self, status):
            """Handle AI status change"""
            from src.modules.ai.ollama_client import OllamaStatus

            status_map = {
                OllamaStatus.ONLINE: ("#22c55e", "AI: Online"),
                OllamaStatus.OFFLINE: ("#ef4444", "AI: Offline"),
                OllamaStatus.BUSY: ("#f97316", "AI: Busy"),
                OllamaStatus.ERROR: ("#ef4444", "AI: Error"),
                OllamaStatus.UNKNOWN: ("#6b7280", "AI: Unknown")
            }

            color, text = status_map.get(status, ("#6b7280", "AI: Unknown"))

            self.ai_indicator.configure(text_color=color)
            self.ai_status_label.configure(text=text, text_color=color)

        def _on_images_selected(self, paths):
            """Handle image selection change"""
            count = len(paths)
            self.selected_count_label.configure(text=f"{count} images selected")
            self._selected_images = paths

        def _on_process_requested(self, paths):
            """Handle process request from scan page"""
            self._selected_images = paths
            self.selected_count_label.configure(text=f"{len(paths)} images selected")
            self.tabview.set("AI Process")

        def _start_ai_processing(self):
            """Start AI processing"""
            if not hasattr(self, '_selected_images') or not self._selected_images:
                from tkinter import messagebox
                messagebox.showwarning("Warning", "No images selected. Use the Scan tab to select images first.")
                return

            if not self.ai_control_page or not self.ai_control_page.is_ready():
                from tkinter import messagebox
                messagebox.showwarning("Warning", "AI is not ready. Check the AI Control tab.")
                return

            self.start_process_btn.configure(state="disabled")
            self.stop_process_btn.configure(state="normal")
            self.process_results.delete("1.0", "end")
            self.process_progress.set(0)

            # Initialize AI in API if not done
            if self.api and not hasattr(self.api, 'vision_analyzer'):
                self.api.init_ai()

            import threading

            def process():
                try:
                    def on_progress(completed, total, current):
                        progress = completed / total if total > 0 else 0
                        self.after(0, lambda: self._update_process_progress(progress, completed, total, current))

                    def on_result(result):
                        self.after(0, lambda r=result: self._add_process_result(r))

                    result = self.api.analyze_batch_ai(
                        self._selected_images,
                        skip_if_has_metadata=self.skip_existing_var.get(),
                        write_metadata=self.write_results_var.get(),
                        on_progress=on_progress,
                        on_result=on_result
                    )

                    self.after(0, lambda: self._on_process_complete(result))

                except Exception as e:
                    logger.error(f"Processing error: {e}")
                    self.after(0, lambda: self._on_process_error(str(e)))

            threading.Thread(target=process, daemon=True).start()

        def _stop_ai_processing(self):
            """Stop AI processing"""
            if self.api and hasattr(self.api, 'vision_analyzer'):
                self.api.vision_analyzer.cancel()
            self.process_status.configure(text="Stopping...", text_color="orange")

        def _update_process_progress(self, progress, completed, total, current):
            """Update processing progress"""
            self.process_progress.set(progress)
            self.process_status.configure(
                text=f"Processing: {completed}/{total} - {current}",
                text_color="orange"
            )

        def _add_process_result(self, result):
            """Add result to display"""
            file_name = Path(result.get("file_path", "")).name
            status = "✓" if result.get("success") else "✗"

            if result.get("success"):
                title = result.get("title", "")[:50]
                kw_count = len(result.get("keywords", []))
                line = f"{status} {file_name} - {title}... ({kw_count} keywords)\n"
            else:
                error = result.get("error", "Unknown error")
                line = f"{status} {file_name} - ERROR: {error}\n"

            self.process_results.insert("end", line)
            self.process_results.see("end")

        def _on_process_complete(self, result):
            """Handle processing complete"""
            self.start_process_btn.configure(state="normal")
            self.stop_process_btn.configure(state="disabled")
            self.process_progress.set(1)

            summary = (
                f"\n{'='*50}\n"
                f"COMPLETE: {result['completed']}/{result['total']} successful\n"
                f"Failed: {result['failed']} | Skipped: {result['skipped']}\n"
                f"Duration: {result['duration_ms']/1000:.1f}s\n"
                f"Success rate: {result['success_rate']:.1f}%\n"
                f"{'='*50}\n"
            )

            self.process_results.insert("end", summary)
            self.process_status.configure(text="Complete!", text_color="green")
            self.status_label.configure(text=f"Processed {result['completed']} images")

        def _on_process_error(self, error):
            """Handle processing error"""
            self.start_process_btn.configure(state="normal")
            self.stop_process_btn.configure(state="disabled")
            self.process_status.configure(text=f"Error: {error}", text_color="red")
            self.process_results.insert("end", f"\n\nERROR: {error}\n")

        def on_closing(self):
            """Handle window close"""
            logger.info("Closing application...")
            if self.api:
                self.api.close()
            self.destroy()

    # Run application
    logger.info("Starting GUI...")
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()

    logger.info("Application closed")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        input("\nPress Enter to exit...")
        sys.exit(1)
