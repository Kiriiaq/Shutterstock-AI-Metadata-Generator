"""
AI Control Page - UI for managing Ollama/LLaMA integration
Provides controls for testing, starting, stopping, and monitoring AI
"""

import customtkinter as ctk
from tkinter import messagebox
import threading
import time
import logging
from typing import Optional, Callable, Dict, Any

from ...modules.ai.ollama_client import OllamaClient, OllamaStatus, ModelInfo
from ...modules.ai.vision_analyzer import VisionAnalyzer
from ..components.tooltips import add_tooltip

logger = logging.getLogger(__name__)


class StatusIndicator(ctk.CTkLabel):
    """Visual status indicator with colored dot"""

    COLORS = {
        "green": "#22c55e",
        "red": "#ef4444",
        "orange": "#f97316",
        "gray": "#6b7280"
    }

    def __init__(self, parent, size: int = 16, **kwargs):
        self._color = "gray"
        super().__init__(
            parent,
            text="●",
            font=ctk.CTkFont(size=size),
            text_color=self.COLORS[self._color],
            width=size + 4,
            **kwargs
        )

    def set_color(self, color: str):
        """Set indicator color: green, red, orange, gray"""
        if color in self.COLORS:
            self._color = color
            self.configure(text_color=self.COLORS[color])


class AIControlPage(ctk.CTkFrame):
    """
    AI Control page for Ollama/LLaMA management
    Provides:
    - Connection status and control
    - Model selection and management
    - Test functionality
    - Performance monitoring
    """

    def __init__(
        self,
        parent,
        settings: Dict[str, Any] = None,
        on_status_change: Callable[[OllamaStatus], None] = None,
        **kwargs
    ):
        super().__init__(parent, **kwargs)

        self.settings = settings or {}
        self.on_status_change = on_status_change

        # Initialize client
        self._init_client()

        # State
        self._checking = False
        self._testing = False

        # Create UI
        self._create_ui()

        # Initial check
        self.after(500, self._check_connection_async)

    def _init_client(self):
        """Initialize Ollama client"""
        url = self.settings.get("ollama_url", "http://localhost:11434")
        timeout = int(self.settings.get("ollama_timeout", 120))

        self.client = OllamaClient(
            base_url=url,
            timeout=timeout,
            on_status_change=self._handle_status_change
        )
        self.analyzer = VisionAnalyzer(
            client=self.client,
            model=self.settings.get("ollama_model")
        )

    def _create_ui(self):
        """Create the UI components"""
        self.grid_columnconfigure(0, weight=1)

        # Title
        title = ctk.CTkLabel(
            self,
            text="AI Control Panel",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title.grid(row=0, column=0, pady=(10, 20), sticky="w", padx=20)

        # Main container
        container = ctk.CTkFrame(self)
        container.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        container.grid_columnconfigure(0, weight=1)

        # Connection section
        self._create_connection_section(container)

        # Model section
        self._create_model_section(container)

        # Test section
        self._create_test_section(container)

        # Status section
        self._create_status_section(container)

    def _create_connection_section(self, parent):
        """Create connection controls"""
        section = ctk.CTkFrame(parent)
        section.pack(fill="x", padx=10, pady=10)

        # Header with indicator
        header = ctk.CTkFrame(section, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            header,
            text="Connection",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(side="left")

        self.connection_indicator = StatusIndicator(header)
        self.connection_indicator.pack(side="left", padx=10)

        self.connection_label = ctk.CTkLabel(
            header,
            text="Unknown",
            text_color="gray"
        )
        self.connection_label.pack(side="left")

        # URL display
        url_frame = ctk.CTkFrame(section, fg_color="transparent")
        url_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(url_frame, text="Server URL:", width=100).pack(side="left")

        self.url_entry = ctk.CTkEntry(url_frame, width=300)
        self.url_entry.insert(0, self.settings.get("ollama_url", "http://localhost:11434"))
        self.url_entry.pack(side="left", padx=5)
        add_tooltip(self.url_entry, "ollama_url")

        # Control buttons
        btn_frame = ctk.CTkFrame(section, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=10)

        self.check_btn = ctk.CTkButton(
            btn_frame,
            text="Check Connection",
            width=140,
            command=self._check_connection_async
        )
        self.check_btn.pack(side="left", padx=5)
        add_tooltip(self.check_btn, "Test connection to Ollama server")

        self.refresh_btn = ctk.CTkButton(
            btn_frame,
            text="Refresh Models",
            width=140,
            command=self._refresh_models
        )
        self.refresh_btn.pack(side="left", padx=5)
        add_tooltip(self.refresh_btn, "Refresh list of available models")

    def _create_model_section(self, parent):
        """Create model selection controls"""
        section = ctk.CTkFrame(parent)
        section.pack(fill="x", padx=10, pady=10)

        # Header
        header = ctk.CTkFrame(section, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            header,
            text="Vision Model",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(side="left")

        self.model_indicator = StatusIndicator(header)
        self.model_indicator.pack(side="left", padx=10)

        # Model selector
        model_frame = ctk.CTkFrame(section, fg_color="transparent")
        model_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(model_frame, text="Select Model:", width=100).pack(side="left")

        self.model_combo = ctk.CTkComboBox(
            model_frame,
            values=["Loading..."],
            width=300,
            command=self._on_model_selected
        )
        self.model_combo.pack(side="left", padx=5)
        add_tooltip(self.model_combo, "ollama_model")

        # Model info
        self.model_info_label = ctk.CTkLabel(
            section,
            text="",
            text_color="gray"
        )
        self.model_info_label.pack(padx=10, pady=5, anchor="w")

        # Load/Unload buttons
        btn_frame = ctk.CTkFrame(section, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=10)

        self.load_btn = ctk.CTkButton(
            btn_frame,
            text="Load Model",
            width=120,
            fg_color="green",
            command=self._load_model
        )
        self.load_btn.pack(side="left", padx=5)
        add_tooltip(self.load_btn, "Load selected model into memory")

        self.unload_btn = ctk.CTkButton(
            btn_frame,
            text="Unload Model",
            width=120,
            fg_color="gray",
            command=self._unload_model
        )
        self.unload_btn.pack(side="left", padx=5)
        add_tooltip(self.unload_btn, "Unload model from memory to free resources")

    def _create_test_section(self, parent):
        """Create test controls"""
        section = ctk.CTkFrame(parent)
        section.pack(fill="x", padx=10, pady=10)

        # Header
        ctk.CTkLabel(
            section,
            text="Test AI",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(padx=10, pady=5, anchor="w")

        # Test button
        btn_frame = ctk.CTkFrame(section, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=5)

        self.test_btn = ctk.CTkButton(
            btn_frame,
            text="Test AI Response",
            width=160,
            fg_color="blue",
            command=self._test_ai
        )
        self.test_btn.pack(side="left", padx=5)
        add_tooltip(self.test_btn, "Send a test prompt to verify AI is working")

        self.test_status = ctk.CTkLabel(
            btn_frame,
            text="",
            text_color="gray"
        )
        self.test_status.pack(side="left", padx=10)

        # Test result
        self.test_result = ctk.CTkTextbox(section, height=80)
        self.test_result.pack(fill="x", padx=10, pady=5)
        self.test_result.insert("1.0", "Click 'Test AI Response' to verify the AI is working...")
        self.test_result.configure(state="disabled")

    def _create_status_section(self, parent):
        """Create status display"""
        section = ctk.CTkFrame(parent)
        section.pack(fill="x", padx=10, pady=10)

        # Header
        ctk.CTkLabel(
            section,
            text="Status Information",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(padx=10, pady=5, anchor="w")

        # Status grid
        self.status_text = ctk.CTkTextbox(section, height=120)
        self.status_text.pack(fill="x", padx=10, pady=5)
        self.status_text.configure(state="disabled")

        self._update_status_display()

    # ==================== Event Handlers ====================

    def _handle_status_change(self, status: OllamaStatus):
        """Handle status change from client"""
        # Update UI on main thread
        self.after(0, lambda: self._update_ui_for_status(status))

        if self.on_status_change:
            self.on_status_change(status)

    def _update_ui_for_status(self, status: OllamaStatus):
        """Update UI elements based on status"""
        status_map = {
            OllamaStatus.ONLINE: ("green", "Online"),
            OllamaStatus.OFFLINE: ("red", "Offline"),
            OllamaStatus.BUSY: ("orange", "Busy"),
            OllamaStatus.ERROR: ("red", "Error"),
            OllamaStatus.UNKNOWN: ("gray", "Unknown")
        }

        color, text = status_map.get(status, ("gray", "Unknown"))

        self.connection_indicator.set_color(color)
        self.connection_label.configure(text=text)

        # Enable/disable controls
        is_online = status == OllamaStatus.ONLINE
        self.model_combo.configure(state="normal" if is_online else "disabled")
        self.load_btn.configure(state="normal" if is_online else "disabled")
        self.test_btn.configure(state="normal" if is_online else "disabled")

        self._update_status_display()

    def _check_connection_async(self):
        """Check connection in background"""
        if self._checking:
            return

        self._checking = True
        self.check_btn.configure(state="disabled", text="Checking...")

        def check():
            # Update URL from entry
            new_url = self.url_entry.get().strip()
            if new_url != self.client.base_url:
                self.client.base_url = new_url

            connected = self.client.check_connection()

            # Update models if connected
            if connected:
                self.client.list_models(refresh=True)

            self.after(0, self._on_check_complete)

        threading.Thread(target=check, daemon=True).start()

    def _on_check_complete(self):
        """Called when connection check completes"""
        self._checking = False
        self.check_btn.configure(state="normal", text="Check Connection")

        # Update model list
        self._update_model_list()
        self._update_status_display()

    def _refresh_models(self):
        """Refresh model list"""
        if self.client.status != OllamaStatus.ONLINE:
            messagebox.showwarning("Offline", "Cannot refresh - server is offline")
            return

        self.refresh_btn.configure(state="disabled", text="Refreshing...")

        def refresh():
            self.client.list_models(refresh=True)
            self.after(0, lambda: self._on_refresh_complete())

        threading.Thread(target=refresh, daemon=True).start()

    def _on_refresh_complete(self):
        """Called when model refresh completes"""
        self.refresh_btn.configure(state="normal", text="Refresh Models")
        self._update_model_list()

    def _update_model_list(self):
        """Update model dropdown with available models"""
        models = self.client.list_vision_models()

        if models:
            model_names = [m.name for m in models]
            self.model_combo.configure(values=model_names)

            # Select current or first model
            current = self.settings.get("ollama_model", "")
            if current in model_names:
                self.model_combo.set(current)
            else:
                self.model_combo.set(model_names[0])

            self.model_indicator.set_color("green")
            self._update_model_info()
        else:
            self.model_combo.configure(values=["No vision models found"])
            self.model_combo.set("No vision models found")
            self.model_indicator.set_color("red")
            self.model_info_label.configure(text="No vision-capable models available")

    def _on_model_selected(self, model_name: str):
        """Handle model selection"""
        self._update_model_info()

    def _update_model_info(self):
        """Update model info display"""
        model_name = self.model_combo.get()
        model = self.client.get_model_info(model_name)

        if model:
            info = f"Size: {model.size_gb:.1f} GB | Vision: {'Yes' if model.is_vision else 'No'}"
            self.model_info_label.configure(text=info)
        else:
            self.model_info_label.configure(text="")

    def _load_model(self):
        """Load selected model"""
        model_name = self.model_combo.get()
        if not model_name or model_name == "No vision models found":
            return

        self.load_btn.configure(state="disabled", text="Loading...")
        self.model_indicator.set_color("orange")

        def load():
            success = self.client.load_model(model_name)
            self.after(0, lambda: self._on_model_loaded(success, model_name))

        threading.Thread(target=load, daemon=True).start()

    def _on_model_loaded(self, success: bool, model_name: str):
        """Called when model loading completes"""
        self.load_btn.configure(state="normal", text="Load Model")

        if success:
            self.model_indicator.set_color("green")
            self.model_info_label.configure(text=f"Model loaded: {model_name}")
            self.analyzer.model = model_name
        else:
            self.model_indicator.set_color("red")
            self.model_info_label.configure(text="Failed to load model")

        self._update_status_display()

    def _unload_model(self):
        """Unload current model"""
        if not self.client.current_model:
            return

        self.unload_btn.configure(state="disabled", text="Unloading...")

        def unload():
            self.client.unload_model()
            self.after(0, self._on_model_unloaded)

        threading.Thread(target=unload, daemon=True).start()

    def _on_model_unloaded(self):
        """Called when model unloading completes"""
        self.unload_btn.configure(state="normal", text="Unload Model")
        self.model_indicator.set_color("gray")
        self._update_status_display()

    def _test_ai(self):
        """Run AI test"""
        if self._testing:
            return

        self._testing = True
        self.test_btn.configure(state="disabled", text="Testing...")
        self.test_status.configure(text="Running test...", text_color="orange")

        self.test_result.configure(state="normal")
        self.test_result.delete("1.0", "end")
        self.test_result.insert("1.0", "Testing AI connection...")
        self.test_result.configure(state="disabled")

        def test():
            result = self.client.test_connection()
            self.after(0, lambda: self._on_test_complete(result))

        threading.Thread(target=test, daemon=True).start()

    def _on_test_complete(self, result: Dict):
        """Called when test completes"""
        self._testing = False
        self.test_btn.configure(state="normal", text="Test AI Response")

        self.test_result.configure(state="normal")
        self.test_result.delete("1.0", "end")

        if result["success"]:
            self.test_status.configure(text="Test passed!", text_color="green")
            text = f"Model: {result['model']}\n"
            text += f"Response: {result['message']}\n"
            text += f"Response time: {result['response_time_ms']} ms"
        else:
            self.test_status.configure(text="Test failed", text_color="red")
            text = f"Error: {result['message']}"

        self.test_result.insert("1.0", text)
        self.test_result.configure(state="disabled")

        self._update_status_display()

    def _update_status_display(self):
        """Update status text display"""
        info = self.client.get_status_info()

        lines = [
            f"Status: {info['status'].upper()}",
            f"Server: {info['url']}",
            f"Version: {info.get('version', 'Unknown')}",
            f"Current Model: {info['current_model'] or 'None loaded'}",
            f"Vision Models: {len(self.client.list_vision_models())} available",
        ]

        self.status_text.configure(state="normal")
        self.status_text.delete("1.0", "end")
        self.status_text.insert("1.0", "\n".join(lines))
        self.status_text.configure(state="disabled")

    # ==================== Public API ====================

    def get_client(self) -> OllamaClient:
        """Get the Ollama client"""
        return self.client

    def get_analyzer(self) -> VisionAnalyzer:
        """Get the vision analyzer"""
        return self.analyzer

    def is_ready(self) -> bool:
        """Check if AI is ready for analysis"""
        return (
            self.client.status == OllamaStatus.ONLINE and
            self.analyzer.model is not None
        )

    def get_selected_model(self) -> str:
        """Get currently selected model"""
        return self.model_combo.get()
