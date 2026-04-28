#!/usr/bin/env python3
"""
Splash Screen Module - Loading screen with progress bar
"""

import tkinter as tk
from tkinter import ttk
import threading
import time


class SplashScreen:
    """Modern splash screen with progress bar."""

    def __init__(self, title: str = "Loading...", version: str = "1.0.0",
                 width: int = 400, height: int = 200):
        self.title = title
        self.version = version
        self.width = width
        self.height = height
        self.root = None
        self.progress_var = None
        self.status_var = None
        self.progress_bar = None
        self._closed = False

    def show(self):
        """Display the splash screen."""
        self.root = tk.Tk()
        self.root.overrideredirect(True)  # Remove window decorations

        # Center on screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - self.width) // 2
        y = (screen_height - self.height) // 2
        self.root.geometry(f"{self.width}x{self.height}+{x}+{y}")

        # Style
        self.root.configure(bg="#1a1a2e")

        # Main frame
        frame = tk.Frame(self.root, bg="#1a1a2e")
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Title
        title_label = tk.Label(
            frame,
            text=self.title,
            font=("Segoe UI", 18, "bold"),
            fg="#ffffff",
            bg="#1a1a2e"
        )
        title_label.pack(pady=(10, 5))

        # Version
        version_label = tk.Label(
            frame,
            text=f"Version {self.version}",
            font=("Segoe UI", 10),
            fg="#888888",
            bg="#1a1a2e"
        )
        version_label.pack()

        # Progress bar style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(
            "Custom.Horizontal.TProgressbar",
            troughcolor='#2d2d44',
            background='#4a90d9',
            darkcolor='#4a90d9',
            lightcolor='#4a90d9',
            bordercolor='#1a1a2e'
        )

        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            frame,
            variable=self.progress_var,
            maximum=100,
            length=self.width - 60,
            mode='determinate',
            style="Custom.Horizontal.TProgressbar"
        )
        self.progress_bar.pack(pady=(30, 10))

        # Status label
        self.status_var = tk.StringVar(value="Initializing...")
        status_label = tk.Label(
            frame,
            textvariable=self.status_var,
            font=("Segoe UI", 9),
            fg="#aaaaaa",
            bg="#1a1a2e"
        )
        status_label.pack()

        # Keep on top
        self.root.attributes('-topmost', True)
        self.root.update()

    def update_progress(self, value: float, status: str = None):
        """Update progress bar and status text."""
        if self._closed or not self.root:
            return

        try:
            self.progress_var.set(value)
            if status:
                self.status_var.set(status)
            self.root.update()
        except tk.TclError:
            self._closed = True

    def close(self):
        """Close the splash screen."""
        if self._closed or not self.root:
            return

        self._closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass
        self.root = None


def run_with_splash(app_name: str, version: str, load_steps: list,
                    launch_callback, error_callback=None):
    """
    Run application with splash screen.

    Args:
        app_name: Application name for splash screen
        version: Version string
        load_steps: List of (status_text, load_function) tuples
        launch_callback: Function to call after loading (receives splash to close)
        error_callback: Function to call on error (receives exception)
    """
    splash = SplashScreen(app_name, version)
    splash.show()

    def load():
        try:
            total_steps = len(load_steps)
            for i, (status, func) in enumerate(load_steps):
                progress = (i / total_steps) * 100
                splash.update_progress(progress, status)

                if func:
                    func()

                time.sleep(0.1)  # Small delay for visual feedback

            splash.update_progress(100, "Starting application...")
            time.sleep(0.3)

            # Launch main app
            splash.root.after(100, lambda: launch_callback(splash))

        except Exception as e:
            splash.close()
            if error_callback:
                error_callback(e)
            else:
                raise

    # Start loading in thread
    thread = threading.Thread(target=load, daemon=True)
    thread.start()

    # Run splash mainloop
    splash.root.mainloop()
