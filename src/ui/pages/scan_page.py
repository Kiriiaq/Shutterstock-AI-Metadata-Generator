"""
Scan Page - UI for scanning folders and selecting images for processing
Provides folder selection, recursive scanning, image preview, and batch selection
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Set
from dataclasses import dataclass

from ...modules.workers.worker_pool import collect_image_files
from ...modules.engines.metadata_reader import MetadataReader
from ..components.tooltips import add_tooltip

logger = logging.getLogger(__name__)


@dataclass
class ImageItem:
    """Represents an image in the scan list"""
    path: Path
    size_bytes: int = 0
    width: int = 0
    height: int = 0
    has_metadata: bool = False
    selected: bool = True

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    @property
    def resolution_mp(self) -> float:
        if self.width and self.height:
            return (self.width * self.height) / 1_000_000
        return 0

    @property
    def dimensions_str(self) -> str:
        if self.width and self.height:
            return f"{self.width}x{self.height}"
        return "Unknown"


class ImageListItem(ctk.CTkFrame):
    """Single image item in the list"""

    def __init__(
        self,
        parent,
        image_item: ImageItem,
        on_select: Callable[[Path, bool], None] = None,
        on_preview: Callable[[Path], None] = None,
        **kwargs
    ):
        super().__init__(parent, height=40, **kwargs)

        self.image_item = image_item
        self.on_select = on_select
        self.on_preview = on_preview

        self.grid_columnconfigure(1, weight=1)

        # Checkbox
        self.selected_var = ctk.BooleanVar(value=image_item.selected)
        self.checkbox = ctk.CTkCheckBox(
            self,
            text="",
            variable=self.selected_var,
            width=24,
            command=self._on_checkbox_changed
        )
        self.checkbox.grid(row=0, column=0, padx=5)

        # Filename
        self.name_label = ctk.CTkLabel(
            self,
            text=image_item.path.name,
            anchor="w"
        )
        self.name_label.grid(row=0, column=1, sticky="w", padx=5)
        self.name_label.bind("<Button-1>", self._on_click)

        # Size
        size_text = f"{image_item.size_mb:.1f} MB"
        self.size_label = ctk.CTkLabel(self, text=size_text, width=70)
        self.size_label.grid(row=0, column=2, padx=5)

        # Dimensions
        self.dims_label = ctk.CTkLabel(
            self,
            text=image_item.dimensions_str,
            width=100
        )
        self.dims_label.grid(row=0, column=3, padx=5)

        # Metadata indicator
        meta_text = "Has metadata" if image_item.has_metadata else "No metadata"
        meta_color = "green" if image_item.has_metadata else "gray"
        self.meta_label = ctk.CTkLabel(
            self,
            text=meta_text,
            text_color=meta_color,
            width=100
        )
        self.meta_label.grid(row=0, column=4, padx=5)

    def _on_checkbox_changed(self):
        self.image_item.selected = self.selected_var.get()
        if self.on_select:
            self.on_select(self.image_item.path, self.image_item.selected)

    def _on_click(self, event):
        if self.on_preview:
            self.on_preview(self.image_item.path)

    def set_selected(self, selected: bool):
        self.selected_var.set(selected)
        self.image_item.selected = selected


class ScanPage(ctk.CTkFrame):
    """
    Scan page for folder scanning and image selection
    """

    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.webp'}

    def __init__(
        self,
        parent,
        metadata_reader: MetadataReader = None,
        on_images_selected: Callable[[List[Path]], None] = None,
        on_process_requested: Callable[[List[Path]], None] = None,
        **kwargs
    ):
        super().__init__(parent, **kwargs)

        self.metadata_reader = metadata_reader
        self.on_images_selected = on_images_selected
        self.on_process_requested = on_process_requested

        # State
        self._images: List[ImageItem] = []
        self._scanning = False
        self._current_folder: Optional[Path] = None

        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # Create UI
        self._create_ui()

    def _create_ui(self):
        """Create the UI components"""
        # Folder selection section
        self._create_folder_section()

        # Options section
        self._create_options_section()

        # Image list section
        self._create_image_list_section()

        # Preview section (right panel)
        self._create_preview_section()

        # Action buttons
        self._create_action_buttons()

    def _create_folder_section(self):
        """Create folder selection controls"""
        section = ctk.CTkFrame(self)
        section.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=10)

        ctk.CTkLabel(
            section,
            text="Source Folder",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=10, pady=5)

        # Folder row
        folder_row = ctk.CTkFrame(section, fg_color="transparent")
        folder_row.pack(fill="x", padx=10, pady=5)

        self.folder_entry = ctk.CTkEntry(folder_row, width=400)
        self.folder_entry.pack(side="left", padx=5)
        add_tooltip(self.folder_entry, "Path to folder containing images")

        self.browse_btn = ctk.CTkButton(
            folder_row,
            text="Browse",
            width=80,
            command=self._browse_folder
        )
        self.browse_btn.pack(side="left", padx=5)

        self.scan_btn = ctk.CTkButton(
            folder_row,
            text="Scan",
            width=80,
            fg_color="green",
            command=self._start_scan
        )
        self.scan_btn.pack(side="left", padx=5)

    def _create_options_section(self):
        """Create scan options"""
        section = ctk.CTkFrame(self)
        section.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=5)

        options_row = ctk.CTkFrame(section, fg_color="transparent")
        options_row.pack(fill="x", padx=10, pady=5)

        # Recursive option
        self.recursive_var = ctk.BooleanVar(value=True)
        self.recursive_cb = ctk.CTkCheckBox(
            options_row,
            text="Recursive (include subfolders)",
            variable=self.recursive_var
        )
        self.recursive_cb.pack(side="left", padx=10)
        add_tooltip(self.recursive_cb, "recursive_scan")

        # Filter options
        self.filter_var = ctk.StringVar(value="all")
        ctk.CTkLabel(options_row, text="Filter:").pack(side="left", padx=(20, 5))

        self.filter_combo = ctk.CTkComboBox(
            options_row,
            values=["All images", "Without metadata", "With metadata"],
            width=150,
            command=self._apply_filter
        )
        self.filter_combo.set("All images")
        self.filter_combo.pack(side="left", padx=5)

        # Status label
        self.status_label = ctk.CTkLabel(
            options_row,
            text="",
            text_color="gray"
        )
        self.status_label.pack(side="right", padx=10)

    def _create_image_list_section(self):
        """Create image list with scrolling"""
        # Container with list and preview
        container = ctk.CTkFrame(self)
        container.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(1, weight=1)

        # Header
        header = ctk.CTkFrame(container, height=30)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)

        # Selection controls
        self.select_all_btn = ctk.CTkButton(
            header,
            text="Select All",
            width=80,
            height=24,
            command=self._select_all
        )
        self.select_all_btn.pack(side="left", padx=5, pady=3)

        self.deselect_all_btn = ctk.CTkButton(
            header,
            text="Deselect All",
            width=80,
            height=24,
            command=self._deselect_all
        )
        self.deselect_all_btn.pack(side="left", padx=5, pady=3)

        self.invert_btn = ctk.CTkButton(
            header,
            text="Invert",
            width=60,
            height=24,
            command=self._invert_selection
        )
        self.invert_btn.pack(side="left", padx=5, pady=3)

        # Count label
        self.count_label = ctk.CTkLabel(header, text="0 images | 0 selected")
        self.count_label.pack(side="right", padx=10)

        # Column headers
        col_header = ctk.CTkFrame(container, height=25)
        col_header.grid(row=1, column=0, sticky="ew")
        col_header.grid_propagate(False)
        col_header.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(col_header, text="", width=30).grid(row=0, column=0)
        ctk.CTkLabel(col_header, text="Filename", anchor="w").grid(row=0, column=1, sticky="w", padx=5)
        ctk.CTkLabel(col_header, text="Size", width=70).grid(row=0, column=2)
        ctk.CTkLabel(col_header, text="Dimensions", width=100).grid(row=0, column=3)
        ctk.CTkLabel(col_header, text="Metadata", width=100).grid(row=0, column=4)

        # Scrollable list
        self.list_frame = ctk.CTkScrollableFrame(container, height=400)
        self.list_frame.grid(row=2, column=0, sticky="nsew", pady=5)
        self.list_frame.grid_columnconfigure(0, weight=1)

        self._list_widgets: List[ImageListItem] = []

    def _create_preview_section(self):
        """Create image preview panel"""
        section = ctk.CTkFrame(self, width=300)
        section.grid(row=2, column=1, sticky="nsew", padx=10, pady=5)
        section.grid_propagate(False)

        ctk.CTkLabel(
            section,
            text="Preview",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)

        # Preview canvas
        self.preview_label = ctk.CTkLabel(section, text="Select an image")
        self.preview_label.pack(pady=10)

        # Image info
        self.preview_info = ctk.CTkTextbox(section, height=150)
        self.preview_info.pack(fill="x", padx=10, pady=5)
        self.preview_info.configure(state="disabled")

    def _create_action_buttons(self):
        """Create action buttons"""
        section = ctk.CTkFrame(self)
        section.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=10)

        # Process button
        self.process_btn = ctk.CTkButton(
            section,
            text="Process Selected Images",
            width=200,
            height=40,
            fg_color="green",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._process_selected
        )
        self.process_btn.pack(side="right", padx=10)
        add_tooltip(self.process_btn, "Start AI analysis on selected images")

        # Export list button
        self.export_btn = ctk.CTkButton(
            section,
            text="Export List",
            width=100,
            command=self._export_list
        )
        self.export_btn.pack(side="left", padx=10)

    # ==================== Actions ====================

    def _browse_folder(self):
        """Open folder browser"""
        folder = filedialog.askdirectory(
            title="Select Image Folder"
        )
        if folder:
            self.folder_entry.delete(0, "end")
            self.folder_entry.insert(0, folder)
            self._start_scan()

    def _start_scan(self):
        """Start folder scanning"""
        folder = self.folder_entry.get().strip()
        if not folder:
            messagebox.showwarning("Warning", "Please select a folder first")
            return

        folder_path = Path(folder)
        if not folder_path.exists():
            messagebox.showerror("Error", f"Folder not found: {folder}")
            return

        self._scanning = True
        self._current_folder = folder_path
        self.scan_btn.configure(state="disabled", text="Scanning...")
        self.status_label.configure(text="Scanning...", text_color="orange")

        # Clear current list
        self._clear_list()

        def scan():
            try:
                # Collect files
                files = collect_image_files(
                    folder_path,
                    recursive=self.recursive_var.get(),
                    extensions=self.SUPPORTED_EXTENSIONS
                )

                # Get metadata status for each
                images = []
                for i, f in enumerate(files):
                    item = ImageItem(path=f)

                    try:
                        # Get file size
                        item.size_bytes = f.stat().st_size

                        # Get dimensions using PIL
                        with Image.open(f) as img:
                            item.width, item.height = img.size

                        # Check metadata
                        if self.metadata_reader:
                            meta = self.metadata_reader.read_quick_info(f)
                            if meta:
                                item.has_metadata = bool(
                                    meta.get("has_iptc") or meta.get("has_xmp")
                                )
                    except Exception as e:
                        logger.debug(f"Error reading {f}: {e}")

                    images.append(item)

                    # Update progress
                    if i % 10 == 0:
                        self.after(0, lambda c=i+1, t=len(files):
                            self.status_label.configure(text=f"Scanning... {c}/{t}"))

                self._images = images
                self.after(0, self._on_scan_complete)

            except Exception as e:
                logger.error(f"Scan error: {e}")
                self.after(0, lambda: self._on_scan_error(str(e)))

        threading.Thread(target=scan, daemon=True).start()

    def _on_scan_complete(self):
        """Called when scan completes"""
        self._scanning = False
        self.scan_btn.configure(state="normal", text="Scan")

        count = len(self._images)
        self.status_label.configure(
            text=f"Found {count} images",
            text_color="green"
        )

        self._populate_list()
        self._update_counts()

    def _on_scan_error(self, error: str):
        """Called when scan fails"""
        self._scanning = False
        self.scan_btn.configure(state="normal", text="Scan")
        self.status_label.configure(text=f"Error: {error}", text_color="red")
        messagebox.showerror("Scan Error", error)

    def _clear_list(self):
        """Clear the image list"""
        for widget in self._list_widgets:
            widget.destroy()
        self._list_widgets.clear()
        self._images.clear()

    def _populate_list(self):
        """Populate the list with images"""
        self._clear_list()

        for item in self._images:
            widget = ImageListItem(
                self.list_frame,
                item,
                on_select=self._on_item_selected,
                on_preview=self._show_preview
            )
            widget.pack(fill="x", pady=1)
            self._list_widgets.append(widget)

    def _apply_filter(self, filter_value: str):
        """Apply filter to image list"""
        for widget in self._list_widgets:
            show = True

            if filter_value == "Without metadata":
                show = not widget.image_item.has_metadata
            elif filter_value == "With metadata":
                show = widget.image_item.has_metadata

            if show:
                widget.pack(fill="x", pady=1)
            else:
                widget.pack_forget()

    def _select_all(self):
        """Select all visible images"""
        for widget in self._list_widgets:
            if widget.winfo_ismapped():
                widget.set_selected(True)
        self._update_counts()

    def _deselect_all(self):
        """Deselect all images"""
        for widget in self._list_widgets:
            widget.set_selected(False)
        self._update_counts()

    def _invert_selection(self):
        """Invert selection"""
        for widget in self._list_widgets:
            widget.set_selected(not widget.selected_var.get())
        self._update_counts()

    def _on_item_selected(self, path: Path, selected: bool):
        """Handle item selection change"""
        self._update_counts()

    def _update_counts(self):
        """Update selection counts"""
        total = len(self._images)
        selected = sum(1 for img in self._images if img.selected)
        self.count_label.configure(text=f"{total} images | {selected} selected")

        # Notify listener
        if self.on_images_selected:
            selected_paths = [img.path for img in self._images if img.selected]
            self.on_images_selected(selected_paths)

    def _show_preview(self, path: Path):
        """Show image preview"""
        try:
            # Load and resize image
            with Image.open(path) as img:
                # Calculate resize
                max_size = 250
                ratio = min(max_size / img.width, max_size / img.height)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img_resized = img.resize(new_size, Image.Resampling.LANCZOS)

                # Convert to CTk image
                photo = ctk.CTkImage(img_resized, size=new_size)
                self.preview_label.configure(image=photo, text="")
                self.preview_label._image = photo  # Keep reference

                # Update info
                self._update_preview_info(path, img)

        except Exception as e:
            self.preview_label.configure(image=None, text=f"Error: {e}")

    def _update_preview_info(self, path: Path, img: Image.Image):
        """Update preview info text"""
        info_lines = [
            f"Filename: {path.name}",
            f"Size: {path.stat().st_size / (1024*1024):.2f} MB",
            f"Dimensions: {img.width}x{img.height}",
            f"Resolution: {(img.width * img.height) / 1_000_000:.1f} MP",
            f"Format: {img.format}",
            f"Mode: {img.mode}",
        ]

        # Get metadata info if reader available
        if self.metadata_reader:
            try:
                meta = self.metadata_reader.read(path)
                if meta:
                    info_lines.append("")
                    info_lines.append("--- Metadata ---")
                    if meta.iptc.headline:
                        info_lines.append(f"Title: {meta.iptc.headline[:50]}...")
                    if meta.iptc.keywords:
                        info_lines.append(f"Keywords: {len(meta.iptc.keywords)}")
                    info_lines.append(f"EXIF: {'Yes' if meta.has_exif else 'No'}")
                    info_lines.append(f"IPTC: {'Yes' if meta.has_iptc else 'No'}")
                    info_lines.append(f"XMP: {'Yes' if meta.has_xmp else 'No'}")
            except Exception as e:
                info_lines.append(f"Metadata error: {e}")

        self.preview_info.configure(state="normal")
        self.preview_info.delete("1.0", "end")
        self.preview_info.insert("1.0", "\n".join(info_lines))
        self.preview_info.configure(state="disabled")

    def _process_selected(self):
        """Start processing selected images"""
        selected = [img.path for img in self._images if img.selected]

        if not selected:
            messagebox.showwarning("Warning", "No images selected")
            return

        if self.on_process_requested:
            self.on_process_requested(selected)
        else:
            messagebox.showinfo(
                "Process",
                f"Would process {len(selected)} images.\n"
                "Connect to AI Control page for actual processing."
            )

    def _export_list(self):
        """Export file list to CSV"""
        if not self._images:
            messagebox.showwarning("Warning", "No images to export")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("Path,Size (MB),Width,Height,Has Metadata,Selected\n")
                    for img in self._images:
                        f.write(f'"{img.path}",{img.size_mb:.2f},{img.width},{img.height},{img.has_metadata},{img.selected}\n')

                messagebox.showinfo("Export", f"Exported to {file_path}")

            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")

    # ==================== Public API ====================

    def get_selected_images(self) -> List[Path]:
        """Get list of selected image paths"""
        return [img.path for img in self._images if img.selected]

    def get_all_images(self) -> List[Path]:
        """Get list of all image paths"""
        return [img.path for img in self._images]

    def set_folder(self, folder: str):
        """Set folder path and scan"""
        self.folder_entry.delete(0, "end")
        self.folder_entry.insert(0, folder)
        self._start_scan()
