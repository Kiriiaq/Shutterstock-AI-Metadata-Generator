"""
Write Page - Write metadata directly to image files
"""

from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional

import customtkinter as ctk
from PIL import Image

from ...modules.engines.iptc_engine import IPTCEngine
from ...modules.engines.metadata_reader import MetadataReader
from ...modules.engines.metadata_writer import MetadataWriter
from ...modules.models.metadata_models import ImageMetadata, IPTCFields
from ...modules.storage.database import ActionType, Database


class WritePage(ctk.CTkFrame):
    """
    Page for writing metadata directly to image files
    """

    def __init__(
        self,
        parent,
        database: Database,
        metadata_reader: Optional[MetadataReader] = None,
        metadata_writer: Optional[MetadataWriter] = None,
        **kwargs,
    ):
        super().__init__(parent, **kwargs)

        self.database = database
        self.reader = metadata_reader
        self.writer = metadata_writer
        self.iptc_engine = IPTCEngine()

        # Current state
        self._current_file: Optional[Path] = None
        self._current_metadata: Optional[ImageMetadata] = None
        self._modified = False

        # Setup UI
        self._create_widgets()

    def _create_widgets(self):
        """Create UI widgets"""
        # Configure grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ============ Left Panel - File Browser ============
        left_panel = ctk.CTkFrame(self, width=300)
        left_panel.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="nsew")
        left_panel.grid_rowconfigure(1, weight=1)
        left_panel.grid_propagate(False)

        # Folder selection
        folder_frame = ctk.CTkFrame(left_panel)
        folder_frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkButton(folder_frame, text="Select Folder", command=self._select_folder).pack(side="left", padx=5, pady=5)

        self.folder_label = ctk.CTkLabel(folder_frame, text="No folder selected")
        self.folder_label.pack(side="left", padx=5, fill="x", expand=True)

        # File list
        list_frame = ctk.CTkFrame(left_panel)
        list_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.file_list = ctk.CTkScrollableFrame(list_frame)
        self.file_list.pack(fill="both", expand=True)

        # File count
        self.file_count_label = ctk.CTkLabel(left_panel, text="0 files")
        self.file_count_label.pack(pady=5)

        # ============ Middle Panel - Image Preview ============
        middle_panel = ctk.CTkFrame(self, width=350)
        middle_panel.grid(row=0, column=1, padx=5, pady=10, sticky="nsew")
        middle_panel.grid_rowconfigure(0, weight=1)
        middle_panel.grid_propagate(False)

        # Image preview
        self.preview_label = ctk.CTkLabel(middle_panel, text="Select an image", width=330, height=330)
        self.preview_label.pack(pady=10)

        # File info
        self.info_frame = ctk.CTkFrame(middle_panel)
        self.info_frame.pack(fill="x", padx=10, pady=5)

        self.file_name_label = ctk.CTkLabel(self.info_frame, text="", font=ctk.CTkFont(weight="bold"))
        self.file_name_label.pack()

        self.file_info_label = ctk.CTkLabel(self.info_frame, text="", text_color="gray")
        self.file_info_label.pack()

        # Metadata status indicators
        status_frame = ctk.CTkFrame(middle_panel)
        status_frame.pack(fill="x", padx=10, pady=5)

        self.exif_indicator = self._create_indicator(status_frame, "EXIF")
        self.iptc_indicator = self._create_indicator(status_frame, "IPTC")
        self.xmp_indicator = self._create_indicator(status_frame, "XMP")

        # ============ Right Panel - Metadata Editor ============
        right_panel = ctk.CTkFrame(self)
        right_panel.grid(row=0, column=2, padx=(5, 10), pady=10, sticky="nsew")
        right_panel.grid_columnconfigure(0, weight=1)
        right_panel.grid_rowconfigure(1, weight=1)

        # Template selector
        template_frame = ctk.CTkFrame(right_panel)
        template_frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(template_frame, text="Template:").pack(side="left", padx=5)
        self.template_combo = ctk.CTkComboBox(
            template_frame,
            values=["None"] + self.iptc_engine.list_templates(),
            width=150,
            command=self._on_template_change,
        )
        self.template_combo.set("None")
        self.template_combo.pack(side="left", padx=5)

        ctk.CTkButton(template_frame, text="Apply", width=60, command=self._apply_template).pack(side="left", padx=5)

        # Metadata editor tabs
        self.editor_tabs = ctk.CTkTabview(right_panel)
        self.editor_tabs.pack(fill="both", expand=True, padx=5, pady=5)

        # IPTC Tab
        iptc_tab = self.editor_tabs.add("IPTC")
        self._create_iptc_editor(iptc_tab)

        # XMP Tab
        xmp_tab = self.editor_tabs.add("XMP")
        self._create_xmp_editor(xmp_tab)

        # EXIF Tab (read-only)
        exif_tab = self.editor_tabs.add("EXIF")
        self._create_exif_viewer(exif_tab)

        # Action buttons
        button_frame = ctk.CTkFrame(right_panel)
        button_frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkButton(button_frame, text="Read from File", command=self._read_metadata).pack(
            side="left", padx=5, pady=5
        )

        ctk.CTkButton(button_frame, text="Write to File", fg_color="green", command=self._write_metadata).pack(
            side="left", padx=5, pady=5
        )

        ctk.CTkButton(button_frame, text="Clear All", fg_color="gray", command=self._clear_metadata).pack(
            side="left", padx=5, pady=5
        )

        # Batch operations
        batch_frame = ctk.CTkFrame(right_panel)
        batch_frame.pack(fill="x", padx=5, pady=(0, 5))

        # Batch write is a stub today; surface that honestly in the UI rather
        # than letting users click an enabled button that just shows a popup.
        ctk.CTkButton(
            batch_frame,
            text="Write to All Files (coming soon)",
            fg_color="gray",
            state="disabled",
            command=self._write_batch,
        ).pack(side="left", padx=5, pady=5)

        self.backup_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(batch_frame, text="Create Backup", variable=self.backup_var).pack(side="left", padx=5)

    def _create_indicator(self, parent, label: str) -> ctk.CTkLabel:
        """Create a metadata status indicator"""
        frame = ctk.CTkFrame(parent)
        frame.pack(side="left", padx=10, pady=5)

        indicator = ctk.CTkLabel(frame, text="", width=15, height=15, corner_radius=7, fg_color="gray")
        indicator.pack(side="left", padx=(0, 5))

        ctk.CTkLabel(frame, text=label).pack(side="left")

        return indicator

    def _create_iptc_editor(self, parent):
        """Create IPTC metadata editor"""
        scroll = ctk.CTkScrollableFrame(parent)
        scroll.pack(fill="both", expand=True)

        # Title/Object Name
        ctk.CTkLabel(scroll, text="Title (Object Name):").pack(anchor="w", padx=5, pady=(10, 0))
        self.iptc_title = ctk.CTkEntry(scroll, width=350)
        self.iptc_title.pack(fill="x", padx=5, pady=2)

        # Headline
        ctk.CTkLabel(scroll, text="Headline:").pack(anchor="w", padx=5, pady=(10, 0))
        self.iptc_headline = ctk.CTkEntry(scroll, width=350)
        self.iptc_headline.pack(fill="x", padx=5, pady=2)

        # Caption/Description
        ctk.CTkLabel(scroll, text="Caption/Description:").pack(anchor="w", padx=5, pady=(10, 0))
        self.iptc_caption = ctk.CTkTextbox(scroll, height=80)
        self.iptc_caption.pack(fill="x", padx=5, pady=2)

        # Keywords
        ctk.CTkLabel(scroll, text="Keywords (comma separated):").pack(anchor="w", padx=5, pady=(10, 0))
        self.iptc_keywords = ctk.CTkTextbox(scroll, height=60)
        self.iptc_keywords.pack(fill="x", padx=5, pady=2)

        # Creator/Byline
        ctk.CTkLabel(scroll, text="Creator/Byline:").pack(anchor="w", padx=5, pady=(10, 0))
        self.iptc_byline = ctk.CTkEntry(scroll, width=350)
        self.iptc_byline.pack(fill="x", padx=5, pady=2)

        # Copyright
        ctk.CTkLabel(scroll, text="Copyright Notice:").pack(anchor="w", padx=5, pady=(10, 0))
        self.iptc_copyright = ctk.CTkEntry(scroll, width=350)
        self.iptc_copyright.pack(fill="x", padx=5, pady=2)

        # Location section
        loc_frame = ctk.CTkFrame(scroll)
        loc_frame.pack(fill="x", padx=5, pady=10)

        ctk.CTkLabel(loc_frame, text="Location:", font=ctk.CTkFont(weight="bold")).pack(anchor="w")

        loc_grid = ctk.CTkFrame(loc_frame)
        loc_grid.pack(fill="x", pady=5)

        ctk.CTkLabel(loc_grid, text="City:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.iptc_city = ctk.CTkEntry(loc_grid, width=150)
        self.iptc_city.grid(row=0, column=1, padx=5, pady=2)

        ctk.CTkLabel(loc_grid, text="State:").grid(row=0, column=2, padx=5, pady=2, sticky="w")
        self.iptc_state = ctk.CTkEntry(loc_grid, width=150)
        self.iptc_state.grid(row=0, column=3, padx=5, pady=2)

        ctk.CTkLabel(loc_grid, text="Country:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.iptc_country = ctk.CTkEntry(loc_grid, width=150)
        self.iptc_country.grid(row=1, column=1, padx=5, pady=2)

        ctk.CTkLabel(loc_grid, text="Code:").grid(row=1, column=2, padx=5, pady=2, sticky="w")
        self.iptc_country_code = ctk.CTkEntry(loc_grid, width=50)
        self.iptc_country_code.grid(row=1, column=3, padx=5, pady=2, sticky="w")

        # Special Instructions
        ctk.CTkLabel(scroll, text="Special Instructions:").pack(anchor="w", padx=5, pady=(10, 0))
        self.iptc_instructions = ctk.CTkEntry(scroll, width=350)
        self.iptc_instructions.pack(fill="x", padx=5, pady=2)

    def _create_xmp_editor(self, parent):
        """Create XMP metadata editor"""
        scroll = ctk.CTkScrollableFrame(parent)
        scroll.pack(fill="both", expand=True)

        # Rating
        ctk.CTkLabel(scroll, text="Rating (0-5):").pack(anchor="w", padx=5, pady=(10, 0))
        self.xmp_rating = ctk.CTkSlider(scroll, from_=0, to=5, number_of_steps=5)
        self.xmp_rating.set(0)
        self.xmp_rating.pack(fill="x", padx=5, pady=2)

        # Label/Color
        ctk.CTkLabel(scroll, text="Label:").pack(anchor="w", padx=5, pady=(10, 0))
        self.xmp_label = ctk.CTkComboBox(scroll, values=["None", "Red", "Yellow", "Green", "Blue", "Purple"])
        self.xmp_label.set("None")
        self.xmp_label.pack(fill="x", padx=5, pady=2)

        # Subject/Keywords
        ctk.CTkLabel(scroll, text="Subject (XMP Keywords):").pack(anchor="w", padx=5, pady=(10, 0))
        self.xmp_subject = ctk.CTkTextbox(scroll, height=80)
        self.xmp_subject.pack(fill="x", padx=5, pady=2)

    def _create_exif_viewer(self, parent):
        """Create EXIF data viewer (read-only)"""
        self.exif_text = ctk.CTkTextbox(parent, state="disabled")
        self.exif_text.pack(fill="both", expand=True, padx=5, pady=5)

    def _select_folder(self):
        """Select folder with images"""
        folder = filedialog.askdirectory(title="Select Image Folder")
        if not folder:
            return

        folder_path = Path(folder)
        self.folder_label.configure(text=folder_path.name)

        # Clear existing file list
        for widget in self.file_list.winfo_children():
            widget.destroy()

        # Find image files
        extensions = [".jpg", ".jpeg", ".tif", ".tiff", ".png"]
        files = []
        for ext in extensions:
            files.extend(folder_path.glob(f"*{ext}"))
            files.extend(folder_path.glob(f"*{ext.upper()}"))

        files = sorted(set(files))
        self.file_count_label.configure(text=f"{len(files)} files")

        # Add file buttons
        for file_path in files:
            btn = ctk.CTkButton(
                self.file_list, text=file_path.name, anchor="w", command=lambda fp=file_path: self._select_file(fp)
            )
            btn.pack(fill="x", pady=2)

    def _select_file(self, file_path: Path):
        """Select a file for editing"""
        self._current_file = file_path

        # Update preview
        self._update_preview(file_path)

        # Load metadata
        self._read_metadata()

    def _update_preview(self, file_path: Path):
        """Update image preview"""
        try:
            # Load and resize image
            img = Image.open(file_path)
            img.thumbnail((320, 320), Image.Resampling.LANCZOS)

            # Convert to CTk image
            photo = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
            self.preview_label.configure(image=photo, text="")
            self.preview_label.image = photo  # Keep reference

            # Update file info
            stat = file_path.stat()
            self.file_name_label.configure(text=file_path.name)

            width, height = Image.open(file_path).size
            size_mb = stat.st_size / (1024 * 1024)
            self.file_info_label.configure(text=f"{width}x{height} | {size_mb:.2f} MB")

        except Exception as e:
            self.preview_label.configure(image=None, text=f"Preview error:\n{e}")

    def _read_metadata(self):
        """Read metadata from current file"""
        if not self._current_file or not self.reader:
            return

        try:
            # Read metadata
            self._current_metadata = self.reader.read(self._current_file)

            # Update indicators
            has_meta = self.reader.has_metadata(self._current_file)
            self._update_indicator(self.exif_indicator, has_meta.get("exif", False))
            self._update_indicator(self.iptc_indicator, has_meta.get("iptc", False))
            self._update_indicator(self.xmp_indicator, has_meta.get("xmp", False))

            # Populate IPTC fields
            iptc = self._current_metadata.iptc
            self.iptc_title.delete(0, "end")
            self.iptc_title.insert(0, iptc.object_name or "")

            self.iptc_headline.delete(0, "end")
            self.iptc_headline.insert(0, iptc.headline or "")

            self.iptc_caption.delete("1.0", "end")
            self.iptc_caption.insert("1.0", iptc.caption or "")

            self.iptc_keywords.delete("1.0", "end")
            self.iptc_keywords.insert("1.0", ", ".join(iptc.keywords or []))

            self.iptc_byline.delete(0, "end")
            self.iptc_byline.insert(0, iptc.byline or "")

            self.iptc_copyright.delete(0, "end")
            self.iptc_copyright.insert(0, iptc.copyright_notice or "")

            self.iptc_city.delete(0, "end")
            self.iptc_city.insert(0, iptc.city or "")

            self.iptc_state.delete(0, "end")
            self.iptc_state.insert(0, iptc.province_state or "")

            self.iptc_country.delete(0, "end")
            self.iptc_country.insert(0, iptc.country_name or "")

            self.iptc_country_code.delete(0, "end")
            self.iptc_country_code.insert(0, iptc.country_code or "")

            self.iptc_instructions.delete(0, "end")
            self.iptc_instructions.insert(0, iptc.special_instructions or "")

            # Populate XMP fields
            self.xmp_rating.set(self._current_metadata.xmp_rating or 0)
            self.xmp_label.set(self._current_metadata.xmp_label or "None")

            self.xmp_subject.delete("1.0", "end")
            self.xmp_subject.insert("1.0", ", ".join(self._current_metadata.xmp_subject or []))

            # Populate EXIF viewer
            self._update_exif_viewer()

            # Log action
            self.database.log_action(ActionType.METADATA_READ, file_path=str(self._current_file), success=True)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to read metadata: {e}")
            self.database.log_action(
                ActionType.METADATA_READ, file_path=str(self._current_file), success=False, error_message=str(e)
            )

    def _update_indicator(self, indicator: ctk.CTkLabel, has_data: bool):
        """Update status indicator color"""
        color = "green" if has_data else "gray"
        indicator.configure(fg_color=color)

    def _update_exif_viewer(self):
        """Update EXIF data display"""
        if not self._current_metadata:
            return

        self.exif_text.configure(state="normal")
        self.exif_text.delete("1.0", "end")

        md = self._current_metadata

        lines = [
            f"Camera: {md.camera_make} {md.camera_model}",
            f"Lens: {md.lens_model}",
            f"Focal Length: {md.focal_length}mm",
            f"Aperture: f/{md.aperture}",
            f"Shutter Speed: {md.shutter_speed}",
            f"ISO: {md.iso}",
            f"Flash: {'Yes' if md.flash_fired else 'No'}",
            f"Date Taken: {md.date_taken}",
            f"Dimensions: {md.width}x{md.height}",
            f"Megapixels: {md.megapixels}",
            f"Color Space: {md.color_space}",
        ]

        if md.gps_latitude and md.gps_longitude:
            lines.append(f"GPS: {md.gps_latitude:.6f}, {md.gps_longitude:.6f}")

        self.exif_text.insert("1.0", "\n".join(lines))
        self.exif_text.configure(state="disabled")

    def _get_iptc_from_form(self) -> IPTCFields:
        """Get IPTC data from form fields"""
        keywords_text = self.iptc_keywords.get("1.0", "end").strip()
        keywords = [k.strip() for k in keywords_text.split(",") if k.strip()]

        return IPTCFields(
            object_name=self.iptc_title.get().strip() or None,
            headline=self.iptc_headline.get().strip() or None,
            caption=self.iptc_caption.get("1.0", "end").strip() or None,
            keywords=keywords,
            byline=self.iptc_byline.get().strip() or None,
            copyright_notice=self.iptc_copyright.get().strip() or None,
            city=self.iptc_city.get().strip() or None,
            province_state=self.iptc_state.get().strip() or None,
            country_name=self.iptc_country.get().strip() or None,
            country_code=self.iptc_country_code.get().strip() or None,
            special_instructions=self.iptc_instructions.get().strip() or None,
        )

    def _write_metadata(self):
        """Write metadata to current file"""
        if not self._current_file or not self.writer:
            messagebox.showwarning("Warning", "No file selected or writer not available")
            return

        try:
            # Get IPTC from form
            iptc = self._get_iptc_from_form()

            # Validate
            is_valid, errors, warnings = self.iptc_engine.validate_iptc(iptc)

            if errors:
                if not messagebox.askyesno(
                    "Validation Errors", "Found validation errors:\n\n" + "\n".join(errors) + "\n\nWrite anyway?"
                ):
                    return

            # Configure backup
            self.writer.create_backup = self.backup_var.get()

            # Write IPTC
            self.writer.write_iptc(self._current_file, iptc)

            # Write XMP
            xmp_data = {
                "rating": int(self.xmp_rating.get()),
                "label": self.xmp_label.get() if self.xmp_label.get() != "None" else None,
            }

            subject_text = self.xmp_subject.get("1.0", "end").strip()
            if subject_text:
                xmp_data["subject"] = [s.strip() for s in subject_text.split(",") if s.strip()]

            self.writer.write_xmp(self._current_file, xmp_data)

            # Log success
            self.database.log_action(
                ActionType.METADATA_WRITE,
                file_path=str(self._current_file),
                success=True,
                details={"fields_written": list(iptc.to_dict().keys())},
            )

            # Save to history
            self.database.save_metadata_history(
                file_path=str(self._current_file),
                file_hash="",  # TODO: compute hash
                metadata_type="iptc",
                metadata=iptc.to_dict(),
                source="user_input",
            )

            messagebox.showinfo("Success", f"Metadata written to:\n{self._current_file.name}")

            # Refresh
            self._read_metadata()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to write metadata: {e}")
            self.database.log_action(
                ActionType.METADATA_WRITE, file_path=str(self._current_file), success=False, error_message=str(e)
            )

    def _clear_metadata(self):
        """Clear all form fields"""
        # IPTC
        self.iptc_title.delete(0, "end")
        self.iptc_headline.delete(0, "end")
        self.iptc_caption.delete("1.0", "end")
        self.iptc_keywords.delete("1.0", "end")
        self.iptc_byline.delete(0, "end")
        self.iptc_copyright.delete(0, "end")
        self.iptc_city.delete(0, "end")
        self.iptc_state.delete(0, "end")
        self.iptc_country.delete(0, "end")
        self.iptc_country_code.delete(0, "end")
        self.iptc_instructions.delete(0, "end")

        # XMP
        self.xmp_rating.set(0)
        self.xmp_label.set("None")
        self.xmp_subject.delete("1.0", "end")

    def _on_template_change(self, template_name: str):
        """Handle template selection change"""
        pass  # Just for selection, apply on button click

    def _apply_template(self):
        """Apply selected template to form"""
        template_name = self.template_combo.get()
        if template_name == "None":
            return

        template = self.iptc_engine.get_template(template_name)
        if not template:
            return

        # Apply template defaults
        if template.byline and not self.iptc_byline.get():
            self.iptc_byline.delete(0, "end")
            self.iptc_byline.insert(0, template.byline)

        if template.copyright_notice and not self.iptc_copyright.get():
            self.iptc_copyright.delete(0, "end")
            self.iptc_copyright.insert(0, template.copyright_notice)

        if template.base_keywords:
            current = self.iptc_keywords.get("1.0", "end").strip()
            if current:
                new_keywords = current + ", " + ", ".join(template.base_keywords)
            else:
                new_keywords = ", ".join(template.base_keywords)
            self.iptc_keywords.delete("1.0", "end")
            self.iptc_keywords.insert("1.0", new_keywords)

        if template.is_editorial:
            self.iptc_instructions.delete(0, "end")
            self.iptc_instructions.insert(0, template.editorial_instructions or "EDITORIAL USE ONLY")

        messagebox.showinfo("Template Applied", f"Applied template: {template.name}")

    def _write_batch(self):
        """Write metadata to all files in folder"""
        if not messagebox.askyesno(
            "Batch Write",
            "This will write the current metadata to ALL files in the selected folder.\n\n"
            "Are you sure you want to continue?",
        ):
            return

        # Stub: list collection + batch loop deferred until a proper
        # path-bound model replaces button-text introspection.
        messagebox.showinfo("Batch Write", "Batch write feature coming soon...")
