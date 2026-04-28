"""
Audit Page - View processing history, logs, and statistics
"""

import customtkinter as ctk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Callable
import threading

from ...modules.storage.database import Database, ActionType, AuditLog


class AuditPage(ctk.CTkFrame):
    """
    Audit page showing processing history and statistics
    """

    def __init__(self, parent, database: Database, **kwargs):
        """
        Initialize Audit Page

        Args:
            parent: Parent widget
            database: Database instance
        """
        super().__init__(parent, **kwargs)

        self.database = database

        # Filters
        self._filter_action: Optional[ActionType] = None
        self._filter_start_date: Optional[datetime] = None
        self._filter_end_date: Optional[datetime] = None
        self._filter_success: Optional[bool] = None

        # Setup UI
        self._create_widgets()
        self._load_data()

    def _create_widgets(self):
        """Create UI widgets"""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # ============ Statistics Section ============
        stats_frame = ctk.CTkFrame(self)
        stats_frame.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        stats_frame.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

        # Stats labels
        self.stat_total = self._create_stat_card(stats_frame, "Total Processed", "0", 0)
        self.stat_ai = self._create_stat_card(stats_frame, "AI Analysis", "0", 1)
        self.stat_metadata = self._create_stat_card(stats_frame, "With Metadata", "0", 2)
        self.stat_errors = self._create_stat_card(stats_frame, "Recent Errors", "0", 3)
        self.stat_batches = self._create_stat_card(stats_frame, "Total Batches", "0", 4)

        # ============ Filters Section ============
        filter_frame = ctk.CTkFrame(self)
        filter_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        # Action type filter
        ctk.CTkLabel(filter_frame, text="Action:").pack(side="left", padx=(10, 5))
        self.action_filter = ctk.CTkComboBox(
            filter_frame,
            values=["All"] + [a.value for a in ActionType],
            width=150,
            command=self._on_filter_change
        )
        self.action_filter.set("All")
        self.action_filter.pack(side="left", padx=5)

        # Date filter
        ctk.CTkLabel(filter_frame, text="Period:").pack(side="left", padx=(20, 5))
        self.date_filter = ctk.CTkComboBox(
            filter_frame,
            values=["All Time", "Today", "Last 7 Days", "Last 30 Days", "This Month"],
            width=120,
            command=self._on_filter_change
        )
        self.date_filter.set("Last 7 Days")
        self.date_filter.pack(side="left", padx=5)

        # Success filter
        ctk.CTkLabel(filter_frame, text="Status:").pack(side="left", padx=(20, 5))
        self.success_filter = ctk.CTkComboBox(
            filter_frame,
            values=["All", "Success", "Failed"],
            width=100,
            command=self._on_filter_change
        )
        self.success_filter.set("All")
        self.success_filter.pack(side="left", padx=5)

        # Refresh button
        ctk.CTkButton(
            filter_frame,
            text="Refresh",
            width=80,
            command=self._load_data
        ).pack(side="left", padx=(20, 5))

        # Export button
        ctk.CTkButton(
            filter_frame,
            text="Export",
            width=80,
            command=self._export_logs
        ).pack(side="left", padx=5)

        # ============ Log Table Section ============
        table_frame = ctk.CTkFrame(self)
        table_frame.grid(row=2, column=0, padx=10, pady=(5, 10), sticky="nsew")
        table_frame.grid_columnconfigure(0, weight=1)
        table_frame.grid_rowconfigure(0, weight=1)

        # Treeview for logs
        columns = ("timestamp", "action", "file", "status", "duration", "batch")
        self.log_tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            selectmode="browse"
        )

        # Configure columns
        self.log_tree.heading("timestamp", text="Timestamp", anchor="w")
        self.log_tree.heading("action", text="Action", anchor="w")
        self.log_tree.heading("file", text="File", anchor="w")
        self.log_tree.heading("status", text="Status", anchor="center")
        self.log_tree.heading("duration", text="Duration", anchor="center")
        self.log_tree.heading("batch", text="Batch ID", anchor="w")

        self.log_tree.column("timestamp", width=150, minwidth=130)
        self.log_tree.column("action", width=120, minwidth=100)
        self.log_tree.column("file", width=300, minwidth=200)
        self.log_tree.column("status", width=80, minwidth=60)
        self.log_tree.column("duration", width=80, minwidth=60)
        self.log_tree.column("batch", width=120, minwidth=100)

        # Scrollbars
        y_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.log_tree.yview)
        x_scroll = ttk.Scrollbar(table_frame, orient="horizontal", command=self.log_tree.xview)
        self.log_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        # Grid layout
        self.log_tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")

        # Bind double-click for details
        self.log_tree.bind("<Double-1>", self._show_log_details)

        # Style for treeview
        style = ttk.Style()
        style.configure("Treeview", rowheight=25)
        style.configure("Treeview.Heading", font=('Segoe UI', 10, 'bold'))

        # ============ Details Panel ============
        details_frame = ctk.CTkFrame(self)
        details_frame.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="ew")

        ctk.CTkLabel(
            details_frame,
            text="Double-click a log entry to view details",
            text_color="gray"
        ).pack(pady=10)

        self.details_text = ctk.CTkTextbox(details_frame, height=100)
        self.details_text.pack(fill="x", padx=10, pady=(0, 10))
        self.details_text.configure(state="disabled")

    def _create_stat_card(self, parent, label: str, value: str, column: int) -> ctk.CTkLabel:
        """Create a statistics card"""
        frame = ctk.CTkFrame(parent)
        frame.grid(row=0, column=column, padx=5, pady=5, sticky="ew")

        ctk.CTkLabel(
            frame,
            text=label,
            font=ctk.CTkFont(size=11),
            text_color="gray"
        ).pack(pady=(5, 0))

        value_label = ctk.CTkLabel(
            frame,
            text=value,
            font=ctk.CTkFont(size=24, weight="bold")
        )
        value_label.pack(pady=(0, 5))

        return value_label

    def _on_filter_change(self, *args):
        """Handle filter changes"""
        # Parse action filter
        action_val = self.action_filter.get()
        if action_val == "All":
            self._filter_action = None
        else:
            self._filter_action = ActionType(action_val)

        # Parse date filter
        date_val = self.date_filter.get()
        now = datetime.now()

        if date_val == "All Time":
            self._filter_start_date = None
            self._filter_end_date = None
        elif date_val == "Today":
            self._filter_start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
            self._filter_end_date = now
        elif date_val == "Last 7 Days":
            self._filter_start_date = now - timedelta(days=7)
            self._filter_end_date = now
        elif date_val == "Last 30 Days":
            self._filter_start_date = now - timedelta(days=30)
            self._filter_end_date = now
        elif date_val == "This Month":
            self._filter_start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            self._filter_end_date = now

        # Parse success filter
        success_val = self.success_filter.get()
        if success_val == "All":
            self._filter_success = None
        elif success_val == "Success":
            self._filter_success = True
        else:
            self._filter_success = False

        # Reload data
        self._load_data()

    def _load_data(self):
        """Load data from database"""
        # Load in background thread
        threading.Thread(target=self._load_data_thread, daemon=True).start()

    def _load_data_thread(self):
        """Background data loading"""
        try:
            # Load statistics
            stats = self.database.get_statistics()

            # Update stats on main thread
            self.after(0, lambda: self._update_stats(stats))

            # Load logs with filters
            logs = self.database.get_audit_logs(
                action_type=self._filter_action,
                start_date=self._filter_start_date,
                end_date=self._filter_end_date,
                limit=500
            )

            # Filter by success if needed
            if self._filter_success is not None:
                logs = [l for l in logs if l.success == self._filter_success]

            # Update table on main thread
            self.after(0, lambda: self._update_table(logs))

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to load data: {e}"))

    def _update_stats(self, stats: dict):
        """Update statistics display"""
        self.stat_total.configure(text=str(stats.get('total_processed', 0)))
        self.stat_ai.configure(text=str(stats.get('with_ai_analysis', 0)))
        self.stat_metadata.configure(text=str(stats.get('with_metadata', 0)))
        self.stat_errors.configure(text=str(stats.get('recent_errors', 0)))
        self.stat_batches.configure(text=str(stats.get('total_batches', 0)))

    def _update_table(self, logs: List[AuditLog]):
        """Update log table"""
        # Clear existing items
        for item in self.log_tree.get_children():
            self.log_tree.delete(item)

        # Add new items
        for log in logs:
            status = "Success" if log.success else "Failed"
            duration = f"{log.duration_ms}ms" if log.duration_ms else "-"
            file_name = Path(log.file_path).name if log.file_path else "-"

            self.log_tree.insert("", "end", values=(
                log.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                log.action_type.value,
                file_name,
                status,
                duration,
                log.batch_id[:8] if log.batch_id else "-"
            ), tags=("error",) if not log.success else ())

        # Style error rows
        self.log_tree.tag_configure("error", foreground="red")

    def _show_log_details(self, event):
        """Show details for selected log entry"""
        selection = self.log_tree.selection()
        if not selection:
            return

        item = self.log_tree.item(selection[0])
        values = item['values']

        # Get full log entry
        timestamp_str = values[0]
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

        logs = self.database.get_audit_logs(
            start_date=timestamp - timedelta(seconds=1),
            end_date=timestamp + timedelta(seconds=1),
            limit=1
        )

        if logs:
            log = logs[0]
            details = f"Timestamp: {log.timestamp}\n"
            details += f"Action: {log.action_type.value}\n"
            details += f"File: {log.file_path}\n"
            details += f"Status: {'Success' if log.success else 'Failed'}\n"
            details += f"Duration: {log.duration_ms}ms\n"
            details += f"Batch ID: {log.batch_id}\n"

            if log.error_message:
                details += f"\nError: {log.error_message}\n"

            if log.details:
                details += f"\nDetails: {log.details}\n"

            self.details_text.configure(state="normal")
            self.details_text.delete("1.0", "end")
            self.details_text.insert("1.0", details)
            self.details_text.configure(state="disabled")

    def _export_logs(self):
        """Export audit logs to file"""
        file_path = filedialog.asksaveasfilename(
            title="Export Audit Logs",
            defaultextension=".json",
            filetypes=[
                ("JSON Files", "*.json"),
                ("CSV Files", "*.csv"),
                ("All Files", "*.*")
            ]
        )

        if not file_path:
            return

        try:
            file_path = Path(file_path)
            format_type = "csv" if file_path.suffix.lower() == ".csv" else "json"

            count = self.database.export_audit_log(file_path, format=format_type)
            messagebox.showinfo("Export Complete", f"Exported {count} log entries to:\n{file_path}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export logs: {e}")

    def refresh(self):
        """Refresh the audit page"""
        self._load_data()
