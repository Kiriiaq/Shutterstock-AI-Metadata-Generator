"""WorkspaceView — single-screen Atelier with 7 tool panels.

Design rule: per tool, one panel with its indicators and its actions
visible at all times. No duplicated entry-point. Deep editing surfaces
(full settings form, full audit table, etc.) open as modals from the
corresponding panel's "Détail…" button — exactly one path per tool.

Layout (1400×900 fits comfortably; both columns are CTkScrollableFrames
so the panels stay reachable when the user shrinks the window):
    LEFT col (≈ 60 %)               RIGHT col (≈ 40 %)
    ─────────────────────           ─────────────────────
    SOURCES & TRI         (big)     MODÈLE IA            (compact)
    ÉDITION IPTC          (med)     VALIDATION           (compact)
    ANALYSE IA            (med)     HISTORIQUE           (compact)
                                    PARAMÈTRES           (compact)
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from tkinter import filedialog
from typing import TYPE_CHECKING, Any

import customtkinter as ctk

from app.components.data_table import Column, DataTable
from app.config.theme import (
    RADIUS_MD,
    RADIUS_SM,
    SPACE_LG,
    SPACE_MD,
    SPACE_SM,
    SPACE_XS,
    get_font,
    palette_pair,
)
from app.utils.formatters import fmt_int, fmt_size
from app.views.base_view import BaseView

if TYPE_CHECKING:
    from app.app import App

logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
REFRESH_INTERVAL_MS = 5000
HISTORY_TAIL = 5


class WorkspaceView(BaseView):
    view_id = "home"

    def __init__(self, master: ctk.CTkFrame, *, app: "App") -> None:
        super().__init__(master)
        self.app = app
        # State
        self._scanned: list[dict[str, Any]] = []
        self._current_path: Path | None = None
        self._processing = False
        self._refresh_after_id: str | None = None
        # Widgets
        self._iptc_fields: dict[str, ctk.CTkEntry] = {}
        self._editor_collapsed: bool = False
        self._build()

    # ------------------------------------------------------------------
    # Layout

    def _build(self) -> None:
        # Two-column layout. Both columns are plain ``CTkFrame``s now —
        # the unified vertical scroll lives at ``App._center`` (one
        # ``CTkScrollableFrame`` for the whole window). When content
        # overflows, the WHOLE workspace scrolls together; when it
        # fits, no scrollbar.
        #
        # Bottom alignment: each column reserves a stretching row at
        # its bottom (``grid_rowconfigure(LAST, weight=1)`` + the last
        # panel grids ``sticky="nsew"``) so the columns end on the
        # same horizontal line regardless of how many panels each
        # column holds (3 left, 4 right).
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2, minsize=320)
        self.grid_rowconfigure(0, weight=1)

        # Phase G (2026-05-16) — marges externes augmentées (SPACE_LG
        # au lieu de SPACE_MD à gauche du gauche / droite du droit, et
        # en haut/bas) pour que le workspace ne soit pas collé au bord
        # de la fenêtre Windows (effet « scotché » remonté par
        # l'utilisateur — on perd 4 px supplémentaires de chaque côté
        # mais la lecture devient plus aérée et le focus visuel est
        # mieux séparé du chrome de l'OS).
        left = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        left.grid(row=0, column=0, sticky="nsew", padx=(SPACE_LG, SPACE_SM), pady=SPACE_LG)
        left.grid_columnconfigure(0, weight=1)
        # Last panel (Analyse IA, row 2) absorbs vertical slack — its
        # textbox naturally grows, but the row weight guarantees the
        # column reaches the same y-bottom as the right column even
        # when the textbox is empty.
        left.grid_rowconfigure(2, weight=1)

        right = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        right.grid(row=0, column=1, sticky="nsew", padx=(SPACE_SM, SPACE_LG), pady=SPACE_LG)
        right.grid_columnconfigure(0, weight=1)
        # Last panel on the right is Paramètres (row 3) — same trick:
        # weight=1 on its row stretches it to match the left column.
        right.grid_rowconfigure(3, weight=1)

        self._build_sources_panel(left, row=0)
        self._build_editor_panel(left, row=1)
        self._build_analyze_panel(left, row=2)

        self._build_model_panel(right, row=0)
        self._build_validate_panel(right, row=1)
        self._build_history_panel(right, row=2)
        self._build_settings_panel(right, row=3)

        # Initial state computation. ``_sync_sources_state`` projects
        # the (empty) model into the counter, app_state and the two
        # row-level buttons (Supprimer / Vider) — and itself calls
        # ``_refresh_action_states`` for the Analyse IA Démarrer/Arrêter
        # pair. Without this initial call, Démarrer would stay in its
        # default visual state until the user touches Sources.
        self._sync_sources_state()

    # ==================================================================
    # LEFT COLUMN — production loop
    # ==================================================================

    # ----- Panel: Sources & tri ---------------------------------------

    def _build_sources_panel(self, parent: ctk.CTkFrame, row: int) -> None:
        # bg_key="bg_deep" → soft-gray bg_secondary in light mode (so it
        # blends with the new neutral palette instead of glowing white),
        # slate-950 in dark mode for the "workspace floor" feel.
        #
        # Phase F (2026-05-14, audit T-030..T-033) — refonte du
        # modèle de fichiers : on conserve l'entrée + bouton Scanner
        # historique (compat workflow par dossier), mais on ajoute une
        # ligne d'actions explicites "Ajouter fichiers / Ajouter dossier
        # / Supprimer / Vider" et un compteur permanent toujours
        # visible. Le modèle ``_scanned`` est désormais incrémentiel
        # (extends au lieu de replace), avec dédoublonnage par chemin.
        section = self._panel(parent, row, "SOURCES & TRI", bg_key="bg_deep", icon="📁")
        # Phase G (2026-05-18) — la rangée "opts" (compteur seul) est
        # supprimée et le compteur est déplacé inline dans la rangée
        # d'actions. Le DataTable passe de row=4 à row=3.
        section.grid_rowconfigure(3, weight=1)

        # Phase G (2026-05-16) : la checkbox "Récursif" est déplacée
        # sur la même ligne que les boutons (juste après "Scanner") au
        # lieu d'avoir sa propre rangée — gain de place vertical et
        # cohérence (les trois éléments qui pilotent le scan dossier
        # sont sur le même axe). La ligne 2 ne contient plus que le
        # compteur permanent.
        bar = ctk.CTkFrame(section, fg_color="transparent")
        bar.grid(row=1, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_XS))
        bar.grid_columnconfigure(0, weight=1)

        self._folder_entry = ctk.CTkEntry(bar, font=get_font("body"), placeholder_text="Dossier source…")
        self._folder_entry.grid(row=0, column=0, sticky="ew", padx=(0, SPACE_XS))
        ctk.CTkButton(bar, text="…", width=36, command=self._browse).grid(row=0, column=1, padx=2)
        self._scan_btn = ctk.CTkButton(
            bar,
            text="Scanner",
            width=90,
            fg_color=palette_pair("accent"),
            hover_color=palette_pair("accent_hover"),
            text_color=palette_pair("accent_fg"),
            command=self._scan,
        )
        self._scan_btn.grid(row=0, column=2, padx=(2, 0))
        self._recursive_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(bar, text="Récursif", variable=self._recursive_var, font=get_font("body")).grid(
            row=0, column=3, padx=(SPACE_SM, 0)
        )

        # Phase G (2026-05-18) — la ligne 2 "opts" (qui n'avait que le
        # compteur) est supprimée. Le compteur _sources_status est
        # déplacé à droite des boutons d'action (ligne 3) — gain de
        # place vertical et l'info "combien de fichiers / sélectionnés"
        # est désormais sur la même ligne que les actions qui les
        # modifient.

        # Dedicated action row for the incremental model. The buttons
        # all share the same vertical band immediately above the table
        # so the relationship between "what's listed" and "what can I
        # do with it" is unambiguous.
        actions = ctk.CTkFrame(section, fg_color="transparent")
        actions.grid(row=2, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_XS))
        self._add_files_btn = ctk.CTkButton(
            actions,
            text="+ Fichiers…",
            width=110,
            height=26,
            fg_color=palette_pair("bg_hover"),
            hover_color=palette_pair("bg_active"),
            text_color=palette_pair("fg"),
            border_width=1,
            border_color=palette_pair("border"),
            command=self._add_files,
        )
        self._add_files_btn.pack(side="left", padx=(0, SPACE_XS))
        self._add_folder_btn = ctk.CTkButton(
            actions,
            text="+ Dossier…",
            width=110,
            height=26,
            fg_color=palette_pair("bg_hover"),
            hover_color=palette_pair("bg_active"),
            text_color=palette_pair("fg"),
            border_width=1,
            border_color=palette_pair("border"),
            command=self._add_folder,
        )
        self._add_folder_btn.pack(side="left", padx=SPACE_XS)
        self._remove_btn = ctk.CTkButton(
            actions,
            text="Supprimer",
            width=100,
            height=26,
            fg_color=palette_pair("bg_hover"),
            hover_color=palette_pair("bg_active"),
            text_color=palette_pair("fg"),
            border_width=1,
            border_color=palette_pair("border"),
            state="disabled",
            command=self._remove_selected,
        )
        self._remove_btn.pack(side="left", padx=SPACE_XS)
        self._clear_btn = ctk.CTkButton(
            actions,
            text="Vider",
            width=70,
            height=26,
            fg_color=palette_pair("bg_hover"),
            hover_color=palette_pair("error"),
            text_color=palette_pair("fg"),
            border_width=1,
            border_color=palette_pair("border"),
            state="disabled",
            command=self._clear_all,
        )
        self._clear_btn.pack(side="left", padx=SPACE_XS)

        # Compteur permanent — désormais aligné à droite, sur la même
        # ligne que les boutons. Format demandé : "nombre de fichiers :
        # N · M sélectionné(s)". Visible dès le démarrage en gris doux,
        # passe en couleur accent dès qu'il y a des fichiers.
        self._sources_status = ctk.CTkLabel(
            actions,
            text="nombre de fichiers : 0",
            font=get_font("small"),
            text_color=palette_pair("fg_muted"),
        )
        self._sources_status.pack(side="right", padx=(SPACE_SM, 0))

        self._sources_table = DataTable(
            section,
            columns=[
                Column(id="name", label="Fichier", width=240),
                Column(id="size", label="Taille", width=70, anchor="e"),
                Column(id="dim", label="Dim.", width=80, anchor="center"),
                Column(id="meta", label="Méta", width=50, anchor="center"),
            ],
            select_mode="extended",
        )
        self._sources_table.grid(row=3, column=0, sticky="nsew", padx=SPACE_SM, pady=(0, SPACE_SM))
        self._sources_table.on_select(self._on_sources_select)
        self._sources_table.on_activate(self._on_sources_activate)
        # Suppr clavier — fait le même travail que le bouton "Supprimer".
        try:
            self._sources_table._tree.bind("<Delete>", lambda _e: self._remove_selected())
        except Exception:
            logger.exception("Could not bind <Delete> on sources table")

    # ----- Sources model: incremental add / remove / clear -----------

    def _browse(self) -> None:
        path = filedialog.askdirectory(title="Choisir un dossier")
        if path:
            self._folder_entry.delete(0, "end")
            self._folder_entry.insert(0, path)
            self._scan()

    def _scan(self) -> None:
        """Scanner historique — REMPLACE le modèle par les fichiers du
        dossier (compat workflow legacy). Pour ajouter sans écraser,
        utiliser ``Ajouter dossier`` (méthode ``_add_folder``)."""
        folder = self._folder_entry.get().strip()
        if not folder or not Path(folder).is_dir():
            self.app.toasts.show("Dossier introuvable.", kind="error")
            return
        self._scan_btn.configure(state="disabled", text="Scan…")
        self._sources_status.configure(text="Recherche…", text_color=palette_pair("warning"))
        threading.Thread(
            target=self._collect_worker,
            args=(Path(folder), self._recursive_var.get(), True),
            daemon=True,
        ).start()

    def _add_files(self) -> None:
        """Ajout incrémental d'un ou plusieurs fichiers via filedialog."""
        patterns = [("Images", " ".join(f"*{e}" for e in SUPPORTED_EXTS)), ("Tous les fichiers", "*.*")]
        paths = filedialog.askopenfilenames(title="Ajouter des fichiers", filetypes=patterns)
        if not paths:
            return
        files = [Path(p) for p in paths if Path(p).is_file()]
        if not files:
            return
        self._sources_status.configure(text="Lecture…", text_color=palette_pair("warning"))
        threading.Thread(target=self._enrich_and_append_worker, args=(files, None), daemon=True).start()

    def _add_folder(self) -> None:
        """Ajout incrémental — sélection d'un dossier, append (pas
        replace) au modèle existant. Respecte l'option ``Récursif``."""
        path = filedialog.askdirectory(title="Ajouter un dossier")
        if not path:
            return
        self._sources_status.configure(text="Recherche…", text_color=palette_pair("warning"))
        threading.Thread(
            target=self._collect_worker,
            args=(Path(path), self._recursive_var.get(), False),
            daemon=True,
        ).start()

    def _remove_selected(self) -> None:
        """Retire les lignes sélectionnées du modèle et de la table.
        Synchronise compteur, app_state et états des boutons."""
        selected = self._sources_table.get_selected()
        if not selected:
            return
        to_remove = {r.get("_path") for r in selected}
        self._scanned = [r for r in self._scanned if r.get("_path") not in to_remove]
        self._sources_table.set_rows(self._scanned)
        self._sync_sources_state(message=f"{fmt_int(len(to_remove))} fichier(s) retiré(s)", kind="warning")

    def _clear_all(self) -> None:
        """Vide totalement le modèle + la table + l'app_state."""
        if not self._scanned:
            return
        n = len(self._scanned)
        self._scanned = []
        self._sources_table.set_rows([])
        self._sync_sources_state(message=f"{fmt_int(n)} fichier(s) supprimé(s)", kind="warning")

    def _collect_worker(self, folder: Path, recursive: bool, replace: bool) -> None:
        """Récupère la liste d'images d'un dossier puis enrichit les
        métadonnées. Fait le travail hors mainloop pour rester réactif.

        ``replace=True`` pour le Scanner historique, ``False`` pour
        l'ajout incrémental.
        """
        try:
            from src.modules.workers.worker_pool import collect_image_files

            files = collect_image_files(folder, recursive=recursive, extensions=list(SUPPORTED_EXTS))
            self.after(0, lambda fld=folder, fl=files, rep=replace: self._continue_enrich(fld, fl, rep))
        except Exception as e:
            logger.exception("Collect failed")
            self.after(0, lambda err=str(e): self._on_sources_failed(err))

    def _continue_enrich(self, folder: Path, files: list[Path], replace: bool) -> None:
        """Phase 2 du collect_worker — démarre l'enrichissement."""
        if replace:
            # Scanner-style: reset before enrichment.
            self._scanned = []
            self._sources_table.set_rows([])
            self.app.app_state.set("source_folder", folder)
        threading.Thread(
            target=self._enrich_and_append_worker,
            args=(files, folder),
            daemon=True,
        ).start()

    def _enrich_and_append_worker(self, files: list[Path], folder: Path | None) -> None:
        """Lit dimensions + métadonnées pour chaque fichier, puis
        re-injecte dans le modèle en évitant les doublons par chemin."""
        try:
            from PIL import Image

            api = self.app.api
            reader = api.metadata_reader if api else None
            existing = {r["_path"] for r in self._scanned}
            new_rows: list[dict[str, Any]] = []
            for f in files:
                if f in existing:
                    continue
                row: dict[str, Any] = {
                    "_path": f,
                    "name": f.name,
                    "size": fmt_size(f.stat().st_size) if f.exists() else "—",
                    "dim": "—",
                    "meta": "—",
                }
                try:
                    with Image.open(f) as im:
                        row["dim"] = f"{im.width}×{im.height}"
                except Exception:
                    pass
                if reader is not None:
                    try:
                        row["meta"] = "Oui" if reader.get_quick_info(f) else "Non"
                    except Exception:
                        row["meta"] = "?"
                new_rows.append(row)
            self.after(0, lambda rows=new_rows, fld=folder: self._on_sources_appended(rows, fld))
        except Exception as e:
            logger.exception("Enrich failed")
            self.after(0, lambda err=str(e): self._on_sources_failed(err))

    def _on_sources_appended(self, new_rows: list[dict[str, Any]], folder: Path | None) -> None:
        """Met à jour le modèle + la table avec les nouvelles lignes."""
        self._scanned.extend(new_rows)
        self._sources_table.set_rows(self._scanned)
        self._scan_btn.configure(state="normal", text="Scanner")
        suffix = f" · {folder.name}" if folder is not None else ""
        added = len(new_rows)
        kind = "success" if new_rows else "warning"
        msg = (
            f"+{fmt_int(added)} ajouté(s){suffix}"
            if added
            else f"Aucun nouveau fichier{suffix} (déjà présent ou format non supporté)"
        )
        self._sync_sources_state(message=msg, kind=kind)
        # Validation panel can summarise too.
        try:
            self._validate_summary.configure(text=f"{fmt_int(len(self._scanned))} fichier(s) · non validés")
        except Exception:
            pass

    def _on_sources_failed(self, err: str) -> None:
        self._scan_btn.configure(state="normal", text="Scanner")
        self._sources_status.configure(text=f"Erreur : {err}", text_color=palette_pair("error"))

    def _sync_sources_state(self, *, message: str | None = None, kind: str = "fg_muted") -> None:
        """Source of truth pour : compteur, app_state, états boutons.

        Appelée après toute mutation du modèle (add / remove / clear /
        sélection). Garantit que les 6 surfaces dépendant de la
        sélection sont cohérentes :
            1. Label compteur ``_sources_status``
            2. app_state["scanned_images"]
            3. app_state["selected_paths"]
            4. Bouton Supprimer (selon sélection)
            5. Bouton Vider (selon contenu)
            6. Bouton Démarrer (centralisé via ``_refresh_action_states``)
        """
        n_total = len(self._scanned)
        selected = self._sources_table.get_selected() if hasattr(self, "_sources_table") else []
        n_sel = len(selected)

        kind_to_color = {
            "success": palette_pair("success"),
            "warning": palette_pair("warning"),
            "error": palette_pair("error"),
            "fg_muted": palette_pair("fg_muted"),
            "fg": palette_pair("fg"),
        }
        # Phase G (2026-05-18) — format demandé par l'utilisateur :
        # "nombre de fichiers : N" (avec compteur sélection si > 0).
        if message:
            sel_part = f" · {fmt_int(n_sel)} sélectionné(s)" if n_sel else ""
            text = f"nombre de fichiers : {fmt_int(n_total)}{sel_part} — {message}"
            color = kind_to_color.get(kind, palette_pair("fg_muted"))
        elif n_total == 0:
            text = "nombre de fichiers : 0"
            color = palette_pair("fg_muted")
        elif n_sel:
            text = f"nombre de fichiers : {fmt_int(n_total)} · {fmt_int(n_sel)} sélectionné(s)"
            color = palette_pair("fg")
        else:
            text = f"nombre de fichiers : {fmt_int(n_total)}"
            color = palette_pair("fg_muted")

        self._sources_status.configure(text=text, text_color=color)

        # Sync app_state for downstream consumers (analyze, validate…)
        self.app.app_state.set("scanned_images", [r["_path"] for r in self._scanned])
        self.app.app_state.set("selected_paths", [r["_path"] for r in selected])

        # Sync row-level action buttons
        try:
            self._remove_btn.configure(state="normal" if n_sel else "disabled")
            self._clear_btn.configure(state="normal" if n_total else "disabled")
        except Exception:
            pass

        # Analyse IA dependents
        try:
            self._analyze_summary.configure(
                text=(f"{fmt_int(n_sel)} / {fmt_int(n_total)} sélectionnée(s)" if n_total else "Aucune image")
            )
        except Exception:
            pass
        self._refresh_action_states()

    def _on_sources_select(self, _selected: list[dict[str, Any]]) -> None:
        # Toute mutation de sélection passe par ``_sync_sources_state``
        # — qui se charge des trois projections (app_state, compteur,
        # états boutons). Garantit qu'on n'oublie pas une projection
        # quand on ajoute une nouvelle surface dépendante.
        self._sync_sources_state()

    def _on_sources_activate(self, row: dict[str, Any]) -> None:
        path = row.get("_path")
        if isinstance(path, Path):
            self._select_for_edit(path)

    # ----- Panel: Édition IPTC ----------------------------------------

    def _build_editor_panel(self, parent: ctk.CTkFrame, row: int) -> None:
        # Custom build (skips _panel helper) so the title bar can host a
        # collapse chevron. Body widgets live in ``self._editor_body``;
        # ``_toggle_editor_collapsed`` grids/un-grids that wrapper. The
        # title row stays visible when collapsed so the panel never
        # disappears entirely.
        section = ctk.CTkFrame(
            parent,
            fg_color=palette_pair("bg_elevated"),
            border_color=palette_pair("border"),
            border_width=1,
            corner_radius=RADIUS_MD,
        )
        section.grid(row=row, column=0, sticky="nsew", pady=(0, SPACE_SM))
        section.grid_columnconfigure(0, weight=1)

        # Header: icon + title (top-left) + chevron toggle (right)
        header = ctk.CTkFrame(section, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=SPACE_SM, pady=(SPACE_SM, SPACE_XS))
        header.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            header,
            text="✎",
            font=get_font("body_strong"),
            text_color=palette_pair("fg_muted"),
            width=18,
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=(0, SPACE_XS))
        ctk.CTkLabel(
            header,
            text="ÉDITION IPTC",
            font=get_font("small"),
            text_color=palette_pair("fg_subtle"),
            anchor="w",
        ).grid(row=0, column=1, sticky="w")
        self._editor_toggle_btn = ctk.CTkButton(
            header,
            text="▼",
            width=24,
            height=20,
            corner_radius=RADIUS_SM,
            fg_color="transparent",
            hover_color=palette_pair("bg_hover"),
            text_color=palette_pair("fg_muted"),
            font=get_font("small"),
            command=self._toggle_editor_collapsed,
        )
        self._editor_toggle_btn.grid(row=0, column=2, sticky="e")

        # Body wrapper — grid_remove'd when collapsed.
        self._editor_body = ctk.CTkFrame(section, fg_color="transparent")
        self._editor_body.grid(row=1, column=0, sticky="ew")
        self._editor_body.grid_columnconfigure(0, weight=1)

        head = ctk.CTkFrame(self._editor_body, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_XS))
        head.grid_columnconfigure(0, weight=1)
        self._editor_path_label = ctk.CTkLabel(
            head,
            text="(double-cliquez sur une image dans Sources)",
            font=get_font("small"),
            text_color=palette_pair("fg_muted"),
            anchor="w",
        )
        self._editor_path_label.grid(row=0, column=0, sticky="ew")

        form = ctk.CTkFrame(self._editor_body, fg_color="transparent")
        form.grid(row=1, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_XS))
        form.grid_columnconfigure(1, weight=1)
        for r, (key, label) in enumerate(
            [
                ("headline", "Titre"),
                ("caption", "Description"),
                ("keywords", "Mots-clés"),
                ("byline", "Auteur"),
                ("copyright_notice", "Copyright"),
            ]
        ):
            ctk.CTkLabel(
                form, text=label, font=get_font("small"), text_color=palette_pair("fg_muted"), width=80, anchor="w"
            ).grid(row=r, column=0, sticky="w", padx=(0, SPACE_XS), pady=1)
            entry = ctk.CTkEntry(form, font=get_font("body"), height=24)
            entry.grid(row=r, column=1, sticky="ew", pady=1)
            self._iptc_fields[key] = entry

        actions = ctk.CTkFrame(self._editor_body, fg_color="transparent")
        actions.grid(row=2, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_SM))
        ctk.CTkButton(actions, text="Lire", width=60, height=26, command=self._editor_read).pack(
            side="left", padx=(0, SPACE_XS)
        )
        ctk.CTkButton(
            actions,
            text="Écrire",
            width=80,
            height=26,
            fg_color=palette_pair("accent"),
            hover_color=palette_pair("accent_hover"),
            text_color=palette_pair("accent_fg"),
            command=self._editor_write,
        ).pack(side="left", padx=SPACE_XS)
        ctk.CTkButton(actions, text="Effacer", width=70, height=26, command=self._editor_clear).pack(
            side="left", padx=SPACE_XS
        )
        self._editor_status = ctk.CTkLabel(
            actions, text="", font=get_font("small"), text_color=palette_pair("fg_muted")
        )
        self._editor_status.pack(side="right")

    def _toggle_editor_collapsed(self) -> None:
        """Show/hide the Editor IPTC body, keeping the title bar visible."""
        self._editor_collapsed = not self._editor_collapsed
        if self._editor_collapsed:
            self._editor_body.grid_remove()
            self._editor_toggle_btn.configure(text="▶")
        else:
            self._editor_body.grid()
            self._editor_toggle_btn.configure(text="▼")

    def _select_for_edit(self, path: Path) -> None:
        self._current_path = path
        self._editor_path_label.configure(text=path.name, text_color=palette_pair("fg"))
        self._editor_read()

    def _editor_read(self) -> None:
        if self._current_path is None:
            self.app.toasts.show("Sélectionnez d'abord un fichier.", kind="warning")
            return
        api = self.app.api
        if api is None or api.metadata_reader is None:
            self._editor_status.configure(text="ExifTool absent", text_color=palette_pair("warning"))
            return
        try:
            metadata = api.read_metadata(self._current_path)
        except Exception:
            logger.exception("read_metadata failed")
            self._editor_status.configure(text="Lecture échouée", text_color=palette_pair("error"))
            return
        if metadata is None:
            self._editor_clear()
            return
        iptc = metadata.iptc
        self._iptc_set("headline", iptc.headline or iptc.object_name or "")
        self._iptc_set("caption", iptc.caption or "")
        self._iptc_set("keywords", ", ".join(iptc.keywords or []))
        self._iptc_set("byline", iptc.byline or "")
        self._iptc_set("copyright_notice", iptc.copyright_notice or "")
        self._editor_status.configure(text="Lu", text_color=palette_pair("success"))

    def _editor_write(self) -> None:
        if self._current_path is None:
            self.app.toasts.show("Sélectionnez d'abord un fichier.", kind="warning")
            return
        api = self.app.api
        if api is None or api.metadata_writer is None:
            self._editor_status.configure(text="ExifTool absent", text_color=palette_pair("warning"))
            return
        from src.modules.models.metadata_models import IPTCFields

        kw = [k.strip() for k in self._iptc_get("keywords").split(",") if k.strip()]
        iptc = IPTCFields(
            headline=self._iptc_get("headline") or None,
            caption=self._iptc_get("caption") or None,
            keywords=kw,
            byline=self._iptc_get("byline") or None,
            copyright_notice=self._iptc_get("copyright_notice") or None,
        )
        try:
            ok = api.write_metadata(self._current_path, iptc=iptc)
        except Exception:
            logger.exception("write_metadata failed")
            self._editor_status.configure(text="Écriture échouée", text_color=palette_pair("error"))
            return
        if ok:
            self._editor_status.configure(text="Écrit", text_color=palette_pair("success"))
            self.app.toasts.show(f"Métadonnées écrites : {self._current_path.name}", kind="success")
        else:
            self._editor_status.configure(text="Échec", text_color=palette_pair("error"))

    def _editor_clear(self) -> None:
        for key in self._iptc_fields:
            self._iptc_set(key, "")
        self._editor_status.configure(text="Effacé", text_color=palette_pair("fg_muted"))

    def _iptc_set(self, key: str, value: str) -> None:
        widget = self._iptc_fields[key]
        widget.delete(0, "end")
        widget.insert(0, value)

    def _iptc_get(self, key: str) -> str:
        return self._iptc_fields[key].get()

    # ----- Panel: Analyse IA ------------------------------------------

    def _build_analyze_panel(self, parent: ctk.CTkFrame, row: int) -> None:
        # Last panel of the left column → ``sticky="nsew"`` so the
        # column stretches down to align with the right column's bottom.
        #
        # Phase F (2026-05-14, audit T-034..T-036, T-221, T-220, T-218):
        # - "Démarrer" est désormais ``state="disabled"`` par défaut et
        #   ne s'active que via ``_refresh_action_states`` quand il y a
        #   une sélection ET qu'aucun traitement n'est en cours.
        # - La barre de progression vit sur sa propre ligne, est titrée
        #   "Progression :" et utilise les couleurs accent / bg_hover
        #   pour un contraste lisible dès 0 %.
        # - ``_analyze_status`` démarre à "0 / 0 — En attente" plutôt
        #   que "Prêt", pour rendre l'état initial immédiatement
        #   compréhensible.
        section = self._panel(parent, row, "ANALYSE IA", icon="🧠", sticky="nsew")
        section.grid_rowconfigure(4, weight=1)
        section.grid_columnconfigure(0, weight=1)

        opts = ctk.CTkFrame(section, fg_color="transparent")
        opts.grid(row=1, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_XS))
        self._skip_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(opts, text="Ignorer si méta", variable=self._skip_var, font=get_font("body")).pack(side="left")
        self._write_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(opts, text="Écrire les résultats", variable=self._write_var, font=get_font("body")).pack(
            side="left", padx=SPACE_MD
        )
        self._analyze_summary = ctk.CTkLabel(
            opts, text="Aucune image", font=get_font("small"), text_color=palette_pair("fg_muted")
        )
        self._analyze_summary.pack(side="right")

        controls = ctk.CTkFrame(section, fg_color="transparent")
        controls.grid(row=2, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_XS))
        controls.grid_columnconfigure(2, weight=1)
        self._start_btn = ctk.CTkButton(
            controls,
            text="Démarrer",
            width=110,
            height=28,
            fg_color=palette_pair("accent"),
            hover_color=palette_pair("accent_hover"),
            text_color=palette_pair("accent_fg"),
            text_color_disabled=palette_pair("fg_subtle"),
            font=get_font("body_strong"),
            state="disabled",
            command=self._analyze_start,
        )
        self._start_btn.grid(row=0, column=0, padx=(0, SPACE_XS))
        self._stop_btn = ctk.CTkButton(
            controls,
            text="Arrêter",
            width=80,
            height=28,
            fg_color=palette_pair("error"),
            text_color=palette_pair("error_fg"),
            text_color_disabled=palette_pair("fg_subtle"),
            state="disabled",
            command=self._analyze_stop,
        )
        self._stop_btn.grid(row=0, column=1, padx=SPACE_XS)
        self._analyze_status = ctk.CTkLabel(
            controls,
            text="0 / 0 — En attente",
            font=get_font("small"),
            text_color=palette_pair("fg_muted"),
        )
        self._analyze_status.grid(row=0, column=2, padx=(SPACE_MD, 0), sticky="e")

        # Progress bar lives on its own row now — gives it a full-width
        # band so it remains visible even when the window is narrow,
        # and the label "Progression :" above resolves the visibility
        # complaint in T-036 ("je ne sais pas où elle est").
        progress_row = ctk.CTkFrame(section, fg_color="transparent")
        progress_row.grid(row=3, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_XS))
        progress_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            progress_row,
            text="Progression :",
            font=get_font("small"),
            text_color=palette_pair("fg_muted"),
            width=90,
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        self._analyze_progress = ctk.CTkProgressBar(
            progress_row,
            height=14,
            corner_radius=RADIUS_SM,
            progress_color=palette_pair("accent"),
            fg_color=palette_pair("bg_hover"),
            border_color=palette_pair("border"),
            border_width=1,
        )
        self._analyze_progress.set(0)
        self._analyze_progress.grid(row=0, column=1, sticky="ew", padx=(SPACE_XS, 0))

        self._analyze_results = ctk.CTkTextbox(
            section,
            font=get_font("code"),
            fg_color=palette_pair("bg"),
            text_color=palette_pair("fg"),
            border_color=palette_pair("border"),
            border_width=1,
            corner_radius=RADIUS_MD,
        )
        self._analyze_results.grid(row=4, column=0, sticky="nsew", padx=SPACE_SM, pady=(0, SPACE_SM))
        self._analyze_results.insert("1.0", "Les résultats apparaîtront ici en temps réel.\n")
        self._analyze_results.configure(state="disabled")

    # ----- Centralised action-state refresh ---------------------------

    def _refresh_action_states(self) -> None:
        """Recompute the enabled/disabled state of Démarrer/Arrêter.

        Single source of truth — every code path that could change the
        gating inputs (selection count, processing flag) calls this
        method instead of poking ``configure(state=…)`` directly.
        Eliminates the class of races where one path forgets to
        re-evaluate after a sibling mutation (cf. D-05).
        """
        try:
            n_sel = len(self.app.app_state.get("selected_paths") or [])
        except Exception:
            n_sel = 0
        processing = self._processing

        if not hasattr(self, "_start_btn"):
            return  # called before _build_analyze_panel completes
        self._start_btn.configure(state="normal" if (n_sel > 0 and not processing) else "disabled")
        self._stop_btn.configure(state="normal" if processing else "disabled")

    def _analyze_start(self) -> None:
        # Garde-fou double-clic (D-04 / T-221) : si un traitement est
        # déjà en cours, on retourne immédiatement avant toute mutation
        # d'état. Le ``state="disabled"`` du bouton est une seconde
        # protection mais ne couvre pas la fenêtre entre les deux clics
        # rapides.
        if self._processing:
            return
        api = self.app.api
        if api is None:
            self.app.toasts.show("Backend indisponible.", kind="error")
            return
        selected = list(self.app.app_state.get("selected_paths") or [])
        if not selected:
            self.app.toasts.show("Aucune image sélectionnée.", kind="warning")
            return
        self._processing = True
        self._refresh_action_states()
        self._analyze_progress.set(0)
        self._analyze_status.configure(
            text=f"0 / {fmt_int(len(selected))} — Initialisation…",
            text_color=palette_pair("fg"),
        )
        self._set_analyze_results("Initialisation…\n")
        threading.Thread(target=self._analyze_worker, args=(api, selected), daemon=True).start()

    def _analyze_worker(self, api: Any, selected: list[Path]) -> None:
        try:

            def on_progress(done: int, total: int, current: str) -> None:
                self.after(0, lambda: self._analyze_on_progress(done, total, current))

            def on_result(res: Any) -> None:
                self.after(0, lambda r=res: self._analyze_on_result(r))

            result = api.analyze_batch_ai(
                selected,
                skip_if_has_metadata=self._skip_var.get(),
                write_metadata=self._write_var.get(),
                on_progress=on_progress,
                on_result=on_result,
            )
            self.after(0, lambda r=result: self._analyze_on_complete(r))
        except Exception as e:
            logger.exception("Analyze worker failed")
            self.after(0, lambda err=str(e): self._analyze_on_failed(err))

    def _analyze_stop(self) -> None:
        api = self.app.api
        analyzer = getattr(api, "vision_analyzer", None) if api else None
        cancel = getattr(analyzer, "cancel", None)
        if callable(cancel):
            cancel()
        self._analyze_status.configure(text="Arrêt…", text_color=palette_pair("warning"))
        logger.info("Arrêt analyse demandé par l'utilisateur")

    def _analyze_on_progress(self, done: int, total: int, current: str) -> None:
        if total > 0:
            self._analyze_progress.set(done / total)
        self._analyze_status.configure(
            text=f"{fmt_int(done)} / {fmt_int(total)} — {Path(current).name if current else ''}",
            text_color=palette_pair("fg"),
        )

    def _analyze_on_result(self, res: Any) -> None:
        if isinstance(res, dict):
            ok = bool(res.get("success", True))
            path = res.get("file_path", "")
        else:
            ok = bool(getattr(res, "success", True))
            path = getattr(res, "file_path", "")
        symbol = "✓" if ok else "✗"
        self._append_analyze_results(f"{symbol} {Path(path).name}\n")

    def _analyze_on_complete(self, result: dict[str, Any]) -> None:
        self._processing = False
        self._refresh_action_states()
        self._analyze_progress.set(1)
        completed = result.get("completed", 0)
        failed = result.get("failed", 0)
        skipped = result.get("skipped", 0)
        self._append_analyze_results(
            f"\n— Terminé : {fmt_int(completed)} succès · {fmt_int(failed)} échecs · {fmt_int(skipped)} ignorés.\n"
        )
        self._analyze_status.configure(text="Terminé", text_color=palette_pair("success"))
        self.app.toasts.show(f"Analyse terminée — {fmt_int(completed)} succès.", kind="success")

    def _analyze_on_failed(self, err: str) -> None:
        self._processing = False
        self._refresh_action_states()
        self._analyze_status.configure(text="Erreur", text_color=palette_pair("error"))
        self._append_analyze_results(f"\nERREUR : {err}\n")

    def _set_analyze_results(self, text: str) -> None:
        self._analyze_results.configure(state="normal")
        self._analyze_results.delete("1.0", "end")
        self._analyze_results.insert("1.0", text)
        self._analyze_results.configure(state="disabled")

    def _append_analyze_results(self, text: str) -> None:
        self._analyze_results.configure(state="normal")
        self._analyze_results.insert("end", text)
        self._analyze_results.see("end")
        self._analyze_results.configure(state="disabled")

    # ==================================================================
    # RIGHT COLUMN — system & control
    # ==================================================================

    # ----- Panel: Modèle IA -------------------------------------------

    def _build_model_panel(self, parent: ctk.CTkFrame, row: int) -> None:
        section = self._panel(parent, row, "MODÈLE IA", icon="🤖")
        section.grid_columnconfigure(0, weight=1)

        body = ctk.CTkFrame(section, fg_color="transparent")
        body.grid(row=1, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_SM))
        body.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(body, text="Statut :", font=get_font("small"), text_color=palette_pair("fg_muted"), width=70).grid(
            row=0, column=0, sticky="w", pady=1
        )
        self._model_status_dot = ctk.CTkLabel(
            body, text="●", font=get_font("body_strong"), text_color=palette_pair("fg_subtle"), width=14
        )
        self._model_status_dot.grid(row=0, column=1, sticky="w")
        self._model_status_text = ctk.CTkLabel(
            body, text="—", font=get_font("body"), text_color=palette_pair("fg"), anchor="w"
        )
        self._model_status_text.grid(row=0, column=2, sticky="ew", padx=(SPACE_XS, 0))

        ctk.CTkLabel(body, text="URL :", font=get_font("small"), text_color=palette_pair("fg_muted"), width=70).grid(
            row=1, column=0, sticky="w", pady=1
        )
        self._model_url_label = ctk.CTkLabel(
            body, text="—", font=get_font("code"), text_color=palette_pair("fg"), anchor="w"
        )
        self._model_url_label.grid(row=1, column=1, columnspan=2, sticky="ew", pady=1)

        ctk.CTkLabel(body, text="Modèle :", font=get_font("small"), text_color=palette_pair("fg_muted"), width=70).grid(
            row=2, column=0, sticky="w", pady=1
        )
        self._model_name_label = ctk.CTkLabel(
            body, text="—", font=get_font("body_strong"), text_color=palette_pair("fg"), anchor="w"
        )
        self._model_name_label.grid(row=2, column=1, columnspan=2, sticky="ew", pady=1)

        actions = ctk.CTkFrame(section, fg_color="transparent")
        actions.grid(row=2, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_XS))
        # Phase G (2026-05-18) — bouton "▶ Démarrer Ollama" qui tente
        # de lancer le serveur local (subprocess.Popen détaché). Mis
        # en premier car c'est le pré-requis pour que Tester /
        # Configurer marchent.
        self._ollama_start_btn = ctk.CTkButton(
            actions,
            text="▶ Démarrer Ollama",
            width=150,
            height=26,
            fg_color=palette_pair("accent"),
            hover_color=palette_pair("accent_hover"),
            text_color=palette_pair("accent_fg"),
            command=self._start_ollama_server,
        )
        self._ollama_start_btn.pack(side="left", padx=(0, SPACE_XS))
        ctk.CTkButton(actions, text="Tester", width=80, height=26, command=self._model_test).pack(
            side="left", padx=SPACE_XS
        )
        ctk.CTkButton(
            actions,
            text="Configurer…",
            width=110,
            height=26,
            command=lambda: self.app.open_in_modal("ai_control"),
        ).pack(side="left", padx=SPACE_XS)

        # Phase G (2026-05-19) — la zone de feedback (résultat des tests
        # Ollama, démarrage serveur, etc.) est désormais sur sa PROPRE
        # rangée sous les boutons. Avant : packée à droite des boutons,
        # elle pouvait être tronquée sur les petites fenêtres et nuisait
        # à la lisibilité. Maintenant : ligne dédiée alignée à gauche,
        # plus de surface pour les messages longs.
        self._model_test_msg = ctk.CTkLabel(
            section,
            text="",
            font=get_font("small"),
            text_color=palette_pair("fg_muted"),
            anchor="w",
            wraplength=380,
            justify="left",
        )
        self._model_test_msg.grid(row=3, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_SM))

    def _start_ollama_server(self) -> None:
        """Lance ``ollama serve`` en process détaché (Phase G 2026-05-18).

        Cherche l'exécutable dans :
          1. le PATH (``shutil.which("ollama")``)
          2. les chemins d'installation standard Windows
             (``%LOCALAPPDATA%\\Programs\\Ollama\\ollama.exe`` et
             ``C:/Program Files/Ollama/ollama.exe``)

        Si rien n'est trouvé, affiche un toast d'erreur ; sinon
        démarre le serveur en arrière-plan (DETACHED_PROCESS sous
        Windows pour que le serveur survive à la fermeture de l'app),
        puis programme un refresh dynamique 2,5 s plus tard pour que
        le statut topbar bascule sur « En ligne ».
        """
        import shutil
        import subprocess
        import sys

        exe = shutil.which("ollama")
        if not exe:
            candidates = [
                Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
                Path("C:/Program Files/Ollama/ollama.exe"),
            ]
            for c in candidates:
                try:
                    if c.exists():
                        exe = str(c)
                        break
                except Exception:
                    continue
        if not exe:
            self.app.toasts.show(
                "Exécutable Ollama introuvable. Installez-le depuis ollama.com puis réessayez.",
                kind="error",
                timeout_ms=6000,
            )
            return

        self._ollama_start_btn.configure(state="disabled", text="Démarrage…")
        try:
            kwargs: dict[str, Any] = {
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "stdin": subprocess.DEVNULL,
            }
            if sys.platform == "win32":
                # DETACHED_PROCESS + CREATE_NEW_PROCESS_GROUP : le serveur
                # ne meurt pas avec l'app et ne reçoit pas ses Ctrl+C.
                kwargs["creationflags"] = (
                    subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
                    | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
                )
            subprocess.Popen([exe, "serve"], **kwargs)
            self.app.toasts.show("Démarrage du serveur Ollama…", kind="info", timeout_ms=3000)
            # Laisser ~2,5 s au serveur pour ouvrir le port 11434, puis
            # rafraîchir le statut (qui mettra à jour la chip topbar).
            self.after(2500, self._refresh_dynamic_async)
            self.after(3000, lambda: self._ollama_start_btn.configure(state="normal", text="▶ Démarrer Ollama"))
        except Exception as exc:
            logger.exception("Démarrage Ollama échoué")
            self.app.toasts.show(f"Échec démarrage Ollama : {exc}", kind="error")
            self._ollama_start_btn.configure(state="normal", text="▶ Démarrer Ollama")

    def _model_test(self) -> None:
        api = self.app.api
        if api is None:
            self._model_test_msg.configure(text="Backend absent", text_color=palette_pair("warning"))
            return
        self._model_test_msg.configure(text="Test…", text_color=palette_pair("warning"))
        threading.Thread(target=self._model_test_worker, args=(api,), daemon=True).start()

    def _model_test_worker(self, api: Any) -> None:
        try:
            if not hasattr(api, "ollama_client"):
                api.init_ai()
            result = api.ollama_client.test_connection()
        except Exception as e:
            logger.exception("model test failed")
            self.after(
                0,
                lambda err=str(e): self._model_test_msg.configure(
                    text=f"Échec : {err[:30]}", text_color=palette_pair("error")
                ),
            )
            return
        if result.get("success"):
            ms = result.get("response_time_ms", 0)
            self.after(
                0, lambda m=ms: self._model_test_msg.configure(text=f"OK · {m} ms", text_color=palette_pair("success"))
            )
        else:
            self.after(
                0,
                lambda r=result: self._model_test_msg.configure(
                    text=f"Échec : {r.get('message', '')[:30]}", text_color=palette_pair("error")
                ),
            )

    # ----- Panel: Validation ------------------------------------------

    def _build_validate_panel(self, parent: ctk.CTkFrame, row: int) -> None:
        section = self._panel(parent, row, "VALIDATION", icon="✓")
        section.grid_columnconfigure(0, weight=1)

        self._validate_summary = ctk.CTkLabel(
            section,
            text="Aucun scan",
            font=get_font("body"),
            text_color=palette_pair("fg"),
            anchor="w",
        )
        self._validate_summary.grid(row=1, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_XS))

        self._validate_detail = ctk.CTkLabel(
            section,
            text="—",
            font=get_font("small"),
            text_color=palette_pair("fg_muted"),
            anchor="w",
            wraplength=380,
            justify="left",
        )
        self._validate_detail.grid(row=2, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_XS))

        actions = ctk.CTkFrame(section, fg_color="transparent")
        actions.grid(row=3, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_SM))
        ctk.CTkButton(
            actions,
            text="Lancer",
            width=90,
            height=26,
            fg_color=palette_pair("accent"),
            hover_color=palette_pair("accent_hover"),
            text_color=palette_pair("accent_fg"),
            command=self._validate_run,
        ).pack(side="left", padx=(0, SPACE_XS))
        ctk.CTkButton(
            actions,
            text="Détail…",
            width=90,
            height=26,
            command=lambda: self.app.open_in_modal("validate"),
        ).pack(side="left", padx=SPACE_XS)

    def _validate_run(self) -> None:
        api = self.app.api
        if api is None:
            self._validate_summary.configure(text="Backend indisponible", text_color=palette_pair("warning"))
            return
        files = list(self.app.app_state.get("scanned_images") or [])
        if not files:
            self._validate_summary.configure(text="Scannez d'abord un dossier", text_color=palette_pair("warning"))
            return
        self._validate_summary.configure(
            text=f"Validation de {fmt_int(len(files))} images…", text_color=palette_pair("warning")
        )
        threading.Thread(target=self._validate_worker, args=(api, files), daemon=True).start()

    def _validate_worker(self, api: Any, files: list[Path]) -> None:
        ok = ko = 0
        first_issue = ""
        for f in files:
            try:
                res = api.validate_image(f)
                if getattr(res, "is_valid", False):
                    ok += 1
                else:
                    ko += 1
                    if not first_issue:
                        errs = list(getattr(res, "errors", []))
                        first_issue = f"{f.name} : {errs[0] if errs else 'invalide'}"
            except Exception:
                ko += 1
        self.after(0, lambda: self._validate_done(ok, ko, first_issue))

    def _validate_done(self, ok: int, ko: int, first_issue: str) -> None:
        total = ok + ko
        if ko == 0 and total > 0:
            self._validate_summary.configure(
                text=f"{fmt_int(total)} images · toutes conformes ✓", text_color=palette_pair("success")
            )
        else:
            self._validate_summary.configure(
                text=f"{fmt_int(total)} images · {fmt_int(ok)} OK · {fmt_int(ko)} à corriger",
                text_color=palette_pair("warning") if ko else palette_pair("success"),
            )
        self._validate_detail.configure(text=first_issue or "Aucune anomalie")

    # ----- Panel: Historique ------------------------------------------

    def _build_history_panel(self, parent: ctk.CTkFrame, row: int) -> None:
        section = self._panel(parent, row, "HISTORIQUE", icon="🕐")
        section.grid_columnconfigure(0, weight=1)
        section.grid_rowconfigure(2, weight=1)

        self._history_summary = ctk.CTkLabel(
            section, text="—", font=get_font("body"), text_color=palette_pair("fg"), anchor="w"
        )
        self._history_summary.grid(row=1, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_XS))

        tail = ctk.CTkFrame(section, fg_color=palette_pair("bg"), corner_radius=RADIUS_MD)
        tail.grid(row=2, column=0, sticky="nsew", padx=SPACE_SM, pady=(0, SPACE_XS))
        tail.grid_columnconfigure(0, weight=1)
        self._history_lines: list[ctk.CTkLabel] = []
        for i in range(HISTORY_TAIL):
            label = ctk.CTkLabel(tail, text="", font=get_font("code"), text_color=palette_pair("fg_muted"), anchor="w")
            label.grid(row=i, column=0, sticky="ew", padx=SPACE_SM, pady=0)
            self._history_lines.append(label)

        actions = ctk.CTkFrame(section, fg_color="transparent")
        actions.grid(row=3, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_SM))
        ctk.CTkButton(
            actions,
            text="Tout voir…",
            width=100,
            height=26,
            command=lambda: self.app.open_in_modal("audit"),
        ).pack(side="left", padx=(0, SPACE_XS))
        ctk.CTkButton(
            actions,
            text="Exporter…",
            width=100,
            height=26,
            command=self._history_export,
        ).pack(side="left", padx=SPACE_XS)
        # Phase G+3 (2026-05-19) — bouton « Vider » destructif à droite
        # d'Exporter. Couleurs error pour signaler l'effet, confirm()
        # destructive avant tout DELETE en base.
        ctk.CTkButton(
            actions,
            text="Vider",
            width=80,
            height=26,
            fg_color=palette_pair("error"),
            text_color=palette_pair("error_fg"),
            command=self._history_clear,
        ).pack(side="left", padx=SPACE_XS)

    def _history_export(self) -> None:
        api = self.app.api
        if api is None:
            return
        path = filedialog.asksaveasfilename(
            title="Exporter le journal",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("CSV", "*.csv")],
        )
        if not path:
            return
        out = Path(path)
        try:
            count = api.database.export_audit_log(out, format="csv" if out.suffix.lower() == ".csv" else "json")
            self.app.toasts.show(f"{fmt_int(count)} entrée(s) exportée(s).", kind="success")
        except Exception as e:
            logger.exception("export failed")
            self.app.toasts.show(f"Échec : {e}", kind="error")

    def _history_clear(self) -> None:
        """Phase G+3 — purge complète de la table ``audit_log`` après
        confirmation destructive. Refresh ensuite les indicateurs du
        panneau Historique pour refléter l'état vide.
        """
        api = self.app.api
        if api is None:
            self.app.toasts.show("Backend indisponible.", kind="error")
            return
        if not self.app.confirm_destructive(
            title="Vider l'historique",
            message=(
                "Cette action supprime DÉFINITIVEMENT toutes les entrées "
                "du journal d'audit (analyses, écritures de métadonnées, "
                "erreurs). Cette opération est irréversible. Continuer ?"
            ),
        ):
            return
        try:
            count = api.database.clear_audit_log()
        except Exception as e:
            logger.exception("clear_audit_log failed")
            self.app.toasts.show(f"Échec : {e}", kind="error")
            return
        self.app.toasts.show(f"{fmt_int(count)} entrée(s) supprimée(s).", kind="success")
        # Force un refresh immédiat du panneau (compteurs + tail des
        # 5 dernières lignes) sans attendre le poll automatique de 5 s.
        try:
            self._refresh_dynamic_async()
        except Exception:
            logger.debug("history refresh after clear failed", exc_info=True)

    # ----- Panel: Paramètres ------------------------------------------

    def _build_settings_panel(self, parent: ctk.CTkFrame, row: int) -> None:
        # Last panel of the right column → ``sticky="nsew"`` so the
        # column reaches the same y-bottom as the left column even
        # though it has 4 panels vs the left's 3.
        section = self._panel(parent, row, "PARAMÈTRES", icon="⚙", sticky="nsew")
        section.grid_columnconfigure(0, weight=1)

        body = ctk.CTkFrame(section, fg_color="transparent")
        body.grid(row=1, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_XS))
        body.grid_columnconfigure((0, 1), weight=1)
        self._settings_chips: dict[str, ctk.CTkLabel] = {}
        items = [
            ("workers", "Workers"),
            ("batch", "Batch"),
            ("model", "Modèle"),
            ("backup", "Backup _orig"),
            ("write_iptc", "IPTC"),
            ("write_xmp", "XMP"),
        ]
        for i, (key, label) in enumerate(items):
            row_f = ctk.CTkFrame(body, fg_color="transparent")
            row_f.grid(row=i // 2, column=i % 2, sticky="ew", padx=2, pady=1)
            ctk.CTkLabel(
                row_f, text=label, font=get_font("small"), text_color=palette_pair("fg_muted"), anchor="w"
            ).pack(side="left", padx=(0, SPACE_XS))
            value = ctk.CTkLabel(
                row_f, text="—", font=get_font("body_strong"), text_color=palette_pair("fg"), anchor="w"
            )
            value.pack(side="left")
            self._settings_chips[key] = value

        actions = ctk.CTkFrame(section, fg_color="transparent")
        actions.grid(row=2, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_SM))
        ctk.CTkButton(
            actions,
            text="Modifier…",
            width=120,
            height=26,
            command=lambda: self.app.open_in_modal("settings"),
        ).pack(side="left")

    # ==================================================================
    # Auto-refresh of the right-column live indicators
    # ==================================================================

    def on_enter(self, **_kwargs: Any) -> None:
        self._refresh()

    def on_leave(self) -> None:
        if self._refresh_after_id is not None:
            try:
                self.after_cancel(self._refresh_after_id)
            except Exception:
                pass
            self._refresh_after_id = None

    def _refresh(self) -> None:
        self._refresh_settings_chips()
        self._refresh_dynamic_async()
        self._refresh_after_id = self.after(REFRESH_INTERVAL_MS, self._refresh)

    def _refresh_settings_chips(self) -> None:
        api = self.app.api
        defaults = {
            "workers": 4,
            "batch": 50,
            "model": "—",
            "backup": True,
            "write_iptc": True,
            "write_xmp": True,
        }
        getter = (lambda k, d: api.get_setting(k, d)) if api else (lambda _k, d: d)
        self._settings_chips["workers"].configure(text=str(int(getter("max_workers", defaults["workers"]))))
        self._settings_chips["batch"].configure(text=str(int(getter("batch_size", defaults["batch"]))))
        self._settings_chips["model"].configure(text=str(getter("ollama_model", defaults["model"]))[:18])
        self._settings_chips["backup"].configure(
            text="Oui" if bool(getter("create_backup", defaults["backup"])) else "Non"
        )
        self._settings_chips["write_iptc"].configure(
            text="Oui" if bool(getter("write_iptc", defaults["write_iptc"])) else "Non"
        )
        self._settings_chips["write_xmp"].configure(
            text="Oui" if bool(getter("write_xmp", defaults["write_xmp"])) else "Non"
        )

    def _refresh_dynamic_async(self) -> None:
        api = self.app.api
        if api is None:
            self._set_model_status("muted", "Backend absent", "—", "—")
            self._set_history_summary(0, 0, [])
            # Phase G (2026-05-18) — sync chip topbar Ollama avec l'état
            # "Backend absent" pour ne pas garder un statut périmé.
            try:
                self.app.set_ollama_health("—", "muted")
            except Exception:
                pass
            return
        threading.Thread(target=self._refresh_dynamic_worker, args=(api,), daemon=True).start()

    def _refresh_dynamic_worker(self, api: Any) -> None:
        try:
            ai_status = api.check_ai_status() if hasattr(api, "check_ai_status") else {}
        except Exception:
            logger.exception("ai status failed")
            ai_status = {"available": False, "message": "erreur"}
        try:
            since = datetime.now() - timedelta(hours=24)
            logs = api.database.get_audit_logs(start_date=since, limit=HISTORY_TAIL)
            since_24h = datetime.now() - timedelta(hours=24)
            all_24h = api.database.get_audit_logs(start_date=since_24h, limit=10_000)
            n_ops = len(all_24h)
            n_err = sum(1 for log in all_24h if not log.success)
        except Exception:
            logger.exception("audit fetch failed")
            logs = []
            n_ops = n_err = 0

        url = api.get_setting("ollama_url", "—")
        if ai_status.get("available"):
            kind = "success"
            status_text = f"En ligne · {ai_status.get('version', '')}".strip()
            current_model = ai_status.get("current_model") or "(aucun chargé)"
            chip_label = "En ligne"
        elif ai_status.get("status") == "not_initialized":
            kind = "muted"
            status_text = "Non initialisé"
            current_model = "—"
            chip_label = "Non init."
        else:
            kind = "warning"
            status_text = ai_status.get("message", "Hors ligne") or "Hors ligne"
            current_model = "—"
            chip_label = "Hors ligne"

        self.after(0, lambda: self._set_model_status(kind, status_text, url, current_model))
        self.after(0, lambda lg=logs, no=n_ops, ne=n_err: self._set_history_summary(no, ne, lg))
        # Phase G (2026-05-18) — pousser le statut Ollama vers la
        # chip topbar via App.set_ollama_health(). Le chip se met à
        # jour à chaque tick du refresh dynamique (toutes les 5 s).
        self.after(0, lambda lbl=chip_label, k=kind: self.app.set_ollama_health(lbl, k))

    def _set_model_status(self, kind: str, status_text: str, url: str, model: str) -> None:
        color = {
            "success": palette_pair("success"),
            "warning": palette_pair("warning"),
            "error": palette_pair("error"),
            "muted": palette_pair("fg_muted"),
        }[kind]
        self._model_status_dot.configure(text_color=color)
        self._model_status_text.configure(text=status_text, text_color=color)
        self._model_url_label.configure(text=url)
        self._model_name_label.configure(text=model)

    def _set_history_summary(self, n_ops: int, n_err: int, logs: list[Any]) -> None:
        text = f"{fmt_int(n_ops)} opérations / 24 h · {fmt_int(n_err)} erreur(s)"
        self._history_summary.configure(
            text=text,
            text_color=palette_pair("warning") if n_err else palette_pair("fg"),
        )
        for i, label in enumerate(self._history_lines):
            if i < len(logs):
                log = logs[i]
                ts = log.timestamp.strftime("%H:%M:%S")
                action = log.action_type.value
                fname = Path(log.file_path).name if log.file_path else "—"
                if len(fname) > 22:
                    fname = fname[:19] + "…"
                ok = "✓" if log.success else "✗"
                color = palette_pair("success") if log.success else palette_pair("error")
                label.configure(text=f"{ts}  {ok} {action:<14} {fname}", text_color=color)
            else:
                label.configure(text="", text_color=palette_pair("fg_muted"))

    # ==================================================================
    # Public API consumed by App._focus_workspace_panel (Ctrl+1..3)
    # ==================================================================

    def focus_panel(self, name: str) -> None:
        """Bring a workspace panel into focus.

        Called by the Ctrl+1..3 shortcuts via ``App._focus_workspace_panel``.
        Strategy: focus a meaningful widget inside the target panel
        (entry, primary button, first form field). If the panel is the
        Editor IPTC and is collapsed, expand it first — the shortcut
        is meant to "go to" the panel so a collapsed state would
        defeat the user's intent.

        We also try to scroll the surrounding ``CTkScrollableFrame``
        so the focused widget is within the viewport on small windows;
        any failure there is swallowed because the focus alone is the
        contract.
        """
        targets: dict[str, Any] = {
            "sources": getattr(self, "_folder_entry", None),
            "editor": None,  # resolved below after potential expansion
            "analyze": getattr(self, "_start_btn", None),
        }
        if name == "editor":
            if self._editor_collapsed:
                try:
                    self._toggle_editor_collapsed()
                except Exception:
                    logger.exception("Could not auto-expand editor on Ctrl+2")
            # Pick the first IPTC entry that exists.
            for key in ("headline", "caption", "keywords", "byline", "copyright_notice"):
                widget = self._iptc_fields.get(key)
                if widget is not None:
                    targets["editor"] = widget
                    break

        widget = targets.get(name)
        if widget is None:
            logger.debug("focus_panel: unknown panel %r", name)
            return
        try:
            widget.focus_set()
        except Exception:
            logger.exception("focus_set failed for panel %r", name)
        # Best-effort scroll into view — silent fallback if the
        # surrounding container isn't the expected scrollable frame.
        self._scroll_widget_into_view(widget)

    def _scroll_widget_into_view(self, widget: Any) -> None:
        """Scroll the parent ``CTkScrollableFrame`` so *widget* is visible.

        Used as a best-effort by ``focus_panel``. If the widget is
        already visible or the scrollable infrastructure isn't there,
        this is a no-op. Errors are swallowed: the scroll is a nicety,
        not a contract.
        """
        try:
            self.update_idletasks()
            # ``self.master`` is the CTkScrollableFrame's inner frame;
            # ``self.master._parent_canvas`` is the actual canvas.
            canvas = getattr(self.master, "_parent_canvas", None)
            if canvas is None:
                return
            canvas.update_idletasks()
            wy = widget.winfo_rooty() - canvas.winfo_rooty()
            ch = canvas.winfo_height()
            if 0 <= wy <= ch - widget.winfo_height():
                return  # already visible
            bbox = canvas.bbox("all")
            if not bbox or bbox[3] <= bbox[1]:
                return
            total = bbox[3] - bbox[1]
            target_top = canvas.canvasy(0) + wy - 20  # 20px breathing room
            canvas.yview_moveto(max(0.0, min(1.0, target_top / total)))
        except Exception:
            # Scroll is best-effort. Focus already happened, that's enough.
            pass

    # ==================================================================
    # Helpers

    def _panel(
        self,
        parent: ctk.CTkFrame,
        row: int,
        title: str,
        *,
        bg_key: str = "bg_elevated",
        icon: str | None = None,
        sticky: str = "new",
    ) -> ctk.CTkFrame:
        """Create a titled panel frame.

        ``bg_key`` selects which palette key drives the background; the
        default ``bg_elevated`` gives the standard "card on canvas" look.
        Pass ``bg_key="bg_deep"`` to make a panel sink below the canvas
        in dark mode (used by Sources for a workspace-floor feel).

        ``icon`` (optional, e.g. ``"🧠"``) is rendered immediately to
        the *left* of the title — top-left of the panel — so the panel
        type is identifiable at a glance and the title stays anchored
        to the upper-left corner (per the v3 UI spec).

        ``sticky`` defaults to ``"new"`` (top-anchored, fills width);
        pass ``"nsew"`` for the LAST panel of a column so it stretches
        and pulls the column's bottom edge into alignment with its
        sibling column.
        """
        frame = ctk.CTkFrame(
            parent,
            fg_color=palette_pair(bg_key),
            border_color=palette_pair("border"),
            border_width=1,
            corner_radius=RADIUS_MD,
        )
        frame.grid(row=row, column=0, sticky=sticky, pady=(0, SPACE_SM))
        frame.grid_columnconfigure(0, weight=1)

        # Title row — icon (if any) + label, anchored top-left so every
        # panel's identity sits in the same corner. The header uses a
        # nested transparent frame so the icon and title share a single
        # baseline, and so an optional trailing widget (e.g. a chevron
        # for collapsible panels) can still grid into ``frame`` row 0
        # without colliding.
        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=SPACE_SM, pady=(SPACE_SM, SPACE_XS))
        header.grid_columnconfigure(1, weight=1)
        if icon:
            ctk.CTkLabel(
                header,
                text=icon,
                font=get_font("body_strong"),
                text_color=palette_pair("fg_muted"),
                width=18,
                anchor="w",
            ).grid(row=0, column=0, sticky="w", padx=(0, SPACE_XS))
        ctk.CTkLabel(
            header,
            text=title,
            font=get_font("small"),
            text_color=palette_pair("fg_subtle"),
            anchor="w",
        ).grid(row=0, column=1, sticky="w")
        return frame


_ = SPACE_LG  # keep imported constant for future use
