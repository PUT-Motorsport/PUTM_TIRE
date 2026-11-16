"""Unified Import Hub & Export Studio dialogs for PUTM Tire Lab.

Consolidates all file loading (.mat/.csv/.txt/.dat/.tir/presets) into a single
Import window and all outputs (.tir/.xlsx/.csv/.json/plots) into a single Export
window where engineers customize formatting and preview how the output looks.
"""

from __future__ import annotations

import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from condition_export_dialog import ConditionExportDialog
from fy_excel_dialog import FyExcelDialog
from mf51_model import MF51Model
from tir_parser import load_tir_to_model
from tire_scaler import scale_18x6_to_16x75_r20
from ui_theme import (
    FONT_BOLD,
    FONT_MONO,
    FONT_SMALL,
    FONT_SMALL_BOLD,
    PALETTE,
    ModernCard,
    apply_ttk_theme,
    load_brand_logo,
)


def _format_preview_csv(df: pd.DataFrame, sep: str, max_rows: int = 12) -> str:
    if df is None or df.empty:
        return "(No rows available for preview)"
    head = df.head(max_rows)
    lines = [head.to_csv(sep=sep, index=False)]
    if len(df) > max_rows:
        lines.append(f"... ({len(df):,} total rows, {len(df.columns)} columns)")
    return "\n".join(lines).strip()


# ==============================================================================
# UNIFIED IMPORT HUB
# ==============================================================================
class UnifiedImportDialog(tk.Toplevel):
    """Single centralized modal for all telemetry, .TIR, preset, and fit imports."""

    SOURCES = [
        ("telemetry", "Telemetry Data (.mat / .csv / .txt)", "Queue runs in Step 1 or load Stiffness Tire A/B"),
        ("tir", "Tire Property File (.tir)", "Load PAC2002 / MF 5.1 into Active Model or Slots 2–4"),
        ("presets", "Built-in Presets & FSAE Demo", "Hoosier R25B / 16x7.5 R20 models & sample TTC data"),
        ("sync", "Sync & Extract MF 5.1", "Apply Step 2 fits or run 1-Click Autopilot extraction"),
    ]

    def __init__(self, parent, app, *, initial_mode: str = "telemetry"):
        super().__init__(parent)
        self.app = app
        self.c = app.c
        self._mode = tk.StringVar(value=initial_mode if initial_mode in {k for k, _, _ in self.SOURCES} else "telemetry")
        self._cat_buttons: Dict[str, dict] = {}
        self._last_tir_preview: Optional[Tuple[str, MF51Model, Dict[str, str]]] = None

        # Telemetry options
        self.var_telem_target = tk.StringVar(value="step1_corn")
        self.var_jump_to_target = tk.BooleanVar(value=True)
        self.var_auto_process = tk.BooleanVar(value=True)

        # .TIR target
        self.var_tir_target = tk.StringVar(value="active")

        self.title("PUTM Tire Lab — Unified Import Hub")
        self.geometry("820x560")
        self.minsize(700, 480)
        self.configure(bg=self.c["bg"])
        apply_ttk_theme(self, self.c)

        self._build_ui()
        self._select_category(self._mode.get())

    def _build_ui(self):
        c = self.c

        # Header
        hdr = tk.Frame(self, bg=c["bg_header"], height=48, padx=14)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        self._logo = load_brand_logo(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "img", "logo2.png"),
            max_size=(26, 26),
        )
        if self._logo is not None:
            tk.Label(hdr, image=self._logo, bg=c["bg_header"]).pack(side="left", padx=(0, 8))

        tk.Label(
            hdr,
            text="UNIFIED IMPORT HUB — DATA, TIRE MODELS (.TIR) & PRESETS",
            bg=c["bg_header"],
            fg=c["text"],
            font=("Segoe UI Semibold", 11, "bold"),
        ).pack(side="left")

        tk.Frame(self, bg=c["border"], height=1).pack(fill="x")

        body = tk.Frame(self, bg=c["bg"], padx=12, pady=12)
        body.pack(fill="both", expand=True)

        # Left category list
        left = tk.Frame(body, bg=c["panel"], width=265, highlightbackground=c["border"], highlightthickness=1)
        left.pack(side="left", fill="y", padx=(0, 12))
        left.pack_propagate(False)

        tk.Label(
            left,
            text="IMPORT SOURCE",
            bg=c["panel_alt"],
            fg=c["cyan"],
            font=FONT_SMALL_BOLD,
            anchor="w",
            padx=12,
            pady=8,
        ).pack(fill="x")
        tk.Frame(left, bg=c["border"], height=1).pack(fill="x")

        for key, title, sub in self.SOURCES:
            row = tk.Frame(left, bg=c["panel"], cursor="hand2")
            row.pack(fill="x", pady=2)
            bar = tk.Frame(row, bg=c["panel"], width=3)
            bar.pack(side="left", fill="y")
            inner = tk.Frame(row, bg=c["panel"], padx=10, pady=8)
            inner.pack(side="left", fill="both", expand=True)
            lbl_t = tk.Label(inner, text=title, bg=c["panel"], fg=c["text"], font=FONT_BOLD, anchor="w")
            lbl_t.pack(fill="x")
            lbl_s = tk.Label(inner, text=sub, bg=c["panel"], fg=c["text_muted"], font=FONT_SMALL, anchor="w", wraplength=220, justify="left")
            lbl_s.pack(fill="x", pady=(2, 0))

            self._cat_buttons[key] = {"row": row, "bar": bar, "inner": inner, "title": lbl_t, "sub": lbl_s}
            for w in (row, bar, inner, lbl_t, lbl_s):
                w.bind("<Button-1>", lambda _e, k=key: self._select_category(k))

        # Right details panel
        self.right_area = tk.Frame(body, bg=c["bg"])
        self.right_area.pack(side="right", fill="both", expand=True)

        # Bottom footer bar
        tk.Frame(self, bg=c["border"], height=1).pack(fill="x")
        foot = tk.Frame(self, bg=c["bg_sidebar"], padx=14, pady=8)
        foot.pack(fill="x")
        self.lbl_import_feedback = tk.Label(
            foot,
            text="Choose an import category on the left.",
            bg=c["bg_sidebar"],
            fg=c["text_secondary"],
            font=FONT_SMALL,
            anchor="w",
        )
        self.lbl_import_feedback.pack(side="left", fill="x", expand=True)
        ttk.Button(foot, text="Close", command=self.destroy).pack(side="right")

    def _select_category(self, key: str):
        c = self.c
        self._mode.set(key)
        for k, widgets in self._cat_buttons.items():
            active = (k == key)
            bg = c["panel_elevated"] if active else c["panel"]
            bar_bg = c["cyan"] if active else c["panel"]
            fg_t = c["cyan"] if active else c["text"]
            for name in ("row", "inner", "title", "sub"):
                widgets[name].configure(bg=bg)
            widgets["bar"].configure(bg=bar_bg)
            widgets["title"].configure(fg=fg_t)

        for child in self.right_area.winfo_children():
            child.destroy()

        if key == "telemetry":
            self._render_telemetry_panel()
        elif key == "tir":
            self._render_tir_panel()
        elif key == "presets":
            self._render_presets_panel()
        else:
            self._render_sync_panel()

    def _set_feedback(self, msg: str, color_key: str = "ok"):
        self.lbl_import_feedback.config(text=msg, fg=self.c.get(color_key, self.c["text_secondary"]))

    def _render_telemetry_panel(self):
        c = self.c
        card = ModernCard(
            self.right_area,
            "Import Telemetry Measurements (.MAT, .CSV, .TXT, .DAT)",
            accent_color=c["cyan"],
            colors=c,
            pad_x=12,
            pad_y=10,
        )
        card.pack(fill="both", expand=True)

        tk.Label(
            card.body,
            text="Choose where the imported measurement files should be loaded:",
            bg=c["panel"],
            fg=c["text_secondary"],
            font=FONT_SMALL_BOLD,
            anchor="w",
        ).pack(fill="x", pady=(0, 8))

        targets = [
            ("step1_corn", "Step 1 Workflow Queue — Cornering (Fy / Mz)  [uses Step 1 Fz/IA filters]"),
            ("step1_drive", "Step 1 Workflow Queue — Drive / Brake (Fx / SL)"),
            ("stiff_a", "Cornering Stiffness Analyzer (Tool 6) — Tire A (Base, full unfiltered)"),
            ("stiff_b", "Cornering Stiffness Analyzer (Tool 6) — Tire B (Compare, full unfiltered)"),
        ]
        for val, lbl in targets:
            ttk.Radiobutton(card.body, text=lbl, value=val, variable=self.var_telem_target).pack(anchor="w", pady=3)

        opts = tk.Frame(card.body, bg=c["panel"])
        opts.pack(fill="x", pady=(10, 10))
        ttk.Checkbutton(opts, text="Auto-process Step 1 queue immediately after adding files", variable=self.var_auto_process).pack(anchor="w", pady=2)
        ttk.Checkbutton(opts, text="Switch main workspace to target view after import", variable=self.var_jump_to_target).pack(anchor="w", pady=2)

        ttk.Button(
            card.body,
            text="Browse & Import Telemetry Files...",
            style="Cyan.TButton",
            command=self._action_import_telemetry,
        ).pack(fill="x", pady=(4, 10))

        self.txt_telem_log = tk.Text(
            card.body,
            height=9,
            bg=c["input"],
            fg=c["text_secondary"],
            font=FONT_MONO,
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=c["border"],
            wrap="word",
        )
        self.txt_telem_log.pack(fill="both", expand=True)
        self.txt_telem_log.insert(
            "1.0",
            "Supported formats:\n"
            "• MATLAB TTC files (.mat, including structs & scalar P/IA constants)\n"
            "• Delimited text / CSV / Calspan ASCII (.csv, .txt, .dat, .tsv)\n"
            "• Handles unit rows, metadata headers, and comma decimal CSVs automatically.",
        )
        self.txt_telem_log.config(state="disabled")

    def _action_import_telemetry(self):
        target = self.var_telem_target.get()
        paths = filedialog.askopenfilenames(
            parent=self,
            title="Select Telemetry Files (.mat / .csv / .txt / .dat)",
            filetypes=[
                ("Tire Telemetry (*.mat;*.csv;*.txt;*.dat;*.tsv)", "*.mat *.csv *.txt *.dat *.tsv"),
                ("MATLAB TTC (*.mat)", "*.mat"),
                ("CSV / ASCII (*.csv;*.txt;*.dat)", "*.csv *.txt *.dat *.tsv"),
                ("All Files", "*.*"),
            ],
        )
        if not paths:
            return

        log_lines: List[str] = []
        if target in ("stiff_a", "stiff_b"):
            tire = "A" if target == "stiff_a" else "B"
            cnt, errs = self.app.stiffness_tab.load_files(tire, paths, show_errors=False)
            log_lines.append(f"Loaded {cnt} file(s) into Cornering Stiffness Tire {tire}.")
            if errs:
                log_lines.extend(f"ERROR: {e}" for e in errs)
            if cnt > 0:
                self._set_feedback(f"Imported {cnt} file(s) into Stiffness Tire {tire}.", "ok" if not errs else "warn")
                if self.var_jump_to_target.get():
                    self.app.switch_tab("stiffness")
            else:
                self._set_feedback("Import failed — inspect errors below.", "danger")
        else:
            rt = "Cornering" if target == "step1_corn" else "Drive/Brake"
            self.app.var_run_type.set(rt)
            self.app._seg_run_type.refresh()
            self.app._on_run_type_toggle()
            added = self.app.add_runs_from_paths(paths, run_type=rt, process_now=bool(self.var_auto_process.get()))
            log_lines.append(f"Queued {added} file(s) as {rt} in Step 1.")
            if added:
                self._set_feedback(f"Added {added} file(s) to Step 1 ({rt}).", "ok")
                if self.var_jump_to_target.get():
                    self.app.switch_tab("data")
            else:
                self._set_feedback("No files were added (check filter values).", "warn")

        self.txt_telem_log.config(state="normal")
        self.txt_telem_log.delete("1.0", "end")
        self.txt_telem_log.insert("1.0", "\n".join(log_lines))
        self.txt_telem_log.config(state="disabled")

    def _render_tir_panel(self):
        c = self.c
        card = ModernCard(
            self.right_area,
            "Import .TIR Tire Property File (PAC2002 / MF 5.1)",
            accent_color=c["accent"],
            colors=c,
            pad_x=12,
            pad_y=10,
        )
        card.pack(fill="both", expand=True)

        tk.Label(
            card.body,
            text="Select target destination for the .TIR model:",
            bg=c["panel"],
            fg=c["text_secondary"],
            font=FONT_SMALL_BOLD,
            anchor="w",
        ).pack(fill="x", pady=(0, 6))

        for val, lbl in [
            ("active", "Active Model (Step 3 Editor, Step 4 3D Sim & Comparator Slot 1)"),
            ("slot1", "Comparator Slot 2 (Benchmark Tire 2)"),
            ("slot2", "Comparator Slot 3 (Benchmark Tire 3)"),
            ("slot3", "Comparator Slot 4 (Benchmark Tire 4)"),
        ]:
            ttk.Radiobutton(card.body, text=lbl, value=val, variable=self.var_tir_target).pack(anchor="w", pady=2)

        ttk.Button(
            card.body,
            text="Browse & Load .TIR File...",
            style="Accent.TButton",
            command=self._action_import_tir,
        ).pack(fill="x", pady=(10, 10))

        self.txt_tir_preview = tk.Text(
            card.body,
            height=9,
            bg=c["input"],
            fg=c["text_secondary"],
            font=FONT_MONO,
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=c["border"],
            wrap="word",
        )
        self.txt_tir_preview.pack(fill="both", expand=True)
        self.txt_tir_preview.insert("1.0", "Browse for any TYDEX / PAC2002 .tir file to inspect its parameters and apply it.")
        self.txt_tir_preview.config(state="disabled")

    def _action_import_tir(self):
        path = filedialog.askopenfilename(
            parent=self,
            title="Select Pacejka .TIR Property File",
            filetypes=[("ADAMS / CarSim TIR (*.tir)", "*.tir"), ("All Files", "*.*")],
        )
        if not path:
            return
        try:
            model, meta = load_tir_to_model(path)
        except Exception as exc:
            messagebox.showerror(".TIR Read Error", f"Could not parse .tir file:\n{exc}", parent=self)
            self._set_feedback(f"Failed to read {os.path.basename(path)}", "danger")
            return

        name = meta.get("TIRE_NAME") or os.path.splitext(os.path.basename(path))[0]
        dest = self.var_tir_target.get()
        if dest == "active":
            self.app.apply_loaded_tir_model(model, name, path)
            dest_desc = "Active Model (Step 3 / Step 4 / Slot 1)"
        else:
            slot_idx = int(dest.replace("slot", ""))
            self.app.tir_tab.load_tir_path_to_slot(slot_idx, path)
            self.app.switch_tab("compare")
            dest_desc = f"Comparator Slot {slot_idx + 1}"

        summary = (
            f"Loaded .TIR -> {dest_desc}\n"
            f"Tire Name : {name}\n"
            f"Geometry  : R0 = {model.get('UNLOADED_RADIUS'):.4f} m, W = {model.get('WIDTH'):.4f} m, FNOMIN = {model.get('FNOMIN'):.0f} N\n"
            f"Scaling   : LMUX = {model.get('LMUX'):.2f}, LMUY = {model.get('LMUY'):.2f}\n"
            f"Lateral   : PDY1 = {model.get('PDY1'):.3f}, PDY2 = {model.get('PDY2'):.4f}, PKY1 = {model.get('PKY1'):.2f}\n"
            f"Long.     : PDX1 = {model.get('PDX1'):.3f}, PDX2 = {model.get('PDX2'):.4f}, PKX1 = {model.get('PKX1'):.2f}"
        )
        self.txt_tir_preview.config(state="normal")
        self.txt_tir_preview.delete("1.0", "end")
        self.txt_tir_preview.insert("1.0", summary)
        self.txt_tir_preview.config(state="disabled")
        self._set_feedback(f"Loaded '{name}' into {dest_desc}.", "ok")

    def _render_presets_panel(self):
        c = self.c
        card = ModernCard(
            self.right_area,
            "Built-In FSAE Presets & Sample TTC Data",
            accent_color=c["ok"],
            colors=c,
            pad_x=12,
            pad_y=10,
        )
        card.pack(fill="both", expand=True)

        items = [
            (
                "Sample FSAE Demo Telemetry (Cornering + Drive/Brake)",
                "Loads synthetic Hoosier 18x6-10 TTC measurements across multiple Fz and Camber levels.",
                "Load FSAE Demo Data",
                "Cyan.TButton",
                self._load_demo,
            ),
            (
                "Preset: Hoosier 18x6-10 R25B Base Model",
                "Resets the active MF 5.1 model to clean Formula Student 18x6 R25B baseline parameters.",
                "Apply 18x6 R25B Preset",
                "TButton",
                self._preset_18x6,
            ),
            (
                "Preset: Scaled Hoosier 16x7.5-10 R20",
                "Resets to R25B baseline and applies contact-patch dimensional scaling for 16x7.5-10 R20.",
                "Apply Scaled 16x7.5 R20",
                "Accent.TButton",
                self._preset_16x75,
            ),
        ]
        for title, desc, btn_lbl, style_name, cmd in items:
            box = tk.Frame(card.body, bg=c["panel_alt"], highlightbackground=c["border"], highlightthickness=1, padx=10, pady=8)
            box.pack(fill="x", pady=(0, 8))
            tk.Label(box, text=title, bg=c["panel_alt"], fg=c["text"], font=FONT_BOLD, anchor="w").pack(fill="x")
            tk.Label(box, text=desc, bg=c["panel_alt"], fg=c["text_muted"], font=FONT_SMALL, anchor="w", wraplength=430, justify="left").pack(fill="x", pady=(2, 6))
            ttk.Button(box, text=btn_lbl, style=style_name, command=cmd).pack(anchor="e")

    def _load_demo(self):
        self.app.load_fsae_demo()
        self.app.switch_tab("data")
        self._set_feedback("Loaded synthetic FSAE Demo telemetry.", "ok")

    def _preset_18x6(self):
        self.app.mf51_tab.reset_to_fsae_defaults()
        self.app.switch_tab("model")
        self._set_feedback("Applied Hoosier R25B base preset.", "ok")

    def _preset_16x75(self):
        m = MF51Model()
        scale_18x6_to_16x75_r20(m)
        self.app.set_model(m)
        self.app.on_model_updated()
        self.app.switch_tab("model")
        self._set_feedback("Applied scaled Hoosier 16x7.5-10 R20 model.", "ok")

    def _render_sync_panel(self):
        c = self.c
        card = ModernCard(
            self.right_area,
            "Extract / Sync Active MF 5.1 Model",
            accent_color=c["purple"],
            colors=c,
            pad_x=12,
            pad_y=10,
        )
        card.pack(fill="both", expand=True)

        tk.Label(
            card.body,
            text="Use telemetry or Step 2 optimization fits already inside this session to populate the MF 5.1 model:",
            bg=c["panel"],
            fg=c["text_secondary"],
            font=FONT_SMALL,
            wraplength=440,
            justify="left",
            anchor="w",
        ).pack(fill="x", pady=(0, 12))

        ttk.Button(
            card.body,
            text="Run 1-Click Autopilot MF 5.1 Extraction (All Loaded Files)",
            style="Cyan.TButton",
            command=self._run_autopilot_from_hub,
        ).pack(fill="x", pady=(0, 8))

        ttk.Button(
            card.body,
            text="Apply Step 2 Single/Matrix Fits -> Active MF 5.1 Model",
            style="Accent.TButton",
            command=self._apply_fits_from_hub,
        ).pack(fill="x")

    def _run_autopilot_from_hub(self):
        self.app.switch_tab("fit")
        self.app.run_autopilot_step2()
        self._set_feedback("Launched Autopilot MF 5.1 background extraction.", "cyan")
        self.destroy()

    def _apply_fits_from_hub(self):
        self.app.send_fit_to_mf51(silent=False)
        self._set_feedback("Applied Step 2 fit results to active MF 5.1 model.", "ok")


# ==============================================================================
# UNIFIED EXPORT STUDIO
# ==============================================================================
class UnifiedExportDialog(tk.Toplevel):
    """Single centralized Export Studio where engineers pick target, style options, and preview."""

    EXPORT_TYPES = [
        ("tir", ".TIR Tire Property File (PAC2002)", "Standard model file for ADAMS, CarMaker, OptimumLap"),
        ("excel", "Excel Fy(SA) Report (.xlsx)", "Multi-sheet Fz x P workbook, KPIs & native Excel scatter charts"),
        ("matrix", "Condition Matrix Fz x IA x P (.csv)", "Binned experimental telemetry cells with nearest-center dedup"),
        ("ttc_csv", "Processed TTC Telemetry (.csv)", "Filtered Cornering & Drive/Brake samples from Step 1"),
        ("stiffness", "Cornering Stiffness CFa (.csv)", "Linear regression CFa table + Tire B vs A percentage delta"),
        ("compare", "Tire Comparator KPI (.csv)", "Benchmark KPI comparison table across enabled Slots 1–4"),
        ("json", "Parameters & Fits JSON (Schema v2)", "Active MF 5.1 model, Step 2 fits, or 2D Curve Sketcher"),
        ("plot", "Active Workspace Plot (PNG / PDF / SVG)", "Save the currently active tab figure (Dark or Light Print theme)"),
    ]

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.c = app.c
        self._kind = tk.StringVar(value="tir")
        self._cat_widgets: Dict[str, dict] = {}

        # Format / Appearance variables ("choose how it looks")
        self.var_tir_name = tk.StringVar(value="PUTM_Racing_MF51")
        self.var_tir_surface = tk.StringVar(value="keep")
        self.var_csv_sep = tk.StringVar(value=", (Comma)")
        self.var_ttc_scope = tk.StringVar(value="both")
        self.var_stiff_gain = tk.BooleanVar(value=True)
        self.var_json_src = tk.StringVar(value="mf51")
        self.var_plot_fmt = tk.StringVar(value="PNG")
        self.var_plot_dpi = tk.StringVar(value="220")
        self.var_plot_theme = tk.StringVar(value="dark")

        self.title("PUTM Tire Lab — Unified Export Studio")
        self.geometry("960x640")
        self.minsize(820, 540)
        self.configure(bg=self.c["bg"])
        apply_ttk_theme(self, self.c)

        self._build_ui()
        self._select_kind("tir")

    def _check_readiness(self, key: str) -> Tuple[bool, str]:
        app = self.app
        if key == "tir":
            return True, "Ready"
        if key in ("excel", "matrix"):
            n_r = len(app.runs)
            return (n_r > 0), (f"{n_r} runs" if n_r > 0 else "No .mat")
        if key == "ttc_csv":
            nc = len(app.df_cornering) if app.df_cornering is not None else 0
            nd = len(app.df_drive) if app.df_drive is not None else 0
            tot = nc + nd
            return (tot > 0), (f"{tot:,} pts" if tot > 0 else "No data")
        if key == "stiffness":
            df = app.stiffness_tab.get_export_dataframe()
            return (not df.empty), (f"{len(df)} bins" if not df.empty else "Empty")
        if key == "compare":
            df = app.tir_tab.get_kpi_dataframe()
            return (not df.empty), (f"{len(df)} slots" if not df.empty else "No slots")
        if key == "json":
            return True, "Ready"
        if key == "plot":
            return (self._get_active_figure()[0] is not None), self._get_active_figure()[1]
        return False, "--"

    def _get_active_figure(self) -> Tuple[Optional[Any], str]:
        app = self.app
        k = app._active_key
        mapping = {
            "data": (getattr(app, "fig_hist", None), "01 Histogram"),
            "fit": (getattr(app, "fig_fit", None), "02 Fit Plot"),
            "model": (getattr(app.mf51_tab, "fig", None), "03 MF 5.1"),
            "sim3d": (getattr(app.hub_tab, "fig", None), "04 3D / G-G"),
            "compare": (getattr(app.tir_tab, "fig", None), "05 Comparator"),
            "stiffness": (getattr(app.stiffness_tab, "fig", None), "06 Stiffness"),
            "sketch": (getattr(app.sketch_tab, "fig", None), "07 2D Sketch"),
        }
        return mapping.get(k, (None, "No Plot"))

    def _build_ui(self):
        c = self.c
        hdr = tk.Frame(self, bg=c["bg_header"], height=48, padx=14)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        self._logo = load_brand_logo(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "img", "logo2.png"),
            max_size=(26, 26),
        )
        if self._logo is not None:
            tk.Label(hdr, image=self._logo, bg=c["bg_header"]).pack(side="left", padx=(0, 8))

        tk.Label(
            hdr,
            text="UNIFIED EXPORT STUDIO — SELECT OUTPUT FORMAT, OPTIONS & PREVIEW",
            bg=c["bg_header"],
            fg=c["text"],
            font=("Segoe UI Semibold", 11, "bold"),
        ).pack(side="left")

        tk.Frame(self, bg=c["border"], height=1).pack(fill="x")

        body = tk.Frame(self, bg=c["bg"], padx=12, pady=12)
        body.pack(fill="both", expand=True)

        # Left selector list
        left = tk.Frame(body, bg=c["panel"], width=305, highlightbackground=c["border"], highlightthickness=1)
        left.pack(side="left", fill="y", padx=(0, 12))
        left.pack_propagate(False)

        tk.Label(
            left,
            text="EXPORT TARGET",
            bg=c["panel_alt"],
            fg=c["accent"],
            font=FONT_SMALL_BOLD,
            anchor="w",
            padx=12,
            pady=8,
        ).pack(fill="x")
        tk.Frame(left, bg=c["border"], height=1).pack(fill="x")

        for key, title, sub in self.EXPORT_TYPES:
            ready, badge_txt = self._check_readiness(key)
            row = tk.Frame(left, bg=c["panel"], cursor="hand2")
            row.pack(fill="x", pady=1)
            bar = tk.Frame(row, bg=c["panel"], width=3)
            bar.pack(side="left", fill="y")
            inner = tk.Frame(row, bg=c["panel"], padx=9, pady=6)
            inner.pack(side="left", fill="both", expand=True)

            top = tk.Frame(inner, bg=c["panel"])
            top.pack(fill="x")
            lbl_t = tk.Label(top, text=title, bg=c["panel"], fg=c["text"], font=FONT_SMALL_BOLD, anchor="w")
            lbl_t.pack(side="left", fill="x", expand=True)
            lbl_b = tk.Label(
                top,
                text=badge_txt,
                bg=c["ok_soft"] if ready else c["warn_soft"],
                fg=c["ok"] if ready else c["warn"],
                font=("Segoe UI", 7, "bold"),
                padx=5,
                pady=1,
            )
            lbl_b.pack(side="right")

            lbl_s = tk.Label(inner, text=sub, bg=c["panel"], fg=c["text_muted"], font=("Segoe UI", 7), anchor="w")
            lbl_s.pack(fill="x", pady=(1, 0))

            self._cat_widgets[key] = {"row": row, "bar": bar, "inner": inner, "top": top, "title": lbl_t, "sub": lbl_s}
            for w in (row, bar, inner, top, lbl_t, lbl_b, lbl_s):
                w.bind("<Button-1>", lambda _e, k=key: self._select_kind(k))

        # Right pane: Options + Live Preview + Export Button
        right = tk.Frame(body, bg=c["bg"])
        right.pack(side="right", fill="both", expand=True)

        self.cfg_card = ModernCard(right, "Format & Appearance Settings", accent_color=c["accent"], colors=c, pad_x=12, pad_y=10)
        self.cfg_card.pack(fill="x", pady=(0, 10))

        self.prev_card = ModernCard(right, "Live Output Preview", accent_color=c["cyan"], colors=c, pad_x=12, pad_y=10)
        self.prev_card.pack(fill="both", expand=True)

        self.txt_preview = tk.Text(
            self.prev_card.body,
            bg=c["input"],
            fg=c["text_secondary"],
            font=FONT_MONO,
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=c["border"],
            wrap="none",
        )
        ys = ttk.Scrollbar(self.prev_card.body, orient="vertical", command=self.txt_preview.yview)
        xs = ttk.Scrollbar(self.prev_card.body, orient="horizontal", command=self.txt_preview.xview)
        self.txt_preview.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        ys.pack(side="right", fill="y")
        self.txt_preview.pack(fill="both", expand=True)
        xs.pack(fill="x")

        # Footer action bar
        tk.Frame(self, bg=c["border"], height=1).pack(fill="x")
        foot = tk.Frame(self, bg=c["bg_sidebar"], padx=14, pady=8)
        foot.pack(fill="x")

        self.lbl_status = tk.Label(
            foot,
            text="Configure format options above and inspect the live preview before exporting.",
            bg=c["bg_sidebar"],
            fg=c["text_secondary"],
            font=FONT_SMALL,
            anchor="w",
        )
        self.lbl_status.pack(side="left", fill="x", expand=True)

        ttk.Button(foot, text="Close", command=self.destroy).pack(side="right", padx=(6, 0))
        self.btn_export = ttk.Button(foot, text="Export...", style="Accent.TButton", command=self._execute_export)
        self.btn_export.pack(side="right")

    def _select_kind(self, key: str):
        c = self.c
        self._kind.set(key)
        for k, widgets in self._cat_widgets.items():
            active = (k == key)
            bg = c["panel_elevated"] if active else c["panel"]
            bar_bg = c["accent"] if active else c["panel"]
            fg_t = c["accent_hover"] if active else c["text"]
            for name in ("row", "inner", "top", "title", "sub"):
                widgets[name].configure(bg=bg)
            widgets["bar"].configure(bg=bar_bg)
            widgets["title"].configure(fg=fg_t)

        for child in self.cfg_card.body.winfo_children():
            child.destroy()

        builders = {
            "tir": self._ui_tir,
            "excel": self._ui_excel,
            "matrix": self._ui_matrix,
            "ttc_csv": self._ui_ttc_csv,
            "stiffness": self._ui_stiffness,
            "compare": self._ui_compare,
            "json": self._ui_json,
            "plot": self._ui_plot,
        }
        builders[key]()
        self._refresh_preview()

    def _set_preview(self, text: str):
        self.txt_preview.config(state="normal")
        self.txt_preview.delete("1.0", "end")
        self.txt_preview.insert("1.0", text)
        self.txt_preview.config(state="disabled")

    def _sep_char(self) -> str:
        return ";" if self.var_csv_sep.get().startswith(";") else ","

    def _delimiter_row(self, parent):
        row = tk.Frame(parent, bg=self.c["panel"])
        row.pack(fill="x", pady=3)
        tk.Label(row, text="CSV Delimiter:", bg=self.c["panel"], fg=self.c["text_secondary"], font=FONT_SMALL_BOLD).pack(side="left")
        cb = ttk.Combobox(row, textvariable=self.var_csv_sep, values=[", (Comma — Standard)", "; (Semicolon — EU Excel)"], state="readonly", width=24)
        cb.pack(side="left", padx=(8, 0))
        cb.bind("<<ComboboxSelected>>", lambda _e: self._refresh_preview())

    # ------------------------------------------------------------------
    # PER-FORMAT CONFIG UI + LIVE PREVIEW
    # ------------------------------------------------------------------
    def _ui_tir(self):
        c = self.c
        r1 = tk.Frame(self.cfg_card.body, bg=c["panel"])
        r1.pack(fill="x", pady=(0, 6))
        tk.Label(r1, text="Tire Name in Header:", bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL_BOLD).pack(side="left")
        ent = ttk.Entry(r1, textvariable=self.var_tir_name, width=28)
        ent.pack(side="left", padx=(8, 0))
        ent.bind("<KeyRelease>", lambda _e: self._refresh_preview())

        r2 = tk.Frame(self.cfg_card.body, bg=c["panel"])
        r2.pack(fill="x")
        tk.Label(r2, text="Surface Grip Scale:", bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL_BOLD).pack(side="left", padx=(0, 8))
        for val, lbl in (("keep", "Keep Active Model"), ("asphalt", "Force Asphalt (0.65)"), ("ttc", "Force Flat-Trac (1.00)")):
            ttk.Radiobutton(r2, text=lbl, value=val, variable=self.var_tir_surface, command=self._refresh_preview).pack(side="left", padx=(0, 10))
        self.btn_export.configure(text="Export .TIR File...")

    def _ui_excel(self):
        c = self.c
        tk.Label(
            self.cfg_card.body,
            text="Configure Fz x IA x P binning, SA grid step, Pacejka fits, binned means, and native Excel scatter charts inside the Excel Builder.",
            bg=c["panel"],
            fg=c["text_secondary"],
            font=FONT_SMALL,
            wraplength=560,
            justify="left",
            anchor="w",
        ).pack(fill="x")
        self.btn_export.configure(text="Open Fy(SA) Excel Builder...")

    def _ui_matrix(self):
        c = self.c
        tk.Label(
            self.cfg_card.body,
            text="Slice unfiltered Step 1 files into clean Fz x Camber x Pressure cells with preview counts and delimiter choice.",
            bg=c["panel"],
            fg=c["text_secondary"],
            font=FONT_SMALL,
            wraplength=560,
            justify="left",
            anchor="w",
        ).pack(fill="x")
        self.btn_export.configure(text="Open Condition Matrix Builder...")

    def _ui_ttc_csv(self):
        c = self.c
        r1 = tk.Frame(self.cfg_card.body, bg=c["panel"])
        r1.pack(fill="x", pady=(0, 4))
        tk.Label(r1, text="Dataset Scope:", bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL_BOLD).pack(side="left", padx=(0, 8))
        for val, lbl in (("both", "Cornering + Drive/Brake"), ("cornering", "Cornering Only"), ("drive", "Drive/Brake Only")):
            ttk.Radiobutton(r1, text=lbl, value=val, variable=self.var_ttc_scope, command=self._refresh_preview).pack(side="left", padx=(0, 10))
        self._delimiter_row(self.cfg_card.body)
        self.btn_export.configure(text="Export Telemetry CSV...")

    def _ui_stiffness(self):
        self._delimiter_row(self.cfg_card.body)
        ttk.Checkbutton(
            self.cfg_card.body,
            text="Merge Gain B vs A delta columns (delta_N_deg, gain_pct)",
            variable=self.var_stiff_gain,
            command=self._refresh_preview,
        ).pack(anchor="w", pady=(4, 0))
        self.btn_export.configure(text="Export Stiffness CSV...")

    def _ui_compare(self):
        self._delimiter_row(self.cfg_card.body)
        self.btn_export.configure(text="Export Benchmark KPI CSV...")

    def _ui_json(self):
        c = self.c
        tk.Label(self.cfg_card.body, text="Select model source to serialize:", bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL_BOLD, anchor="w").pack(fill="x", pady=(0, 4))
        for val, lbl in (
            ("mf51", "Active MF 5.1 Model (all PAC2002 coefficients)"),
            ("fits", "Step 2 Fitted Models (Schema v2)"),
            ("sketch", "2D Curve Sketcher Models (Schema v2)"),
        ):
            ttk.Radiobutton(self.cfg_card.body, text=lbl, value=val, variable=self.var_json_src, command=self._refresh_preview).pack(anchor="w", pady=2)
        self.btn_export.configure(text="Export JSON...")

    def _ui_plot(self):
        c = self.c
        _fig, name = self._get_active_figure()
        tk.Label(self.cfg_card.body, text=f"Target Figure: {name}", bg=c["panel"], fg=c["cyan"], font=FONT_SMALL_BOLD, anchor="w").pack(fill="x", pady=(0, 6))

        r1 = tk.Frame(self.cfg_card.body, bg=c["panel"])
        r1.pack(fill="x", pady=3)
        tk.Label(r1, text="File Format:", bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL_BOLD).pack(side="left")
        cb_f = ttk.Combobox(r1, textvariable=self.var_plot_fmt, values=["PNG", "PDF", "SVG"], state="readonly", width=10)
        cb_f.pack(side="left", padx=(8, 16))
        cb_f.bind("<<ComboboxSelected>>", lambda _e: self._refresh_preview())

        tk.Label(r1, text="DPI:", bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL_BOLD).pack(side="left")
        cb_d = ttk.Combobox(r1, textvariable=self.var_plot_dpi, values=["150", "220", "300"], state="readonly", width=8)
        cb_d.pack(side="left", padx=(8, 0))
        cb_d.bind("<<ComboboxSelected>>", lambda _e: self._refresh_preview())

        r2 = tk.Frame(self.cfg_card.body, bg=c["panel"])
        r2.pack(fill="x", pady=(4, 0))
        tk.Label(r2, text="Visual Style:", bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL_BOLD).pack(side="left", padx=(0, 8))
        ttk.Radiobutton(r2, text="Dark Telemetry Theme (Screen)", value="dark", variable=self.var_plot_theme, command=self._refresh_preview).pack(side="left", padx=(0, 12))
        ttk.Radiobutton(r2, text="Light Report Theme (White Paper)", value="light", variable=self.var_plot_theme, command=self._refresh_preview).pack(side="left")
        self.btn_export.configure(text="Save Chart Image...")

    def _build_tir_preview_model(self) -> MF51Model:
        m = self.app.model.copy()
        surf = self.var_tir_surface.get()
        if surf == "asphalt":
            m.set("LMUX", 0.65)
            m.set("LMUY", 0.65)
        elif surf == "ttc":
            m.set("LMUX", 1.00)
            m.set("LMUY", 1.00)
        return m

    def _build_ttc_dataframe(self) -> pd.DataFrame:
        scope = self.var_ttc_scope.get()
        chunks = []
        if scope in ("both", "cornering") and self.app.df_cornering is not None and not self.app.df_cornering.empty:
            c_df = self.app.df_cornering.copy()
            c_df.insert(0, "TestType", "Cornering")
            chunks.append(c_df)
        if scope in ("both", "drive") and self.app.df_drive is not None and not self.app.df_drive.empty:
            d_df = self.app.df_drive.copy()
            d_df.insert(0, "TestType", "Drive/Brake")
            chunks.append(d_df)
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    def _refresh_preview(self):
        kind = self._kind.get()
        sep = self._sep_char()

        if kind == "tir":
            m = self._build_tir_preview_model()
            content = m.to_tir_content(self.var_tir_name.get().strip() or "PUTM_Tire")
            self._set_preview("\n".join(content.splitlines()[:32]) + "\n...")
        elif kind == "excel":
            n_runs = len(self.app.runs)
            self._set_preview(
                f"Excel Workbook (.xlsx) Summary\n"
                f"Queued input runs: {n_runs}\n\n"
                f"Workbook layout generated:\n"
                f"  1. Per-load sheets: Fy(SA) @ Fz=...N  (with native ScatterCharts)\n"
                f"  2. Lookup matrix  : Matrix_Fy_SA_Fz_P\n"
                f"  3. KPI sheet      : Summary_KPI (Peak Fy, CFa [N/deg], mu peak, B/C/D/E)\n"
                f"  4. Experimental   : Binned_Measurements (mean & std dev)\n"
                f"  5. Metadata       : Metadata_Info (conventions & equations)\n"
            )
        elif kind == "matrix":
            n_runs = len(self.app.runs)
            self._set_preview(
                f"Condition Matrix CSV Builder\n"
                f"Queued input runs: {n_runs}\n\n"
                f"Bins unfiltered measurements across (Fz_bin_N, IA_bin_deg, P_bin_kPa).\n"
                f"Click 'Open Condition Matrix Builder...' to edit bin edges, verify sample\n"
                f"counts per cell, choose delimiter (comma/semicolon), and save CSV."
            )
        elif kind == "ttc_csv":
            df = self._build_ttc_dataframe()
            self._set_preview(_format_preview_csv(df, sep))
        elif kind == "stiffness":
            df = self.app.stiffness_tab.get_export_dataframe(include_gain=bool(self.var_stiff_gain.get()))
            self._set_preview(_format_preview_csv(df, sep))
        elif kind == "compare":
            df = self.app.tir_tab.get_kpi_dataframe()
            self._set_preview(_format_preview_csv(df, sep))
        elif kind == "json":
            src = self.var_json_src.get()
            payload = self._build_json_payload(src)
            if payload is None:
                self._set_preview(f"(No data available yet for JSON source '{src}')")
            else:
                self._set_preview(json.dumps(payload, indent=2, ensure_ascii=False))
        elif kind == "plot":
            fig, name = self._get_active_figure()
            theme = "White background / high-contrast print ink" if self.var_plot_theme.get() == "light" else "Carbon Dark Telemetry theme"
            self._set_preview(
                f"Active Figure Export Preview\n"
                f"Source tab : {name}\n"
                f"Format     : {self.var_plot_fmt.get()}\n"
                f"Resolution : {self.var_plot_dpi.get()} DPI\n"
                f"Appearance : {theme}\n"
                f"Axes count : {len(fig.axes) if fig is not None else 0}"
            )

    def _build_json_payload(self, src: str) -> Optional[dict]:
        if src == "mf51":
            return {"model_type": "Magic Formula 5.1 / PAC2002", "parameters": dict(self.app.model.p)}
        if src == "fits":
            if not self.app.fit_results:
                return None
            return self.app.build_coeffs_json_payload()
        return self.app.sketch_tab.get_export_payload()

    def _execute_export(self):
        kind = self._kind.get()
        sep = self._sep_char()

        if kind == "excel":
            if not self.app.runs:
                messagebox.showwarning("No Input Files", "Queue .mat / .csv runs in Step 1 first.", parent=self)
                return
            FyExcelDialog(self, self.app.runs, raw_cache=self.app.raw_cache, colors=self.c)
            return

        if kind == "matrix":
            if not self.app.runs:
                messagebox.showwarning("No Input Files", "Queue .mat / .csv runs in Step 1 first.", parent=self)
                return
            ConditionExportDialog(self, self.app.runs, raw_cache=self.app.raw_cache, colors=self.c)
            return

        if kind == "tir":
            m = self._build_tir_preview_model()
            tire_name = self.var_tir_name.get().strip() or "PUTM_MF51_Tire"
            path = filedialog.asksaveasfilename(
                parent=self,
                title="Export .TIR Tire Property File",
                defaultextension=".tir",
                filetypes=[("ADAMS / CarSim TIR", "*.tir"), ("All Files", "*.*")],
                initialfile=f"{tire_name}.tir",
            )
            if not path:
                return
            with open(path, "w", encoding="utf-8") as f:
                f.write(m.to_tir_content(tire_name))
            self._done_export(f"Exported .TIR file to {path}")
            return

        if kind == "ttc_csv":
            df = self._build_ttc_dataframe()
            if df.empty:
                messagebox.showwarning("No Telemetry Data", "Process runs in Step 1 or load FSAE Demo first.", parent=self)
                return
            self._save_df_csv(df, sep, "PUTM_TTC_Filtered.csv", "Export Filtered TTC Data CSV")
            return

        if kind == "stiffness":
            df = self.app.stiffness_tab.get_export_dataframe(include_gain=bool(self.var_stiff_gain.get()))
            if df.empty:
                messagebox.showwarning("No Stiffness Results", "Calculate CFa stiffness in Tool 6 first.", parent=self)
                return
            self._save_df_csv(df, sep, "PUTM_CFa_Stiffness.csv", "Export Cornering Stiffness CSV")
            return

        if kind == "compare":
            df = self.app.tir_tab.get_kpi_dataframe()
            if df.empty:
                messagebox.showwarning("No Active Models", "Enable at least one model in Step 5 Comparator.", parent=self)
                return
            self._save_df_csv(df, sep, "PUTM_Benchmark_KPI.csv", "Export Benchmark KPI CSV")
            return

        if kind == "json":
            payload = self._build_json_payload(self.var_json_src.get())
            if payload is None:
                messagebox.showwarning("No Data", "Selected JSON source has no parameters yet.", parent=self)
                return
            path = filedialog.asksaveasfilename(
                parent=self,
                title="Export Parameters JSON",
                defaultextension=".json",
                filetypes=[("JSON", "*.json")],
                initialfile=f"PUTM_{self.var_json_src.get()}_v2.json",
            )
            if not path:
                return
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            self._done_export(f"Exported JSON to {path}")
            return

        if kind == "plot":
            fig, tab_lbl = self._get_active_figure()
            if fig is None:
                messagebox.showwarning("No Plot", "Switch to a tab with a chart first.", parent=self)
                return
            ext = self.var_plot_fmt.get().lower()
            dpi = int(self.var_plot_dpi.get())
            path = filedialog.asksaveasfilename(
                parent=self,
                title=f"Save Plot ({tab_lbl})",
                defaultextension=f".{ext}",
                filetypes=[(f"{ext.upper()} Image", f"*.{ext}"), ("All Files", "*.*")],
                initialfile=f"PUTM_{tab_lbl.replace(' ', '_')}.{ext}",
            )
            if not path:
                return
            self._save_figure_styled(fig, path, dpi, light_mode=(self.var_plot_theme.get() == "light"))
            self._done_export(f"Saved plot image to {path}")

    def _save_df_csv(self, df: pd.DataFrame, sep: str, initialfile: str, title: str):
        path = filedialog.asksaveasfilename(
            parent=self,
            title=title,
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All Files", "*.*")],
            initialfile=initialfile,
        )
        if not path:
            return
        df.to_csv(path, sep=sep, index=False, encoding="utf-8")
        self._done_export(f"Exported {len(df):,} rows to {path}")

    def _save_figure_styled(self, fig, path: str, dpi: int, *, light_mode: bool):
        if not light_mode:
            fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
            return
        # Temporarily re-theme figure to crisp white background for thesis/design reports, then restore
        saved_fig_fc = fig.get_facecolor()
        saved_axes = []
        try:
            fig.patch.set_facecolor("#FFFFFF")
            for ax in fig.axes:
                entry = {
                    "ax": ax,
                    "fc": ax.get_facecolor(),
                    "title_c": ax.title.get_color(),
                    "xlabel_c": ax.xaxis.label.get_color(),
                    "ylabel_c": ax.yaxis.label.get_color(),
                    "xticks_c": [t.get_color() for t in ax.get_xticklabels()],
                    "yticks_c": [t.get_color() for t in ax.get_yticklabels()],
                    "spines": {k: s.get_edgecolor() for k, s in ax.spines.items()},
                }
                saved_axes.append(entry)
                ax.set_facecolor("#FFFFFF")
                ax.title.set_color("#0F172A")
                ax.xaxis.label.set_color("#1E293B")
                ax.yaxis.label.set_color("#1E293B")
                ax.tick_params(colors="#334155")
                for s in ax.spines.values():
                    s.set_edgecolor("#CBD5E1")
                leg = ax.get_legend()
                if leg is not None:
                    leg.get_frame().set_facecolor("#F8FAFC")
                    leg.get_frame().set_edgecolor("#CBD5E1")
                    for txt in leg.get_texts():
                        txt.set_color("#0F172A")
            fig.savefig(path, dpi=dpi, facecolor="#FFFFFF", bbox_inches="tight")
        finally:
            fig.patch.set_facecolor(saved_fig_fc)
            for e in saved_axes:
                ax = e["ax"]
                ax.set_facecolor(e["fc"])
                ax.title.set_color(e["title_c"])
                ax.xaxis.label.set_color(e["xlabel_c"])
                ax.yaxis.label.set_color(e["ylabel_c"])
                for lbl, col in zip(ax.get_xticklabels(), e["xticks_c"]):
                    lbl.set_color(col)
                for lbl, col in zip(ax.get_yticklabels(), e["yticks_c"]):
                    lbl.set_color(col)
                for k, s in ax.spines.items():
                    s.set_edgecolor(e["spines"][k])
                leg = ax.get_legend()
                if leg is not None:
                    leg.get_frame().set_facecolor(self.c["panel_alt"])
                    leg.get_frame().set_edgecolor(self.c["border_bright"])
                    for txt in leg.get_texts():
                        txt.set_color(self.c["text"])
            if hasattr(fig, "canvas") and fig.canvas is not None:
                fig.canvas.draw_idle()

    def _done_export(self, msg: str):
        self.app.set_status(msg)
        self.lbl_status.config(text=msg, fg=self.c["ok"])
        messagebox.showinfo("Export Complete", msg, parent=self)
