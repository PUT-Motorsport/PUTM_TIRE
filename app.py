"""
PUTM Tire Lab — Formula Student Tire Modeling & Telemetry Suite.

Carbon Motorsport Telemetry Edition:
- Streamlined Top-Bar Unified Import & Export Studio (single entry points, zero clutter)
- 7-step left SidebarNav workflow:
    01. Telemetry Data         -> .mat/.csv/.txt/.dat queue, condition filters, histograms
    02. Model Fitting          -> 1-Click Autopilot MF 5.1 + Single Curve & Matrix Auto-Fit
    03. MF 5.1 Model (.TIR)    -> PAC2002 parameters, surface scaling, 18x6 -> 16x7.5 R20 scaler
    04. 3D & G-G Simulator     -> 3D force surfaces, friction cone, live SA x SL grip monitor
    05. Tire Comparator (.TIR) -> Up to 4 compounds, Benchmark KPI table & Radar
    06. Cornering Stiffness    -> Linear CFa(Fz, IA, P) regression with dwell/SL filtering & A vs B
    07. 2D Curve Sketcher      -> Fast Physical vs B/C/D/E Magic Formula calculator
"""

from __future__ import annotations

import json
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from auto_mf51_extractor import AutoMF51Extractor
from condition_export_dialog import ConditionExportDialog, open_excel_export
from distribution import make_histogram
from fitting import (
    fit_combined_slip,
    fit_load_sensitive,
    fit_pure_magic,
)
from hub_tab import HubTab, generate_fsae_demo_telemetry
from io_dialogs import UnifiedExportDialog, UnifiedImportDialog
from matrix_autofit_dialog import MatrixAutoFitDialog
from mf51_model import MF51Model
from mf51_tab import MF51Tab
from simple_pacejka_tab import SimplePacejkaTab
from stiffness_tab import StiffnessTab
from tire_model import RunConfig, TireDataManager
from tir_analyzer_dialog import TirAnalyzerTab
from tir_parser import load_tir_to_model
from ui_scroll import ScrollableFrame
from ui_theme import (
    CHART_PALETTE,
    FONT_DISPLAY,
    FONT_MONO,
    FONT_SMALL,
    FONT_SMALL_BOLD,
    PALETTE,
    KpiTile,
    ModernCard,
    SegmentedToggle,
    SidebarNav,
    StatusBanner,
    StepHeader,
    apply_ttk_theme,
    load_brand_logo,
    style_axes_2d,
    style_legend,
    style_treeview_zebra,
)


HISTOGRAM_CHANNELS = [
    ("SA", "Slip Angle SA [deg]"),
    ("FY", "Lateral Force FY [N]"),
    ("FZ", "Vertical Load FZ [N]"),
    ("IA", "Camber IA [deg]"),
    ("SL", "Longitudinal Slip SL [-]"),
    ("FX", "Longitudinal Force FX [N]"),
    ("MZ", "Aligning Torque MZ [Nm]"),
    ("P", "Pressure P [kPa / psi]"),
    ("V", "Speed V [km/h]"),
]

FIT_MODELS = [
    ("Fy0", "Fy0(SA) — Pure Lateral Force"),
    ("Fy_load", "Fy_load(SA, Fz) — Load-Sensitive 3D Surface"),
    ("Fx0", "Fx0(SL) — Pure Drive / Braking"),
    ("Mz0", "Mz0(SA) — Aligning Torque"),
    ("CombinedSlip", "CombinedSlip Gyk(SL) — Grip Reduction"),
    ("Mu_Fz", "Mu_Fz(Fz) — Peak Friction vs Load"),
    ("Envelope", "Envelope G-G — Peak Friction Circle"),
]

PARAM_DESCRIPTIONS: Dict[str, tuple] = {
    "B": ("1/rad or -", "Initial stiffness factor"),
    "C": ("-", "Curve shape factor"),
    "D": ("N or Nm", "Peak force / torque amplitude"),
    "E": ("-", "Curvature factor past peak"),
    "Sh": ("rad or -", "Horizontal shift (plysteer / offset)"),
    "Sv": ("N or Nm", "Vertical shift"),
    "Bx": ("-", "Longitudinal stiffness factor (SL)"),
    "Cx": ("-", "Longitudinal shape factor"),
    "Dx": ("N", "Peak longitudinal force"),
    "Ex": ("-", "Longitudinal curvature factor"),
    "Shx": ("-", "Horizontal SL shift"),
    "Svx": ("N", "Vertical Fx shift"),
    "Bz": ("1/rad", "Aligning torque stiffness factor"),
    "Cz": ("-", "Aligning torque shape factor"),
    "Dz": ("Nm", "Peak aligning torque Mz"),
    "Ez": ("-", "Aligning torque curvature factor"),
    "pCy1": ("-", "Load-sensitive Fy0 shape factor"),
    "pDy1": ("-", "Nominal friction mu_y at Fz0"),
    "pDy2": ("-", "Load sensitivity slope d(mu_y)/dfz"),
    "pEy1": ("-", "Lateral curvature at Fz0"),
    "pKy1": ("1/rad", "Normalized cornering stiffness Ky/Fz0"),
    "pKy2": ("-", "Load saturation factor PKY2"),
    "pUy1": ("rad", "Horizontal shift PHY1"),
    "pVy1": ("-", "Vertical shift PVY1/Fz0"),
    "fz0": ("N", "Nominal reference load FNOMIN"),
    "rBy1": ("-", "Combined-slip grip reduction slope Gyk"),
    "rCy1": ("-", "Combined-slip shape factor Gyk"),
    "pDx1": ("-", "Nominal friction mu_x at Fz0"),
    "pDx2": ("-", "Load sensitivity slope d(mu_x)/dfz"),
}

NAV_KEYS = ("data", "fit", "model", "sim3d", "compare", "stiffness", "sketch")


class PUTMTireApp:
    """Main PUTM Tire Lab application with unified Import/Export hubs and telemetry dark UI."""

    def __init__(self, root: Optional[tk.Tk] = None):
        self.root = root or tk.Tk()
        self.c = PALETTE

        self.root.title("PUTM Tire Lab — Formula Student Tire Dynamics Suite")
        self.root.geometry("1346x706+8+8")
        self.root.minsize(1140, 640)
        self.root.configure(bg=self.c["bg"])
        apply_ttk_theme(self.root, self.c)

        # Shared application state
        self.runs: List[RunConfig] = []
        self.raw_cache: dict = {}
        self.df_cornering: Optional[pd.DataFrame] = None
        self.df_drive: Optional[pd.DataFrame] = None
        self.fit_results: Dict[str, dict] = {}
        self.fit_result: Optional[dict] = None
        self.model: MF51Model = MF51Model()
        self._active_model_name: str = "Hoosier 18x6-10 R25B (Preset)"
        self._active_key: str = "data"
        self._syncing_views: bool = False
        self._autopilot_running: bool = False

        self._hist_cache: Optional[dict] = None

        # UI state variables
        self.var_status = tk.StringVar(value="Ready — click 'Import...' to load .MAT/.CSV telemetry, .TIR files, or FSAE Demo.")
        self.var_run_type = tk.StringVar(value="Cornering")
        self.var_target_fz = tk.StringVar(value="900")
        self.var_tol_fz = tk.StringVar(value="200")
        self.var_target_param = tk.StringVar(value="0")
        self.var_tol_param = tk.StringVar(value="0.6")
        self.var_drive_sa = tk.StringVar(value="0")
        self.var_drive_sa_tol = tk.StringVar(value="1.0")

        self.var_hist_source = tk.StringVar(value="Cornering")
        self.var_hist_channel = tk.StringVar(value=HISTOGRAM_CHANNELS[0][1])
        self.var_hist_focus = tk.BooleanVar(value=True)

        self.var_autopilot_p = tk.StringVar(value="")
        self.var_fit_model = tk.StringVar(value=FIT_MODELS[0][1])
        self.var_fit_fz_slice = tk.StringVar(value="")
        self.var_fit_c0 = tk.StringVar(value="1.30")

        self._views: Dict[str, tk.Widget] = {}

        self._build_shell()
        self._build_step1_data()
        self._build_step2_fit()
        self._build_step3_model()
        self._build_step4_sim3d()
        self._build_step5_compare()
        self._build_tool6_stiffness()
        self._build_tool7_sketch()

        self.switch_tab("data")
        self._update_telemetry_badges()

    # ------------------------------------------------------------------
    # MODEL & STATUS COORDINATION
    # ------------------------------------------------------------------
    def set_status(self, msg: str):
        self.var_status.set(msg)

    def get_model(self) -> MF51Model:
        return self.model

    def set_model(self, model: MF51Model):
        self.model = model
        self._active_model_name = self._describe_model(model)

    def _describe_model(self, model: MF51Model) -> str:
        w = model.get("WIDTH", 0.1524)
        r0 = model.get("UNLOADED_RADIUS", 0.2286)
        if abs(w - 0.180) < 0.015 or abs(r0 - 0.205) < 0.012:
            return "Hoosier 16x7.5-10 R20"
        return f"MF 5.1 (R0={r0*1000:.0f}mm)"

    def on_model_updated(self):
        """Sync active MF51Model across Step 3, Step 4, and Step 5."""
        if self._syncing_views:
            return
        self._syncing_views = True
        try:
            if hasattr(self, "mf51_tab"):
                self.mf51_tab.sync_from_model(redraw=True)
            if hasattr(self, "hub_tab"):
                self.hub_tab.sync_from_model(redraw=(self._active_key == "sim3d"))
            if hasattr(self, "tir_tab"):
                self.tir_tab.sync_active_model(redraw=(self._active_key == "compare"))
        finally:
            self._syncing_views = False
        self._update_telemetry_badges()

    def _update_telemetry_badges(self):
        c = self.c
        n_c = len(self.df_cornering) if self.df_cornering is not None else 0
        n_d = len(self.df_drive) if self.df_drive is not None else 0
        total = n_c + n_d

        if total > 0:
            self.lbl_data_pill.config(
                text=f"TTC Data: {total:,} pts",
                bg=c["ok_soft"],
                fg=c["ok"],
            )
            self.sidebar.set_item_status("data", "ok", f"{len(self.runs)} file(s) | {total:,} pts")
            if hasattr(self, "fit_empty_banner"):
                self.fit_empty_banner.pack_forget()
        else:
            self.lbl_data_pill.config(
                text="TTC Data: empty",
                bg=c["panel_alt"],
                fg=c["text_muted"],
            )
            self.sidebar.set_item_status("data", "warn", "Import .mat / Demo")
            if hasattr(self, "fit_empty_banner") and not self.fit_empty_banner.winfo_ismapped():
                self.fit_empty_banner.pack(fill="x", pady=(0, 8), before=self._fit_body)

        lmuy = self.model.get("LMUY", 1.0)
        surf_lbl = "Asphalt" if lmuy < 0.85 else "Flat-Trac"
        self.lbl_model_pill.config(
            text=f"Model: {self._active_model_name} · {surf_lbl} ({lmuy:.2f})",
            bg=c["cyan_soft"],
            fg=c["cyan"],
        )
        self.sidebar.set_item_status("model", "ok", f"{surf_lbl} LMUY={lmuy:.2f}")

        n_fits = len(self.fit_results)
        if n_fits > 0:
            self.sidebar.set_item_status("fit", "ok", f"{n_fits} fit(s) ready")
        elif total > 0:
            self.sidebar.set_item_status("fit", "warn", "Ready to fit")
        else:
            self.sidebar.set_item_status("fit", "idle", "Requires TTC data")

    # ------------------------------------------------------------------
    # SHELL: HEADER, SIDEBAR, FOOTER
    # ------------------------------------------------------------------
    def _build_shell(self):
        c = self.c

        # Top Header Bar with unified Import... and Export... triggers
        hdr = tk.Frame(self.root, bg=c["bg_header"], height=54)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        left_h = tk.Frame(hdr, bg=c["bg_header"], padx=14)
        left_h.pack(side="left", fill="y")

        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img", "logo2.png")
        self._hdr_logo = load_brand_logo(logo_path, max_size=(32, 32))
        if self._hdr_logo is not None:
            tk.Label(left_h, image=self._hdr_logo, bg=c["bg_header"]).pack(side="left", padx=(0, 10))

        title_blk = tk.Frame(left_h, bg=c["bg_header"])
        title_blk.pack(side="left", pady=6)
        tk.Label(
            title_blk,
            text="PUTM TIRE LAB",
            bg=c["bg_header"],
            fg=c["text"],
            font=FONT_DISPLAY,
            anchor="w",
        ).pack(anchor="w")
        tk.Label(
            title_blk,
            text="Formula Student Tire Modeling & Telemetry Suite",
            bg=c["bg_header"],
            fg=c["text_muted"],
            font=FONT_SMALL,
            anchor="w",
        ).pack(anchor="w")

        # Right unified I/O buttons + telemetry status pills
        right_h = tk.Frame(hdr, bg=c["bg_header"], padx=14)
        right_h.pack(side="right", fill="y")

        ttk.Button(
            right_h,
            text="Export...",
            style="Accent.TButton",
            command=self.open_export_dialog,
        ).pack(side="right", pady=11, padx=(8, 0))

        ttk.Button(
            right_h,
            text="Import...",
            style="Cyan.TButton",
            command=self.open_import_dialog,
        ).pack(side="right", pady=11, padx=(12, 0))

        self.lbl_model_pill = tk.Label(
            right_h,
            text="Model: --",
            bg=c["cyan_soft"],
            fg=c["cyan"],
            font=FONT_SMALL_BOLD,
            padx=10,
            pady=4,
        )
        self.lbl_model_pill.pack(side="right", pady=13, padx=(6, 0))

        self.lbl_data_pill = tk.Label(
            right_h,
            text="TTC Data: empty",
            bg=c["panel_alt"],
            fg=c["text_muted"],
            font=FONT_SMALL_BOLD,
            padx=10,
            pady=4,
        )
        self.lbl_data_pill.pack(side="right", pady=13)

        tk.Frame(self.root, bg=c["border"], height=1).pack(fill="x")

        # Main workspace container: Sidebar + Content
        workspace = tk.Frame(self.root, bg=c["bg"])
        workspace.pack(fill="both", expand=True)

        self.sidebar = SidebarNav(workspace, on_select=self.switch_tab, width=228, colors=c)
        self.sidebar.pack(side="left", fill="y")

        self.sidebar.add_section("WORKFLOW")
        self.sidebar.add_item("data", "01", "Telemetry Data", "Run queue, filters & histograms")
        self.sidebar.add_item("fit", "02", "Model Fitting", "Autopilot MF 5.1 & curves")
        self.sidebar.add_item("model", "03", "MF 5.1 Model", "Editor, asphalt, 16x7.5 R20")
        self.sidebar.add_item("sim3d", "04", "3D & G-G Simulator", "Surfaces, cone, SA/SL grip")
        self.sidebar.add_item("compare", "05", "Tire Comparator", "Benchmark 1–4 .TIR models")

        self.sidebar.add_section("ANALYSIS TOOLS")
        self.sidebar.add_item("stiffness", "06", "Cornering Stiffness", "Linear CFa across Fz x IA x P", is_workflow=False)
        self.sidebar.add_item("sketch", "07", "2D Curve Sketcher", "Magic Formula quick calculator", is_workflow=False)

        self.content_area = tk.Frame(workspace, bg=c["bg"], padx=12, pady=10)
        self.content_area.pack(side="right", fill="both", expand=True)

        # Bottom Status Bar
        tk.Frame(self.root, bg=c["border"], height=1).pack(fill="x")
        footer = tk.Frame(self.root, bg=c["bg_sidebar"], height=26, padx=12)
        footer.pack(fill="x")
        footer.pack_propagate(False)

        tk.Label(
            footer,
            textvariable=self.var_status,
            bg=c["bg_sidebar"],
            fg=c["text_secondary"],
            font=FONT_SMALL,
            anchor="w",
        ).pack(side="left", fill="x", expand=True)

        tk.Label(
            footer,
            text="Convention: Fz > 0 [N] | SA [deg] UI / [rad] model | Schema v2",
            bg=c["bg_sidebar"],
            fg=c["text_muted"],
            font=FONT_SMALL,
            anchor="e",
        ).pack(side="right")

    def switch_tab(self, key_or_index):
        """Switch workspace pane by key ('data', 'fit', ...) or numeric index."""
        if str(key_or_index) == "export":
            self.open_export_dialog()
            return
        if isinstance(key_or_index, int):
            key = NAV_KEYS[max(0, min(key_or_index, len(NAV_KEYS) - 1))]
        else:
            key = str(key_or_index)
        if key not in self._views:
            return

        for k, frame in self._views.items():
            if k == key:
                frame.pack(fill="both", expand=True)
            else:
                frame.pack_forget()

        self._active_key = key
        self.sidebar.set_active(key)

        if key == "model" and hasattr(self, "mf51_tab"):
            self.mf51_tab.sync_from_model(redraw=True)
        elif key == "sim3d" and hasattr(self, "hub_tab"):
            self.hub_tab.sync_from_model(redraw=True)
        elif key == "compare" and hasattr(self, "tir_tab"):
            self.tir_tab.sync_active_model(redraw=True)

    # ------------------------------------------------------------------
    # UNIFIED IMPORT / EXPORT DIALOGS & GLOBAL ACTIONS
    # ------------------------------------------------------------------
    def open_import_dialog(self, initial_mode: str = "telemetry"):
        UnifiedImportDialog(self.root, self, initial_mode=initial_mode)

    def open_export_dialog(self):
        UnifiedExportDialog(self.root, self)

    def apply_loaded_tir_model(self, model: MF51Model, tire_name: str, filepath: str = ""):
        self.model = model
        self._active_model_name = tire_name or os.path.basename(filepath) or "MF 5.1"
        self.set_status(f"Loaded active .TIR model: {self._active_model_name}")
        self.on_model_updated()

    def load_fsae_demo(self):
        df_c, df_d = generate_fsae_demo_telemetry()
        self.df_cornering = df_c
        self.df_drive = df_d
        self.set_status(f"Loaded FSAE Demo TTC: {len(df_c)} Cornering + {len(df_d)} Drive/Brake samples.")
        self._refresh_data_kpis()
        self._draw_histogram()
        self._update_telemetry_badges()

    def load_tir_global(self):
        self.open_import_dialog(initial_mode="tir")

    # ------------------------------------------------------------------
    # STEP 1: TELEMETRY DATA
    # ------------------------------------------------------------------
    def _build_step1_data(self):
        c = self.c
        view = ttk.Frame(self.content_area)
        self._views["data"] = view

        StepHeader(
            view,
            "STEP 1 / 5",
            "Telemetry Data Queue (.mat / .csv / .txt)",
            "Configure condition filters, process loaded telemetry runs, and inspect channel distributions.",
            badge_kind="accent",
            next_text="Next: Model Fitting (Step 2) ->",
            next_command=lambda: self.switch_tab("fit"),
            colors=c,
        ).pack(fill="x", pady=(0, 8))

        body = ttk.Frame(view)
        body.pack(fill="both", expand=True)

        left_wrap = tk.Frame(body, bg=c["bg"], width=425)
        left_wrap.pack(side="left", fill="y", padx=(0, 10))
        left_wrap.pack_propagate(False)

        self._data_scroll = ScrollableFrame(left_wrap, bg=c["bg"], padx=2, pady=2)
        self._data_scroll.pack(fill="both", expand=True)
        ctrl = self._data_scroll.inner

        # 1. Condition Filter settings
        card_flt = ModernCard(ctrl, "1. Condition Filter Settings (for Queued Runs)", accent_color=c["accent"], colors=c, pad_x=10, pad_y=8)
        card_flt.pack(fill="x", pady=(0, 8))

        self._seg_run_type = SegmentedToggle(
            card_flt.body,
            self.var_run_type,
            [("Cornering", "Cornering (Fy / Mz)"), ("Drive/Brake", "Drive/Brake (Fx)")],
            command=self._on_run_type_toggle,
            accent_key="accent",
            colors=c,
        )
        self._seg_run_type.pack(fill="x", pady=(0, 8))

        grid_f = tk.Frame(card_flt.body, bg=c["panel"])
        grid_f.pack(fill="x")
        grid_f.columnconfigure(1, weight=1)
        grid_f.columnconfigure(3, weight=1)

        tk.Label(grid_f, text="Fz [N]:", bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL_BOLD).grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(grid_f, textvariable=self.var_target_fz, width=9).grid(row=0, column=1, sticky="ew", padx=(4, 10), pady=2)
        tk.Label(grid_f, text="Tol ±:", bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL).grid(row=0, column=2, sticky="w", pady=2)
        ttk.Entry(grid_f, textvariable=self.var_tol_fz, width=8).grid(row=0, column=3, sticky="ew", padx=(4, 0), pady=2)

        self.lbl_param_target = tk.Label(grid_f, text="IA [deg]:", bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL_BOLD)
        self.lbl_param_target.grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(grid_f, textvariable=self.var_target_param, width=9).grid(row=1, column=1, sticky="ew", padx=(4, 10), pady=2)
        tk.Label(grid_f, text="Tol ±:", bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL).grid(row=1, column=2, sticky="w", pady=2)
        ttk.Entry(grid_f, textvariable=self.var_tol_param, width=8).grid(row=1, column=3, sticky="ew", padx=(4, 0), pady=2)

        self.drive_sa_row = tk.Frame(card_flt.body, bg=c["panel"])
        tk.Label(self.drive_sa_row, text="Opt. |SA| target:", bg=c["panel"], fg=c["text_muted"], font=FONT_SMALL).pack(side="left")
        ttk.Entry(self.drive_sa_row, textvariable=self.var_drive_sa, width=7).pack(side="left", padx=(4, 10))
        tk.Label(self.drive_sa_row, text="Tol ±:", bg=c["panel"], fg=c["text_muted"], font=FONT_SMALL).pack(side="left")
        ttk.Entry(self.drive_sa_row, textvariable=self.var_drive_sa_tol, width=7).pack(side="left", padx=(4, 0))

        # 2. Run queue
        card_runs = ModernCard(ctrl, "2. Telemetry File Queue", accent_color=c["cyan"], colors=c, pad_x=10, pad_y=8)
        card_runs.pack(fill="x", pady=(0, 8))

        self.runs_tree = ttk.Treeview(card_runs.body, columns=("file", "type", "cond", "pts"), show="headings", height=5)
        for key, title, w, anc in [
            ("file", "File", 145, "w"),
            ("type", "Type", 80, "center"),
            ("cond", "Filter", 110, "center"),
            ("pts", "Pts", 52, "e"),
        ]:
            self.runs_tree.heading(key, text=title)
            self.runs_tree.column(key, width=w, anchor=anc, stretch=(key == "file"))
        self.runs_tree.pack(fill="x", pady=(0, 6))
        style_treeview_zebra(self.runs_tree, c)

        r_btns = tk.Frame(card_runs.body, bg=c["panel"])
        r_btns.pack(fill="x", pady=(0, 6))
        ttk.Button(r_btns, text="+ Add Files...", style="Pill.TButton", command=self.add_runs).pack(side="left", padx=(0, 6))
        ttk.Button(r_btns, text="Remove", style="Pill.TButton", command=self.remove_selected_runs).pack(side="left", padx=(0, 6))
        ttk.Button(r_btns, text="Clear", style="Pill.TButton", command=self.clear_all_runs).pack(side="left")

        ttk.Button(card_runs.body, text="Process & Merge Telemetry Data", style="Accent.TButton", command=self.process_runs).pack(fill="x")

        # 3. Histogram inspector controls
        card_hist = ModernCard(ctrl, "3. Channel Distribution Inspector", colors=c, pad_x=10, pad_y=8)
        card_hist.pack(fill="x", pady=(0, 8))

        tk.Label(card_hist.body, text="Source:", bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL_BOLD, anchor="w").pack(fill="x")
        cb_src = ttk.Combobox(card_hist.body, textvariable=self.var_hist_source, values=["Cornering", "Drive/Brake"], state="readonly")
        cb_src.pack(fill="x", pady=(2, 6))
        cb_src.bind("<<ComboboxSelected>>", lambda _e: self._draw_histogram())

        tk.Label(card_hist.body, text="Channel:", bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL_BOLD, anchor="w").pack(fill="x")
        cb_ch = ttk.Combobox(card_hist.body, textvariable=self.var_hist_channel, values=[lbl for _, lbl in HISTOGRAM_CHANNELS], state="readonly")
        cb_ch.pack(fill="x", pady=(2, 6))
        cb_ch.bind("<<ComboboxSelected>>", lambda _e: self._draw_histogram())

        ttk.Checkbutton(
            card_hist.body,
            text="Focus central 1–99% quantile window",
            variable=self.var_hist_focus,
            command=self._draw_histogram,
        ).pack(anchor="w")

        # Right panel: KPI Tiles + Histogram plot
        right = ttk.Frame(body)
        right.pack(side="right", fill="both", expand=True)

        kpi_row = tk.Frame(right, bg=c["bg"])
        kpi_row.pack(fill="x", pady=(0, 8))
        for i in range(4):
            kpi_row.columnconfigure(i, weight=1)

        self.tile_d_corn = KpiTile(kpi_row, "CORNERING (FY)", "0 pts", accent=c["accent"], colors=c)
        self.tile_d_corn.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.tile_d_drive = KpiTile(kpi_row, "DRIVE / BRAKE (FX)", "0 pts", accent=c["cyan"], colors=c)
        self.tile_d_drive.grid(row=0, column=1, sticky="ew", padx=(0, 6))

        self.tile_d_fz = KpiTile(kpi_row, "FZ RANGE", "--", accent=c["ok"], colors=c)
        self.tile_d_fz.grid(row=0, column=2, sticky="ew", padx=(0, 6))

        self.tile_d_slip = KpiTile(kpi_row, "SA / SL RANGE", "--", accent=c["purple"], colors=c)
        self.tile_d_slip.grid(row=0, column=3, sticky="ew")

        hist_card = ModernCard(right, "Histogram & Data Quality Diagnostics", colors=c, pad_x=8, pad_y=6)
        hist_card.pack(fill="both", expand=True)

        self.lbl_hist_hover = tk.Label(
            hist_card.body,
            text="Hover over histogram bars to inspect bin range and sample counts.",
            bg=c["panel"],
            fg=c["text_muted"],
            font=FONT_SMALL,
            anchor="w",
        )
        self.lbl_hist_hover.pack(fill="x", pady=(0, 4))

        self.fig_hist = Figure(figsize=(7.4, 4.6), dpi=100, facecolor=c["panel"])
        self.canvas_hist = FigureCanvasTkAgg(self.fig_hist, master=hist_card.body)
        self.canvas_hist.get_tk_widget().configure(bg=c["panel"], highlightthickness=0)
        self.canvas_hist.get_tk_widget().pack(fill="both", expand=True)
        self.canvas_hist.mpl_connect("motion_notify_event", self._on_histogram_hover)
        self._draw_histogram()

    def _on_run_type_toggle(self):
        rt = self.var_run_type.get()
        if rt == "Cornering":
            self.lbl_param_target.config(text="IA [deg]:")
            self.var_target_param.set("0")
            self.var_tol_param.set("0.6")
            self.drive_sa_row.pack_forget()
        else:
            self.lbl_param_target.config(text="SL [-]:")
            self.var_target_param.set("0")
            self.var_tol_param.set("0.35")
            self.drive_sa_row.pack(fill="x", pady=(4, 0))

    def add_runs_from_paths(self, paths: Iterable[str], *, run_type: Optional[str] = None, process_now: bool = False) -> int:
        try:
            t_fz = float(self.var_target_fz.get().replace(",", "."))
            tol_fz = float(self.var_tol_fz.get().replace(",", "."))
            t_p = float(self.var_target_param.get().replace(",", "."))
            tol_p = float(self.var_tol_param.get().replace(",", "."))
        except ValueError as exc:
            messagebox.showerror("Invalid Filter Values", f"Check numeric condition filter entries:\n{exc}")
            return 0

        rt = run_type or self.var_run_type.get()
        t_sa = tol_sa = None
        if rt == "Drive/Brake":
            try:
                sa_txt = self.var_drive_sa.get().strip()
                if sa_txt:
                    t_sa = float(sa_txt.replace(",", "."))
                    tol_sa = float(self.var_drive_sa_tol.get().replace(",", "."))
            except ValueError:
                t_sa = tol_sa = None

        count = 0
        for p in paths:
            self.runs.append(
                RunConfig(
                    filepath=p,
                    test_type=rt,
                    target_fz=t_fz,
                    tol_fz=tol_fz,
                    target_param=t_p,
                    tol_param=tol_p,
                    target_sa=t_sa,
                    tol_sa=tol_sa,
                )
            )
            count += 1
        self._invalidate_processed_data()
        self._refresh_run_list()
        if process_now and count > 0:
            self.process_runs()
        elif count > 0:
            self.set_status(f"Queued {count} telemetry file(s) (click 'Process & Merge Telemetry Data').")
        return count

    def add_runs(self):
        paths = filedialog.askopenfilenames(
            title="Select Tire Telemetry Files (.mat / .csv / .txt / .dat)",
            filetypes=[
                ("Tire Telemetry (*.mat;*.csv;*.txt;*.dat;*.tsv)", "*.mat *.csv *.txt *.dat *.tsv"),
                ("MATLAB TTC (*.mat)", "*.mat"),
                ("CSV / ASCII (*.csv;*.txt;*.dat)", "*.csv *.txt *.dat *.tsv"),
                ("All Files", "*.*"),
            ],
        )
        if not paths:
            return
        self.add_runs_from_paths(paths, process_now=False)

    def remove_selected_runs(self):
        sel = self.runs_tree.selection()
        if not sel:
            return
        indices = sorted((int(iid) for iid in sel), reverse=True)
        for idx in indices:
            if 0 <= idx < len(self.runs):
                del self.runs[idx]
        self._invalidate_processed_data()
        self._refresh_run_list()

    def clear_all_runs(self):
        self.runs.clear()
        self._invalidate_processed_data()
        self._refresh_run_list()
        self.set_status("Cleared telemetry run queue.")

    def _invalidate_processed_data(self):
        self.df_cornering = None
        self.df_drive = None
        self._refresh_data_kpis()
        self._draw_histogram()
        self._update_telemetry_badges()

    def _refresh_run_list(self):
        for iid in self.runs_tree.get_children():
            self.runs_tree.delete(iid)
        for idx, r in enumerate(self.runs):
            p_lbl = f"IA={r.target_param:g}°" if r.test_type == "Cornering" else f"SL±{r.tol_param:g}"
            cond = f"Fz={r.target_fz:g}±{r.tol_fz:g}, {p_lbl}"
            self.runs_tree.insert(
                "",
                "end",
                iid=str(idx),
                values=(os.path.basename(r.filepath), r.test_type, cond, r.n_points if r.n_points else "--"),
                tags=("even" if idx % 2 == 0 else "odd",),
            )

    def process_runs(self):
        if not self.runs:
            messagebox.showwarning("Queue Empty", "Add telemetry files via Import... or load the FSAE Demo.")
            return

        c_frames = []
        d_frames = []
        errors = []
        for r in self.runs:
            try:
                df = TireDataManager.load_and_filter(r)
                r.n_points = len(df)
                if r.test_type == "Cornering":
                    c_frames.append(df)
                else:
                    d_frames.append(df)
            except Exception as exc:
                errors.append(f"{os.path.basename(r.filepath)}: {exc}")

        self.df_cornering = pd.concat(c_frames, ignore_index=True) if c_frames else None
        self.df_drive = pd.concat(d_frames, ignore_index=True) if d_frames else None

        self._refresh_run_list()
        self._refresh_data_kpis()
        self._draw_histogram()
        self._update_telemetry_badges()

        if errors:
            messagebox.showwarning("Processing Warnings", "\n".join(errors[:6]))
        nc = len(self.df_cornering) if self.df_cornering is not None else 0
        nd = len(self.df_drive) if self.df_drive is not None else 0
        self.set_status(f"Processed telemetry: {nc:,} Cornering samples, {nd:,} Drive/Brake samples.")

    def _refresh_data_kpis(self):
        nc = len(self.df_cornering) if self.df_cornering is not None else 0
        nd = len(self.df_drive) if self.df_drive is not None else 0
        self.tile_d_corn.set_value(f"{nc:,} pts")
        self.tile_d_drive.set_value(f"{nd:,} pts")

        fz_vals = []
        if self.df_cornering is not None and not self.df_cornering.empty:
            fz_vals.append(self.df_cornering["FZ"].to_numpy())
        if self.df_drive is not None and not self.df_drive.empty:
            fz_vals.append(self.df_drive["FZ"].to_numpy())
        if fz_vals:
            all_fz = np.concatenate(fz_vals)
            self.tile_d_fz.set_value(f"{np.nanmin(all_fz):.0f}–{np.nanmax(all_fz):.0f} N")
        else:
            self.tile_d_fz.set_value("--")

        slip_parts = []
        if self.df_cornering is not None and not self.df_cornering.empty and "SA" in self.df_cornering:
            slip_parts.append(f"SA±{np.nanmax(np.abs(self.df_cornering['SA'])):.1f}°")
        if self.df_drive is not None and not self.df_drive.empty and "SL" in self.df_drive:
            slip_parts.append(f"SL±{np.nanmax(np.abs(self.df_drive['SL'])):.2f}")
        self.tile_d_slip.set_value(" | ".join(slip_parts) if slip_parts else "--")

    def _draw_histogram(self):
        c = self.c
        self.fig_hist.clear()
        self.fig_hist.patch.set_facecolor(c["panel"])
        ax = self.fig_hist.add_subplot(111)
        self._hist_cache = None

        src = self.var_hist_source.get()
        df = self.df_cornering if src == "Cornering" else self.df_drive
        lbl_chosen = self.var_hist_channel.get()
        ch_code = next((code for code, lbl in HISTOGRAM_CHANNELS if lbl == lbl_chosen), "SA")

        if df is None or df.empty or ch_code not in df.columns:
            ax.text(0.5, 0.5, f"No processed data for '{src}' ({ch_code})", ha="center", va="center", color=c["text_muted"])
            style_axes_2d(ax, zero_lines=False, c=c)
            self.fig_hist.tight_layout(pad=1.4)
            self.canvas_hist.draw_idle()
            return

        h = make_histogram(df[ch_code].to_numpy(), focus=bool(self.var_hist_focus.get()))
        if h.counts.size == 0:
            ax.text(0.5, 0.5, "No finite samples in selected channel.", ha="center", va="center", color=c["text_muted"])
            style_axes_2d(ax, zero_lines=False, c=c)
            self.fig_hist.tight_layout(pad=1.4)
            self.canvas_hist.draw_idle()
            return

        widths = np.diff(h.edges)
        ax.bar(h.edges[:-1], h.counts, width=widths, align="edge", color=CHART_PALETTE[0], edgecolor=c["panel"], alpha=0.85)
        self._hist_cache = {"ax": ax, "edges": h.edges, "counts": h.counts, "ch": ch_code}

        tail_txt = ""
        if h.below or h.above:
            tail_txt = f" | Outside window: -{h.below} / +{h.above}"
        style_axes_2d(
            ax,
            title=f"Distribution: {lbl_chosen} ({src}, n={h.finite:,}{tail_txt})",
            xlabel=lbl_chosen,
            ylabel="Sample Count",
            zero_lines=False,
            c=c,
        )
        self.fig_hist.tight_layout(pad=1.4)
        self.canvas_hist.draw_idle()

    def _on_histogram_hover(self, event):
        if not self._hist_cache or event.inaxes is not self._hist_cache["ax"] or event.xdata is None:
            return
        edges = self._hist_cache["edges"]
        counts = self._hist_cache["counts"]
        x = float(event.xdata)
        idx = int(np.searchsorted(edges, x, side="right")) - 1
        if 0 <= idx < len(counts):
            self.lbl_hist_hover.config(
                text=f"Channel {self._hist_cache['ch']}: bin [{edges[idx]:.3f}, {edges[idx+1]:.3f}] -> {int(counts[idx]):,} samples",
                fg=self.c["cyan"],
            )

    def _save_histogram_plot(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")], initialfile="PUTM_Histogram_TTC.png")
        if path:
            self.fig_hist.savefig(path, dpi=220, facecolor=self.fig_hist.get_facecolor())
            self.set_status(f"Saved histogram plot: {path}")

    # ------------------------------------------------------------------
    # STEP 2: MODEL FITTING (AUTOPILOT & OPTIMIZER)
    # ------------------------------------------------------------------
    def _build_step2_fit(self):
        c = self.c
        view = ttk.Frame(self.content_area)
        self._views["fit"] = view

        StepHeader(
            view,
            "STEP 2 / 5",
            "Model Fitting (Autopilot & Optimizer)",
            "Extract a complete MF 5.1 tire model in 1 click or fit individual curves with R^2 and NRMSE metrics.",
            badge_kind="accent",
            next_text="Apply Fits & Next: Model (Step 3) ->",
            next_command=self._apply_fit_and_next,
            colors=c,
        ).pack(fill="x", pady=(0, 8))

        self.fit_empty_banner = StatusBanner(
            view,
            "Load telemetry data first (click 'Import...' -> Telemetry / FSAE Demo) to unlock model fitting.",
            kind="warn",
            button_text="Open Import...",
            button_command=lambda: self.open_import_dialog("telemetry"),
            second_button_text="Load FSAE Demo",
            second_button_command=self.load_fsae_demo,
            colors=c,
        )

        self._fit_body = ttk.Frame(view)
        self._fit_body.pack(fill="both", expand=True)

        left_wrap = tk.Frame(self._fit_body, bg=c["bg"], width=425)
        left_wrap.pack(side="left", fill="y", padx=(0, 10))
        left_wrap.pack_propagate(False)

        self._fit_scroll = ScrollableFrame(left_wrap, bg=c["bg"], padx=2, pady=2)
        self._fit_scroll.pack(fill="both", expand=True)
        ctrl = self._fit_scroll.inner

        # 1. Autopilot MF 5.1 Card
        card_auto = ModernCard(
            ctrl,
            "1. Autopilot MF 5.1 (Full 1-Click Extraction)",
            badge="AUTO",
            badge_color="cyan",
            accent_color=c["cyan"],
            colors=c,
            pad_x=10,
            pad_y=8,
        )
        card_auto.pack(fill="x", pady=(0, 8))

        tk.Label(
            card_auto.body,
            text="Automatically fits Fy(SA, Fz), Fx(SL, Fz), Mz(SA), Camber (IA), and Combined Slip directly into PAC2002.",
            bg=c["panel"],
            fg=c["text_muted"],
            font=FONT_SMALL,
            wraplength=375,
            justify="left",
        ).pack(fill="x", pady=(0, 6))

        p_row = tk.Frame(card_auto.body, bg=c["panel"])
        p_row.pack(fill="x", pady=(0, 6))
        tk.Label(p_row, text="Opt. pressure filter [kPa]:", bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL).pack(side="left")
        ttk.Entry(p_row, textvariable=self.var_autopilot_p, width=9).pack(side="right")

        self.btn_autopilot = ttk.Button(
            card_auto.body,
            text="Run Autopilot MF 5.1 (All Channels)",
            style="Cyan.TButton",
            command=self.run_autopilot_step2,
        )
        self.btn_autopilot.pack(fill="x", pady=(0, 6))

        self.prog_autopilot = ttk.Progressbar(card_auto.body, mode="indeterminate", style="Horizontal.TProgressbar")
        self.prog_autopilot.pack(fill="x", pady=(0, 4))

        self.lbl_autopilot_res = tk.Label(
            card_auto.body,
            text="Ready — load telemetry data in Step 1 first.",
            bg=c["panel"],
            fg=c["text_secondary"],
            font=FONT_SMALL,
            wraplength=375,
            justify="left",
            anchor="w",
        )
        self.lbl_autopilot_res.pack(fill="x")

        # 2. Curve Optimizer Card
        card_opt = ModernCard(
            ctrl,
            "2. Precision Optimizer (Single Model)",
            badge="FIT",
            badge_color="accent",
            accent_color=c["accent"],
            colors=c,
            pad_x=10,
            pad_y=8,
        )
        card_opt.pack(fill="x", pady=(0, 8))

        tk.Label(card_opt.body, text="Select model:", bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL_BOLD, anchor="w").pack(fill="x")
        cb_m = ttk.Combobox(card_opt.body, textvariable=self.var_fit_model, values=[lbl for _, lbl in FIT_MODELS], state="readonly")
        cb_m.pack(fill="x", pady=(2, 8))

        opt_grid = tk.Frame(card_opt.body, bg=c["panel"])
        opt_grid.pack(fill="x", pady=(0, 8))
        opt_grid.columnconfigure(1, weight=1)
        opt_grid.columnconfigure(3, weight=1)

        tk.Label(opt_grid, text="Fz Slice [N]:", bg=c["panel"], fg=c["text_muted"], font=FONT_SMALL).grid(row=0, column=0, sticky="w")
        ttk.Entry(opt_grid, textvariable=self.var_fit_fz_slice, width=8).grid(row=0, column=1, sticky="ew", padx=(4, 10))
        tk.Label(opt_grid, text="Shape C0:", bg=c["panel"], fg=c["text_muted"], font=FONT_SMALL).grid(row=0, column=2, sticky="w")
        ttk.Entry(opt_grid, textvariable=self.var_fit_c0, width=7).grid(row=0, column=3, sticky="ew", padx=(4, 0))

        ttk.Button(card_opt.body, text="Fit Selected Model", style="Accent.TButton", command=self.fit_model).pack(fill="x", pady=(0, 5))
        ttk.Button(card_opt.body, text="Matrix Auto-Fit (Fz x IA x P Grid)...", command=self.open_matrix_autofit).pack(fill="x", pady=(0, 5))
        ttk.Button(card_opt.body, text="Send Fits -> Step 3 (MF 5.1)", style="Cyan.TButton", command=self.send_fit_to_mf51).pack(fill="x")

        # 3. Saved Fits status card
        card_saved = ModernCard(ctrl, "3. Session Fits", colors=c, pad_x=10, pad_y=8)
        card_saved.pack(fill="x", pady=(0, 8))

        self.lbl_saved_fits = tk.Label(
            card_saved.body,
            text="No fitted models saved yet.",
            bg=c["panel"],
            fg=c["text_muted"],
            font=FONT_SMALL,
            wraplength=375,
            justify="left",
            anchor="w",
        )
        self.lbl_saved_fits.pack(fill="x")

        # Right panel: KPI Tiles + Plot + Parameter Treeview
        right = ttk.Frame(self._fit_body)
        right.pack(side="right", fill="both", expand=True)

        kpi_row = tk.Frame(right, bg=c["bg"])
        kpi_row.pack(fill="x", pady=(0, 8))
        for i in range(4):
            kpi_row.columnconfigure(i, weight=1)

        self.tile_fit_r2 = KpiTile(kpi_row, "GOODNESS R^2", "--", accent=c["ok"], colors=c)
        self.tile_fit_r2.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.tile_fit_rmse = KpiTile(kpi_row, "FIT RMSE", "--", accent=c["accent"], colors=c)
        self.tile_fit_rmse.grid(row=0, column=1, sticky="ew", padx=(0, 6))

        self.tile_fit_nrmse = KpiTile(kpi_row, "NRMSE [%]", "--", accent=c["cyan"], colors=c)
        self.tile_fit_nrmse.grid(row=0, column=2, sticky="ew", padx=(0, 6))

        self.tile_fit_pts = KpiTile(kpi_row, "MODEL / SAMPLES", "--", accent=c["purple"], colors=c)
        self.tile_fit_pts.grid(row=0, column=3, sticky="ew")

        plot_card = ModernCard(right, "Measured Data vs Magic Formula Fit", colors=c, pad_x=8, pad_y=6)
        plot_card.pack(fill="both", expand=True, pady=(0, 8))

        self.fig_fit = Figure(figsize=(7.5, 3.8), dpi=100, facecolor=c["panel"])
        self.canvas_fit = FigureCanvasTkAgg(self.fig_fit, master=plot_card.body)
        self.canvas_fit.get_tk_widget().configure(bg=c["panel"], highlightthickness=0)
        self.canvas_fit.get_tk_widget().pack(fill="both", expand=True)

        param_card = ModernCard(right, "Fitted Model Coefficients", colors=c, pad_x=8, pad_y=6)
        param_card.pack(fill="x")

        cols = ("param", "val", "unit", "desc")
        self.tree_fit_params = ttk.Treeview(param_card.body, columns=cols, show="headings", height=5)
        for key, title, width, anchor in [
            ("param", "Parameter", 110, "w"),
            ("val", "Value", 120, "e"),
            ("unit", "Unit", 110, "center"),
            ("desc", "Engineering Description", 320, "w"),
        ]:
            self.tree_fit_params.heading(key, text=title)
            self.tree_fit_params.column(key, width=width, anchor=anchor, stretch=(key == "desc"))
        self.tree_fit_params.pack(fill="x")
        style_treeview_zebra(self.tree_fit_params, c)

        self._reset_fit_plot()

    def _apply_fit_and_next(self):
        if self.fit_results:
            self.send_fit_to_mf51(silent=True)
        self.switch_tab("model")

    def run_autopilot_step2(self):
        if self._autopilot_running:
            return
        if (self.df_cornering is None or self.df_cornering.empty) and (self.df_drive is None or self.df_drive.empty) and not self.runs:
            messagebox.showwarning("No Data", "Import telemetry files or load FSAE Demo first.")
            return

        target_p = None
        p_txt = self.var_autopilot_p.get().strip()
        if p_txt:
            try:
                target_p = float(p_txt.replace(",", "."))
            except ValueError:
                messagebox.showerror("Invalid Pressure", "Enter pressure in kPa (e.g. 82.7) or leave blank.")
                return

        self._autopilot_running = True
        self.btn_autopilot.state(["disabled"])
        self.prog_autopilot.start(12)
        self.lbl_autopilot_res.config(text="Running multi-channel MF 5.1 optimization in background...", fg=self.c["cyan"])

        model_copy = self.model.copy()
        runs_snap = list(self.runs)
        df_c_snap = self.df_cornering
        df_d_snap = self.df_drive

        def worker():
            try:
                res = AutoMF51Extractor.extract_from_app_data(
                    model_copy,
                    runs_snap,
                    df_cornering=df_c_snap,
                    df_drive=df_d_snap,
                    target_pressure=target_p,
                )
                err = None
            except Exception as exc:
                res = None
                err = exc
            self.root.after(0, lambda: self._on_autopilot_done(model_copy, res, err))

        threading.Thread(target=worker, daemon=True).start()

    def _on_autopilot_done(self, fitted_model: MF51Model, res, err):
        self._autopilot_running = False
        self.prog_autopilot.stop()
        self.btn_autopilot.state(["!disabled"])

        if err is not None:
            self.lbl_autopilot_res.config(text=f"Autopilot error: {err}", fg=self.c["danger"])
            messagebox.showerror("Autopilot MF 5.1 Error", str(err))
            return

        if not res.success:
            self.lbl_autopilot_res.config(text=res.message, fg=self.c["warn"])
            messagebox.showwarning("Autopilot MF 5.1", res.message)
            return

        self.model = fitted_model
        self._active_model_name = "Autopilot MF 5.1 (TTC)"
        n_p = len(res.parameters)
        msg_short = f"✓ Done ({n_p} MF 5.1 params): PDY1={fitted_model.get('PDY1'):.3f}, PDY2={fitted_model.get('PDY2'):.3f}, PDX1={fitted_model.get('PDX1'):.3f}"
        self.lbl_autopilot_res.config(text=msg_short, fg=self.c["ok"])
        if hasattr(self, "mf51_tab"):
            self.mf51_tab._set_info(
                "--- AUTOMATED MULTI-FZ EXTRACTION REPORT (MF 5.1) ---\n" + "\n".join(res.diagnostics_log)
            )
        self.set_status(msg_short)
        self.on_model_updated()

    def _selected_fit_code(self) -> str:
        lbl = self.var_fit_model.get()
        return next((code for code, name in FIT_MODELS if name == lbl), "Fy0")

    def fit_model(self):
        code = self._selected_fit_code()
        fz_txt = self.var_fit_fz_slice.get().strip()
        fz_filter = None
        if fz_txt:
            try:
                fz_filter = float(fz_txt.replace(",", "."))
            except ValueError:
                messagebox.showerror("Invalid Fz Slice", "Enter numeric Fz load in N or leave blank.")
                return

        try:
            c0 = float(self.var_fit_c0.get().replace(",", "."))
        except ValueError:
            c0 = 1.30

        try:
            if code in ("Fy0", "Mz0"):
                out = self._fit_pure_model(code, fz_filter, c0)
            elif code == "Fy_load":
                out = self._fit_load_model()
            elif code == "Fx0":
                out = self._fit_pure_model("Fx0", fz_filter, c0)
            elif code == "CombinedSlip":
                out = self._fit_combined_model()
            elif code == "Mu_Fz":
                out = self._fit_mu_vs_fz()
            else:
                out = self._fit_envelope()
        except Exception as exc:
            messagebox.showerror("Fitting Error", str(exc))
            return

        if out is not None:
            self.set_status(f"Fitted {code} model (R^2={out.get('r2', 0.0):.4f}).")
            self._update_telemetry_badges()

    def _slice_df(self, df: Optional[pd.DataFrame], fz_filter: Optional[float]) -> pd.DataFrame:
        if df is None or df.empty:
            raise ValueError("No telemetry data loaded — import files or load FSAE Demo first.")
        if fz_filter is None:
            return df
        sub = df[np.abs(df["FZ"] - fz_filter) <= 150.0]
        if len(sub) < 20:
            raise ValueError(f"Insufficient points near Fz={fz_filter:.0f} N ({len(sub)} pts).")
        return sub

    def _fit_pure_model(self, code: str, fz_filter: Optional[float], c0: float) -> dict:
        if code == "Fx0":
            df = self._slice_df(self.df_drive, fz_filter)
            x_raw = df["SL"].to_numpy(dtype=float)
            y_raw = df["FX"].to_numpy(dtype=float)
            out = fit_pure_magic(x_raw, y_raw, model_name="Fx0", slip_unit="ratio", shape_factor=c0)
            x_plot, x_label, y_label = x_raw, "Slip Ratio SL [-]", "Fx [N]"
        elif code == "Mz0":
            df = self._slice_df(self.df_cornering, fz_filter)
            x_deg = df["SA"].to_numpy(dtype=float)
            y_raw = df["MZ"].to_numpy(dtype=float)
            out = fit_pure_magic(np.radians(x_deg), y_raw, model_name="Mz0", slip_unit="rad", shape_factor=c0)
            x_plot, x_label, y_label = x_deg, "SA [deg]", "Mz [Nm]"
        else:
            df = self._slice_df(self.df_cornering, fz_filter)
            x_deg = df["SA"].to_numpy(dtype=float)
            y_raw = -df["FY"].to_numpy(dtype=float)
            out = fit_pure_magic(np.radians(x_deg), y_raw, model_name="Fy0", slip_unit="rad", shape_factor=c0)
            x_plot, x_label, y_label = x_deg, "SA [deg]", "Fy [N]"

        record = {
            "model": code,
            "parameters": dict(out.parameters),
            **{k: float(v) for k, v in out.parameters.items() if isinstance(v, (int, float))},
            "r2": float(out.diagnostics.get("r2", 0.0)),
            "rmse": float(out.diagnostics.get("rmse", 0.0)),
            "nrmse": 100.0 * float(out.diagnostics.get("nrmse", 0.0)),
            "n_points": int(out.diagnostics.get("n_points", len(out.y))),
            "x_plot": x_plot,
            "y_meas": out.y,
            "y_pred": out.prediction,
            "x_label": x_label,
            "y_label": y_label,
        }
        self._record_fit(code, record)
        return record

    def _fit_load_model(self) -> dict:
        df = self._slice_df(self.df_cornering, None)
        alpha = np.radians(df["SA"].to_numpy(dtype=float))
        fz = df["FZ"].to_numpy(dtype=float)
        force = -df["FY"].to_numpy(dtype=float)
        out = fit_load_sensitive(alpha, fz, force)
        record = {
            "model": "Fy_load",
            "parameters": {k: v for k, v in out.parameters.items() if k != "model"},
            **{k: float(v) for k, v in out.parameters.items() if isinstance(v, (int, float))},
            "r2": float(out.diagnostics.get("r2", 0.0)),
            "rmse": float(out.diagnostics.get("rmse", 0.0)),
            "nrmse": 100.0 * float(out.diagnostics.get("nrmse", 0.0)),
            "n_points": int(out.diagnostics.get("n_points", len(out.y))),
            "x_plot": df["SA"].to_numpy(dtype=float),
            "fz_plot": fz,
            "y_meas": out.y,
            "y_pred": out.prediction,
            "x_label": "SA [deg]",
            "y_label": "Fy [N]",
        }
        self._record_fit("Fy_load", record)
        return record

    def _fit_combined_model(self) -> dict:
        df = self._slice_df(self.df_drive, None)
        out = fit_combined_slip(df)
        record = {
            "model": "CombinedSlip",
            "parameters": {k: v for k, v in out.parameters.items() if k != "model"},
            **{k: float(v) for k, v in out.parameters.items() if isinstance(v, (int, float))},
            "r2": float(out.diagnostics.get("r2", 0.0)),
            "rmse": float(out.diagnostics.get("rmse", 0.0)),
            "nrmse": 100.0 * float(out.diagnostics.get("nrmse", 0.0)),
            "n_points": int(out.diagnostics.get("n_points", len(out.y))),
            "x_plot": out.x,
            "y_meas": out.y,
            "y_pred": out.prediction,
            "x_label": "Slip Ratio SL [-]",
            "y_label": "Reduction Factor Gyk [-]",
        }
        self._record_fit("CombinedSlip", record)
        return record

    def _fit_mu_vs_fz(self) -> dict:
        df_c = self._slice_df(self.df_cornering, None)
        fz_levels = np.unique(np.round(df_c["FZ"] / 150.0) * 150.0)
        pts_y = []
        for fz_c in fz_levels:
            sub = df_c[np.abs(df_c["FZ"] - fz_c) <= 120.0]
            if len(sub) >= 15:
                mu_y = float(np.percentile(np.abs(sub["FY"]), 98) / max(np.median(sub["FZ"]), 1.0))
                pts_y.append((float(np.median(sub["FZ"])), mu_y))
        if len(pts_y) < 2:
            raise ValueError("At least 2 distinct Fz levels are required to fit Mu_Fz.")
        fz_arr = np.array([p[0] for p in pts_y])
        mu_arr = np.array([p[1] for p in pts_y])
        fz0 = float(np.median(fz_arr))
        dfz = (fz_arr - fz0) / fz0
        pdy2, pdy1 = np.polyfit(dfz, mu_arr, 1)

        pdx1, pdx2 = float(pdy1 * 1.06), float(pdy2)
        if self.df_drive is not None and not self.df_drive.empty:
            pts_x = []
            for fz_d in np.unique(np.round(self.df_drive["FZ"] / 150.0) * 150.0):
                sub_d = self.df_drive[np.abs(self.df_drive["FZ"] - fz_d) <= 120.0]
                if len(sub_d) >= 15:
                    pts_x.append((float(np.median(sub_d["FZ"])), float(np.percentile(np.abs(sub_d["FX"]), 98) / max(np.median(sub_d["FZ"]), 1.0))))
            if len(pts_x) >= 2:
                fxd = (np.array([p[0] for p in pts_x]) - fz0) / fz0
                pdx2, pdx1 = np.polyfit(fxd, np.array([p[1] for p in pts_x]), 1)

        pred = pdy1 + pdy2 * dfz
        ss_res = float(np.sum((mu_arr - pred) ** 2))
        ss_tot = float(np.sum((mu_arr - np.mean(mu_arr)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
        rmse = float(np.sqrt(np.mean((mu_arr - pred) ** 2)))

        params = {"pDy1": float(pdy1), "pDy2": float(pdy2), "pDx1": float(pdx1), "pDx2": float(pdx2), "fz0": fz0}
        record = {
            "model": "mu_vs_fz",
            "parameters": params,
            **params,
            "r2": r2,
            "rmse": rmse,
            "nrmse": 100.0 * rmse / max(float(np.mean(mu_arr)), 1e-6),
            "n_points": len(pts_y),
            "x_plot": fz_arr,
            "y_meas": mu_arr,
            "y_pred": pred,
            "x_label": "Fz [N]",
            "y_label": "Peak mu_y [-]",
        }
        self._record_fit("mu_vs_fz", record)
        return record

    def _fit_envelope(self) -> dict:
        df_c = self._slice_df(self.df_cornering, None)
        fy_peak = float(np.percentile(np.abs(df_c["FY"]), 99))
        fz_med = float(np.median(df_c["FZ"]))
        fx_peak = float(np.percentile(np.abs(self.df_drive["FX"]), 99)) if self.df_drive is not None and not self.df_drive.empty else fy_peak * 1.05
        th = np.linspace(0, 2 * np.pi, 180)
        fy_e = fy_peak * np.sin(th)
        fx_e = fx_peak * np.cos(th)
        params = {"Fy_peak": fy_peak, "Fx_peak": fx_peak, "Fz_med": fz_med, "mu_y": fy_peak / max(fz_med, 1.0), "mu_x": fx_peak / max(fz_med, 1.0)}
        record = {
            "model": "Envelope",
            "parameters": params,
            **params,
            "r2": 1.0,
            "rmse": 0.0,
            "nrmse": 0.0,
            "n_points": len(df_c),
            "x_plot": fy_e,
            "y_meas": fx_e,
            "y_pred": fx_e,
            "x_label": "Fy [N]",
            "y_label": "Fx [N]",
        }
        self._record_fit("Envelope", record)
        return record

    def _record_fit(self, key: str, record: dict):
        self.fit_results[key] = record
        self.fit_result = record
        self._display_fit_kpis(record)
        self._draw_saved_fit(record)
        self.lbl_saved_fits.config(
            text="Saved models: " + ", ".join(self.fit_results.keys()),
            fg=self.c["cyan"],
        )

    def _display_fit_kpis(self, record: dict):
        r2 = record.get("r2", 0.0)
        rmse = record.get("rmse", 0.0)
        nrmse = record.get("nrmse", 0.0)
        self.tile_fit_r2.set_value(f"{r2:.4f}", self.c["ok"] if r2 >= 0.90 else self.c["warn"])
        self.tile_fit_rmse.set_value(f"{rmse:.2f}")
        self.tile_fit_nrmse.set_value(f"{nrmse:.2f}%")
        self.tile_fit_pts.set_value(f"{record.get('model')} ({record.get('n_points', 0)})")

        for iid in self.tree_fit_params.get_children():
            self.tree_fit_params.delete(iid)
        params = record.get("parameters", {})
        for idx, (k, v) in enumerate(params.items()):
            if not isinstance(v, (int, float)):
                continue
            unit, desc = PARAM_DESCRIPTIONS.get(k, ("-", "Fitted parameter"))
            self.tree_fit_params.insert(
                "",
                "end",
                values=(k, f"{float(v):.5g}", unit, desc),
                tags=("even" if idx % 2 == 0 else "odd",),
            )

    def _reset_fit_plot(self):
        c = self.c
        self.fig_fit.clear()
        self.fig_fit.patch.set_facecolor(c["panel"])
        ax = self.fig_fit.add_subplot(111)
        ax.text(0.5, 0.5, "Choose a model on the left and click 'Fit Selected Model'\nor launch 'Run Autopilot MF 5.1 (All Channels)'", ha="center", va="center", color=c["text_muted"])
        style_axes_2d(ax, zero_lines=False, c=c)
        self.fig_fit.tight_layout(pad=1.4)
        self.canvas_fit.draw_idle()

    def _draw_saved_fit(self, record: dict):
        c = self.c
        self.fig_fit.clear()
        self.fig_fit.patch.set_facecolor(c["panel"])
        ax = self.fig_fit.add_subplot(111)

        xp = np.asarray(record["x_plot"])
        ym = np.asarray(record["y_meas"])
        yp = np.asarray(record["y_pred"])
        mname = record["model"]

        if mname == "Envelope":
            ax.plot(xp, yp, color=CHART_PALETTE[0], linewidth=2.4, label="G-G Envelope")
            ax.set_aspect("equal", adjustable="datalim")
        else:
            step = max(1, len(xp) // 900)
            ax.scatter(xp[::step], ym[::step], color=c["cyan"], s=14, alpha=0.40, label="TTC Telemetry")
            order = np.argsort(xp)
            if mname == "Fy_load":
                fz_arr = np.asarray(record.get("fz_plot", []))
                if fz_arr.size == len(xp):
                    levels = np.unique(np.round(fz_arr / 150.0) * 150.0)
                    for idx, lv in enumerate(levels[:5]):
                        mask = np.abs(fz_arr - lv) <= 120.0
                        if mask.any():
                            ord_l = np.argsort(xp[mask])
                            ax.plot(xp[mask][ord_l], yp[mask][ord_l], color=CHART_PALETTE[idx % len(CHART_PALETTE)], linewidth=2.1, label=f"Fit Fz~{lv:.0f}N")
                else:
                    ax.scatter(xp, yp, color=c["accent"], s=10, label="Prediction")
            else:
                ax.plot(xp[order], yp[order], color=c["accent"], linewidth=2.4, label=f"Fit {mname} (R^2={record.get('r2', 0):.3f})")

        style_axes_2d(ax, title=f"Fit {mname} | R^2={record.get('r2', 0):.4f} | RMSE={record.get('rmse', 0):.2f}", xlabel=record.get("x_label", "X"), ylabel=record.get("y_label", "Y"), c=c)
        style_legend(ax, c=c)
        self.fig_fit.tight_layout(pad=1.4)
        self.canvas_fit.draw_idle()

    def send_fit_to_mf51(self, silent: bool = False):
        if not self.fit_results:
            if not silent:
                messagebox.showinfo("No Fits", "Fit at least one model in Step 2 first.")
            return
        payload = dict(self.fit_results)
        if "CombinedSlip" in payload and "Combined" not in payload:
            payload["Combined"] = payload["CombinedSlip"]
        if "Fx0" in payload:
            fx_rec = dict(payload["Fx0"])
            fx_p = dict(fx_rec.get("parameters", fx_rec))
            for src, dst in (("Bx", "B"), ("Cx", "C"), ("Dx", "D"), ("Ex", "E"), ("Shx", "Sh"), ("Svx", "Sv")):
                if src in fx_p and dst not in fx_p:
                    fx_p[dst] = fx_p[src]
            fx_rec["parameters"] = fx_p
            payload["Fx0"] = fx_rec
        imported = self.model.import_from_fit_results(payload)
        self._active_model_name = "Fit -> MF 5.1 Model"
        self.on_model_updated()
        if not silent:
            messagebox.showinfo("Synced to Step 3", "Imported parameters into active MF 5.1 model:\n" + "\n".join(imported))
        self.set_status("Updated MF 5.1 model from Step 2 fit results.")

    def open_matrix_autofit(self):
        def _on_apply(target, fit_dict):
            key = target
            self.fit_results[key] = fit_dict
            self.fit_result = fit_dict
            self._display_fit_kpis(fit_dict)
            self._update_telemetry_badges()

        MatrixAutoFitDialog(self.root, self.runs, raw_cache=self.raw_cache, on_fit_selected=_on_apply, colors=self.c)

    def save_current_plot(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")], initialfile="PUTM_Fit_Plot.png")
        if path:
            self.fig_fit.savefig(path, dpi=220, facecolor=self.fig_fit.get_facecolor())
            self.set_status(f"Saved fit plot: {path}")

    # ------------------------------------------------------------------
    # STEPS 3 - 7
    # ------------------------------------------------------------------
    def _build_step3_model(self):
        view = ttk.Frame(self.content_area)
        self._views["model"] = view
        self.mf51_tab = MF51Tab(
            view,
            colors=self.c,
            set_status=self.set_status,
            get_fit_results=lambda: self.fit_results,
            get_runs=lambda: self.runs,
            get_df_cornering=lambda: self.df_cornering,
            get_df_drive=lambda: self.df_drive,
            get_model=self.get_model,
            set_model=self.set_model,
            on_model_updated=self.on_model_updated,
            on_next_step=lambda: self.switch_tab("sim3d"),
            show_step_header=True,
        )
        self.mf51_tab.pack(fill="both", expand=True)

    def _build_step4_sim3d(self):
        view = ttk.Frame(self.content_area)
        self._views["sim3d"] = view
        self.hub_tab = HubTab(
            view,
            colors=self.c,
            set_status=self.set_status,
            get_model=self.get_model,
            set_model=self.set_model,
            on_model_updated=self.on_model_updated,
            get_df_cornering=lambda: self.df_cornering,
            get_df_drive=lambda: self.df_drive,
            get_runs=lambda: self.runs,
            on_data_loaded=self._on_demo_loaded_from_hub,
            on_edit_model=lambda: self.switch_tab("model"),
            on_next_step=lambda: self.switch_tab("compare"),
            show_step_header=True,
        )
        self.hub_tab.pack(fill="both", expand=True)

    def _on_demo_loaded_from_hub(self, df_c: pd.DataFrame, df_d: pd.DataFrame):
        self.df_cornering = df_c
        self.df_drive = df_d
        self._refresh_data_kpis()
        self._draw_histogram()
        self._update_telemetry_badges()

    def _build_step5_compare(self):
        view = ttk.Frame(self.content_area)
        self._views["compare"] = view
        self.tir_tab = TirAnalyzerTab(
            view,
            colors=self.c,
            set_status=self.set_status,
            get_active_model=self.get_model,
            on_next_step=lambda: self.switch_tab("stiffness"),
            show_step_header=True,
        )
        self.tir_tab.pack(fill="both", expand=True)

    def _build_tool6_stiffness(self):
        view = ttk.Frame(self.content_area)
        self._views["stiffness"] = view
        self.stiffness_tab = StiffnessTab(
            view,
            colors=self.c,
            set_status=self.set_status,
            get_df_cornering=lambda: self.df_cornering,
            get_runs=lambda: self.runs,
            show_step_header=True,
        )
        self.stiffness_tab.pack(fill="both", expand=True)

    def _build_tool7_sketch(self):
        view = ttk.Frame(self.content_area)
        self._views["sketch"] = view
        self.sketch_tab = SimplePacejkaTab(
            view,
            colors=self.c,
            set_status=self.set_status,
            get_fit_results=lambda: self.fit_results,
            get_runs=lambda: self.runs,
            show_step_header=True,
        )
        self.sketch_tab.pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # EXPORT HELPERS (Used by Unified Export Studio & programmatic callers)
    # ------------------------------------------------------------------
    def build_coeffs_json_payload(self) -> dict:
        clean_fits = {}
        for k, rec in self.fit_results.items():
            clean_fits[k] = {
                "model": rec.get("model", k),
                "parameters": rec.get("parameters", {}),
                "r2": rec.get("r2"),
                "rmse": rec.get("rmse"),
                "nrmse": rec.get("nrmse"),
                "n_points": rec.get("n_points"),
            }
        return {
            "schema_version": 2,
            "conventions": {
                "FZ": "positive compression [N]",
                "SA_model": "radians",
                "SA_UI": "degrees",
                "SL": "dimensionless ratio",
            },
            "runs": [
                {"filepath": r.filepath, "test_type": r.test_type, "target_fz": r.target_fz, "tol_fz": r.tol_fz, "target_param": r.target_param}
                for r in self.runs
            ],
            "fits": clean_fits,
        }

    def export_current_tir(self):
        path = filedialog.asksaveasfilename(
            title="Export Active Model to .TIR",
            defaultextension=".tir",
            filetypes=[("ADAMS / CarSim TIR", "*.tir"), ("All Files", "*.*")],
            initialfile="PUTM_MF51_Tire.tir",
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model.to_tir_content(os.path.splitext(os.path.basename(path))[0]))
        self.set_status(f"Exported .TIR: {path}")

    def open_fy_excel_export(self):
        open_excel_export(self.root, self.runs, raw_cache=self.raw_cache, colors=self.c)

    def open_condition_export(self):
        ConditionExportDialog(self.root, self.runs, raw_cache=self.raw_cache, colors=self.c)

    def export_data(self):
        if (self.df_cornering is None or self.df_cornering.empty) and (self.df_drive is None or self.df_drive.empty):
            messagebox.showwarning("No Data", "Process telemetry runs in Step 1 or load FSAE Demo first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")], initialfile="PUTM_TTC_Filtered.csv")
        if not path:
            return
        chunks = []
        if self.df_cornering is not None and not self.df_cornering.empty:
            c_df = self.df_cornering.copy()
            c_df.insert(0, "TestType", "Cornering")
            chunks.append(c_df)
        if self.df_drive is not None and not self.df_drive.empty:
            d_df = self.df_drive.copy()
            d_df.insert(0, "TestType", "Drive/Brake")
            chunks.append(d_df)
        pd.concat(chunks, ignore_index=True).to_csv(path, index=False)
        self.set_status(f"Saved filtered TTC telemetry: {path}")

    def export_coeffs(self):
        if not self.fit_results:
            messagebox.showwarning("No Fits", "No fitted parameters stored in Step 2.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")], initialfile="PUTM_Fit_Coefficients_v2.json")
        if not path:
            return
        payload = self.build_coeffs_json_payload()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        self.set_status(f"Exported JSON v2 coefficients: {path}")


def main():
    root = tk.Tk()
    PUTMTireApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
