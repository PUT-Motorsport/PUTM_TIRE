"""mf51_tab.py — Step 3: Magic Formula 5.1 (MF-Tyre / PAC2002) Editor & Scaling.
=================================================================================
PUT MOTORSPORT — Formula Student Tire Dynamics Suite.
"""

from __future__ import annotations

import csv
import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Dict, List, Optional

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from auto_mf51_extractor import AutoMF51Extractor
from mf51_model import DEFAULT_MF51_PARAMS, MF51Model
from tir_parser import load_tir_to_model
from tire_scaler import scale_18x6_to_16x75_r20
from ui_scroll import ScrollableFrame
from ui_theme import (
    CHART_PALETTE,
    CollapsibleCard,
    FONT_MONO,
    FONT_SMALL,
    FONT_SMALL_BOLD,
    KpiTile,
    ModernCard,
    PALETTE,
    SegmentedToggle,
    StepHeader,
    style_axes_2d,
    style_legend,
)

CHANNELS = [
    "Fy (Lateral Force)",
    "Fx (Longitudinal Force)",
    "Mz (Aligning Torque)",
    "Combined (Combined Slip)",
    "Friction Coefficient mu vs Fz",
]
SWEEP_MODES = ["Vertical Load Fz Sweep", "Inclination Angle (Camber / IA) Sweep"]

PARAM_SHORT_NAMES = {
    "FNOMIN": "Fz0 (Nominal) [N]",
    "UNLOADED_RADIUS": "R0 (Radius) [m]",
    "WIDTH": "Width [m]",
    "LMUX": "LMUX (mu_x Scale)",
    "LMUY": "LMUY (mu_y Scale)",
    # Fy
    "PCY1": "Cy (Shape)",
    "PDY1": "PDY1 (Base mu_y0)",
    "PDY2": "PDY2 (dmu_y/dfz)",
    "PDY3": "PDY3 (Camber Drop)",
    "PEY1": "Ey (Curvature)",
    "PEY2": "PEY2 (dEy/dfz)",
    "PEY3": "PEY3 (dEy/gamma)",
    "PEY4": "PEY4 (Ey Asym.)",
    "PKY1": "PKY1 (Ky Stiffness)",
    "PKY2": "PKY2 (Fz Saturation)",
    "PKY3": "PKY3 (Camber Stiff.)",
    "PHY1": "PHY1 (Shy Shift)",
    "PHY2": "PHY2 (dShy/dfz)",
    "PHY3": "PHY3 (Camber Shy)",
    "PVY1": "PVY1 (Svy Shift)",
    "PVY2": "PVY2 (dSvy/dfz)",
    "PVY3": "PVY3 (Camber Thrust)",
    "PVY4": "PVY4 (dPVY3/dfz)",
    # Fx
    "PCX1": "Cx (Shape)",
    "PDX1": "PDX1 (Base mu_x0)",
    "PDX2": "PDX2 (dmu_x/dfz)",
    "PDX3": "PDX3 (Camber Drop)",
    "PEX1": "Ex (Curvature)",
    "PEX2": "PEX2 (dEx/dfz)",
    "PEX3": "PEX3 (dEx/dfz^2)",
    "PEX4": "PEX4 (Ex Asym.)",
    "PKX1": "PKX1 (Kx Stiffness)",
    "PKX2": "PKX2 (dKx/dfz)",
    "PKX3": "PKX3 (Kx Exp.)",
    "PHX1": "PHX1 (Shx Shift)",
    "PHX2": "PHX2 (dShx/dfz)",
    "PVX1": "PVX1 (Svx Shift)",
    "PVX2": "PVX2 (dSvx/dfz)",
    # Mz
    "QCZ1": "QCZ1 (Trail Shape)",
    "QDZ1": "QDZ1 (Trail Arm t0)",
    "QDZ2": "QDZ2 (dDt/dfz)",
    "QDZ3": "QDZ3 (Camber Trail)",
    "QBZ1": "QBZ1 (Bt Stiffness)",
    "QBZ2": "QBZ2 (dBt/dfz)",
    "QBZ5": "QBZ5 (Bt Camber)",
    "QEZ1": "QEZ1 (Et Curvature)",
    "QHZ1": "QHZ1 (Sht Shift)",
    "QDZ6": "QDZ6 (Mzr Residual)",
    "QDZ8": "QDZ8 (Mzr Camber)",
    "QBZ9": "QBZ9 (Br Stiffness)",
    # Combined
    "RBY1": "RBY1 (Fy Reduction)",
    "RBY2": "RBY2 (kappa Coupling)",
    "RCY1": "RCY1 (Gyk Shape)",
    "RBX1": "RBX1 (Fx Reduction)",
    "RBX2": "RBX2 (SA Coupling)",
    "RCX1": "RCX1 (Gxa Shape)",
}

PARAM_CATEGORIES = {
    "Primary Performance (Grip & Load Sensitivity)": [
        "FNOMIN", "LMUY", "LMUX", "PDY1", "PDY2", "PKY1", "PKY2",
        "PDX1", "PDX2", "PKX1", "PKX2",
        "QCZ1", "QDZ1", "QBZ1",
    ],
    "Camber Effects (Inclination Angle IA)": [
        "PVY3", "PDY3", "PKY3", "PHY3",
        "PDX3",
        "QDZ3", "QBZ5", "QDZ8",
    ],
    "Curve Shape, Offsets & Combined Slip (Advanced)": [
        "UNLOADED_RADIUS", "WIDTH", "PCY1", "PEY1", "PEY2", "PEY3", "PEY4",
        "PHY1", "PHY2", "PVY1", "PVY2", "PVY4",
        "PCX1", "PEX1", "PEX2", "PEX3", "PEX4", "PKX3", "PHX1", "PHX2", "PVX1", "PVX2",
        "QDZ2", "QBZ2", "QEZ1", "QHZ1", "QDZ6", "QBZ9",
        "RBY1", "RBY2", "RCY1", "RBX1", "RBX2", "RCX1",
    ],
}

_known_keys = set().union(*PARAM_CATEGORIES.values())
for _k in DEFAULT_MF51_PARAMS:
    if _k not in _known_keys:
        PARAM_CATEGORIES["Curve Shape, Offsets & Combined Slip (Advanced)"].append(_k)

CHANNEL_PARAM_KEYS = {
    "Fy (Lateral Force)": [
        "FNOMIN", "UNLOADED_RADIUS", "LMUY",
        "PCY1", "PDY1", "PDY2", "PDY3",
        "PEY1", "PEY2", "PEY3", "PEY4",
        "PKY1", "PKY2", "PKY3",
        "PHY1", "PHY3", "PVY1", "PVY3",
    ],
    "Fx (Longitudinal Force)": [
        "FNOMIN", "UNLOADED_RADIUS", "LMUX",
        "PCX1", "PDX1", "PDX2", "PDX3",
        "PEX1", "PEX2", "PKX1", "PKX2",
        "PHX1", "PVX1",
    ],
    "Mz (Aligning Torque)": [
        "FNOMIN", "UNLOADED_RADIUS", "LMUY",
        "QCZ1", "QDZ1", "QDZ2", "QDZ3",
        "QBZ1", "QBZ2", "QBZ5",
        "QEZ1", "QHZ1",
        "QDZ6", "QDZ8", "QBZ9",
        "PCY1", "PDY1", "PKY1",
    ],
    "Combined (Combined Slip)": [
        "FNOMIN", "UNLOADED_RADIUS", "LMUX", "LMUY",
        "RBY1", "RBY2", "RCY1",
        "RBX1", "RBX2", "RCX1",
        "PCY1", "PDY1", "PCX1", "PDX1",
    ],
    "Friction Coefficient mu vs Fz": [
        "FNOMIN", "LMUX", "LMUY", "PDY1", "PDY2", "PDY3", "PDX1", "PDX2", "PDX3",
    ],
}


class MF51Tab(tk.Frame):
    """Step 3 — PAC2002 / MF 5.1 Tire Parameter Editor & Dimensional Scaler."""

    def __init__(
        self,
        parent,
        *,
        colors: Optional[dict] = None,
        fonts: Optional[dict] = None,
        set_status=None,
        get_fit_results=None,
        get_runs=None,
        get_df_cornering=None,
        get_df_drive=None,
        get_model: Optional[Callable[[], MF51Model]] = None,
        set_model: Optional[Callable[[MF51Model], None]] = None,
        on_model_updated: Optional[Callable[[], None]] = None,
        on_next_step: Optional[Callable[[], None]] = None,
        show_step_header: bool = True,
    ):
        c = colors or PALETTE
        super().__init__(parent, bg=c["bg"])
        self.colors = c
        fonts = fonts or {}
        self.font_mono = fonts.get("mono", FONT_MONO)
        self.font_small = fonts.get("small", FONT_SMALL)
        self._set_status = set_status or (lambda text: None)
        self._get_fit_results = get_fit_results or (lambda: {})
        self._get_runs = get_runs or (lambda: [])
        self._get_df_cornering = get_df_cornering or (lambda: None)
        self._get_df_drive = get_df_drive or (lambda: None)
        self._get_model = get_model
        self._set_model = set_model
        self._on_model_updated = on_model_updated or (lambda: None)
        self._on_next_step = on_next_step
        self._show_step_header = show_step_header
        self._last_diagnostics_text = ""

        self._own_model = MF51Model()

        self.var_channel = tk.StringVar(value=CHANNELS[0])
        self.var_sweep_mode = tk.StringVar(value=SWEEP_MODES[0])
        self.var_sweep_levels = tk.StringVar(value="400, 800, 1200, 1600")
        self.var_const_camber = tk.StringVar(value="0.0")
        self.var_range = tk.StringVar(value="15.0")
        self.var_combined_kappa = tk.StringVar(value="0.0, 0.05, 0.12, 0.25")
        self.var_surface = tk.StringVar(value="ttc")
        self._syncing_surface = False

        self.param_vars: Dict[str, tk.StringVar] = {}
        for key, val in DEFAULT_MF51_PARAMS.items():
            self.param_vars[key] = tk.StringVar(value=f"{val:g}")

        self._build_ui()
        self.sync_from_model(redraw=False)
        self._on_channel_change()
        self.draw(notify=False)

    @property
    def model(self) -> MF51Model:
        if self._get_model is not None:
            return self._get_model()
        return self._own_model

    @model.setter
    def model(self, value: MF51Model) -> None:
        if self._set_model is not None:
            self._set_model(value)
        else:
            self._own_model = value

    # --------------------------------------------------------------------------
    # UI CONSTRUCTION
    # --------------------------------------------------------------------------
    def _build_ui(self):
        c = self.colors

        if self._show_step_header:
            hdr = StepHeader(
                self,
                "STEP 3 / 5",
                "MF 5.1 Tire Model (PAC2002) & Scaling",
                "Tune Magic Formula coefficients, toggle Asphalt/Flat-Trac friction scaling, and convert 18x6 -> 16x7.5 R20 longitudinal data.",
                badge_kind="accent",
                next_text="Next: 3D Simulator (Step 4) ->" if self._on_next_step else None,
                next_command=self._on_next_step,
                colors=c,
            )
            hdr.pack(fill="x", padx=10, pady=(8, 4))

        main = tk.Frame(self, bg=c["bg"])
        main.pack(fill="both", expand=True)

        left_scroll = ScrollableFrame(main, bg=c["bg"], width=445)
        left_scroll.pack(side="left", fill="y", padx=(10, 5), pady=(4, 8))
        left = left_scroll.inner

        # 1. Surface & Geometric Scaling (Imports & Exports unified in Header Import/Export)
        scale_card = CollapsibleCard(
            left, "1. Surface Scale & Geometry (18x6 -> 16x7.5)", is_open=True, badge="FSAE Scale", badge_color="accent", colors=c, pad_x=10, pad_y=8
        )
        scale_card.pack(fill="x", pady=(0, 8))

        tk.Label(
            scale_card.body, text="Track Surface Multiplier (LMUX / LMUY):",
            fg=c["text_secondary"], bg=c["panel"], font=FONT_SMALL_BOLD
        ).pack(anchor="w", pady=(0, 4))

        self.surface_toggle = SegmentedToggle(
            scale_card.body,
            self.var_surface,
            [("asphalt", "FSAE Asphalt (0.65)"), ("ttc", "Flat-Trac TTC (1.00)")],
            command=self._on_surface_change,
            accent_key="accent",
            colors=c,
        )
        self.surface_toggle.pack(fill="x", pady=(0, 8))

        info_box = tk.Frame(
            scale_card.body, bg=c["input"], padx=8, pady=6,
            highlightbackground=c["border"], highlightthickness=1
        )
        info_box.pack(fill="x", pady=(0, 6))
        tk.Label(
            info_box,
            text="Fx Dimensional Scaling (18x6-10 -> 16x7.5-10 R20):\n"
                 "• R0=0.205 m, W=0.180 m · PDX2=0.80·PDY2 · PKX1=28.0",
            fg=c["text_muted"], bg=c["input"], font=FONT_SMALL, justify="left", anchor="w"
        ).pack(anchor="w")

        s_btns = tk.Frame(scale_card.body, bg=c["panel"])
        s_btns.pack(fill="x", pady=(0, 4))
        ttk.Button(
            s_btns, text="Scale Fx: 18x6 -> 16x7.5 R20",
            command=self.run_tire_scaling, style="Accent.TButton"
        ).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(
            s_btns, text="Sync Fit (Step 2)",
            command=self.load_from_fit_results, style="Pill.TButton"
        ).pack(side="left")

        self.lbl_scale_status = tk.Label(
            scale_card.body, text="● Scaling: Base Model (unscaled)",
            fg=c["text_muted"], bg=c["panel"], font=FONT_SMALL
        )
        self.lbl_scale_status.pack(anchor="w")

        # 2. Curve simulation setup
        cfg_card = ModernCard(left, "2. Sweep Simulation Configuration", colors=c, pad_x=10, pad_y=8)
        cfg_card.pack(fill="x", pady=(0, 8))
        cfg = cfg_card.body
        cfg.columnconfigure(1, weight=1)

        tk.Label(cfg, text="Channel:", fg=c["text_secondary"], bg=c["panel"], font=FONT_SMALL_BOLD).grid(
            row=0, column=0, sticky="w", pady=3, padx=(0, 6)
        )
        cb_channel = ttk.Combobox(cfg, textvariable=self.var_channel, values=CHANNELS, state="readonly", width=26)
        cb_channel.grid(row=0, column=1, columnspan=3, sticky="ew", pady=3)
        cb_channel.bind("<<ComboboxSelected>>", lambda _e: self._on_channel_change())

        tk.Label(cfg, text="Sweep:", fg=c["text_secondary"], bg=c["panel"], font=FONT_SMALL_BOLD).grid(
            row=1, column=0, sticky="w", pady=3, padx=(0, 6)
        )
        self.cb_sweep = ttk.Combobox(cfg, textvariable=self.var_sweep_mode, values=SWEEP_MODES, state="readonly", width=18)
        self.cb_sweep.grid(row=1, column=1, sticky="ew", pady=3)
        self.cb_sweep.bind("<<ComboboxSelected>>", lambda _e: self._on_sweep_mode_change())

        tk.Label(cfg, text="Range ±:", fg=c["text_secondary"], bg=c["panel"], font=FONT_SMALL_BOLD).grid(
            row=1, column=2, sticky="w", padx=(8, 4), pady=3
        )
        ent_range = ttk.Entry(cfg, textvariable=self.var_range, width=7)
        ent_range.grid(row=1, column=3, sticky="e", pady=3)
        ent_range.bind("<Return>", lambda _e: self.draw())

        self.lbl_sweep_vals = tk.Label(cfg, text="Fz Levels [N]:", fg=c["text_secondary"], bg=c["panel"], font=FONT_SMALL_BOLD)
        self.lbl_sweep_vals.grid(row=2, column=0, sticky="w", pady=3, padx=(0, 6))
        self.ent_sweep_vals = ttk.Entry(cfg, textvariable=self.var_sweep_levels, width=18)
        self.ent_sweep_vals.grid(row=2, column=1, sticky="ew", pady=3)
        self.ent_sweep_vals.bind("<Return>", lambda _e: self.draw())

        self.lbl_const_param = tk.Label(cfg, text="Camber IA [°]:", fg=c["text_secondary"], bg=c["panel"], font=FONT_SMALL_BOLD)
        self.lbl_const_param.grid(row=2, column=2, sticky="w", padx=(8, 4), pady=3)
        self.ent_const_param = ttk.Entry(cfg, textvariable=self.var_const_camber, width=7)
        self.ent_const_param.grid(row=2, column=3, sticky="e", pady=3)
        self.ent_const_param.bind("<Return>", lambda _e: self.draw())

        # 3. Parameter accordions
        self.params_frame = tk.Frame(left, bg=c["bg"])
        self.params_frame.pack(fill="x", pady=(0, 8))

        self.category_cards: Dict[str, CollapsibleCard] = {}
        self.category_widgets: Dict[str, Dict[str, tk.Frame]] = {}

        for idx, (cat_name, cat_keys) in enumerate(PARAM_CATEGORIES.items()):
            is_open = (idx == 0)
            cat_card = CollapsibleCard(
                self.params_frame, cat_name, is_open=is_open,
                badge="Core" if idx == 0 else "", badge_color="ok",
                colors=c, pad_x=8, pad_y=6
            )
            self.category_cards[cat_name] = cat_card
            self.category_widgets[cat_name] = {}

            for key in cat_keys:
                short_lbl = PARAM_SHORT_NAMES.get(key, key)
                cell = tk.Frame(cat_card.body, bg=c["panel"])
                lbl = tk.Label(cell, text=short_lbl, fg=c["text_secondary"], bg=c["panel"], font=FONT_SMALL, width=17, anchor="w")
                lbl.pack(side="left", padx=(2, 2))
                ent = ttk.Entry(cell, textvariable=self.param_vars[key], width=8)
                ent.pack(side="right", padx=(0, 2))
                ent.bind("<Return>", lambda _e: self.draw())
                self.category_widgets[cat_name][key] = cell

        # 4. Actions (Recompute / Reset — use Header Import/Export for .TIR/CSV/JSON)
        act_card = ModernCard(left, "3. Evaluate & Manage Active Model", colors=c, pad_x=10, pad_y=8)
        act_card.pack(fill="x", pady=(0, 6))

        ttk.Button(act_card.body, text="Recompute & Refresh Model", command=self.draw, style="Accent.TButton").pack(
            fill="x", pady=(0, 6)
        )
        sub_a = tk.Frame(act_card.body, bg=c["panel"])
        sub_a.pack(fill="x")
        ttk.Button(sub_a, text="Reset Hoosier R25B Preset", command=self.reset_to_fsae_defaults, style="Pill.TButton").pack(
            side="left", fill="x", expand=True, padx=(0, 4)
        )
        ttk.Button(sub_a, text="Model Report & Logs", command=self.open_diagnostics_dialog, style="Pill.TButton").pack(
            side="left", fill="x", expand=True
        )

        self.txt_summary = None

        # ==================================================================
        # RIGHT PANEL: Plot + KPI Tiles
        # ==================================================================
        right = tk.Frame(main, bg=c["bg"])
        right.pack(side="left", fill="both", expand=True, padx=(5, 10), pady=(4, 8))

        plot_card = ModernCard(
            right, "Magic Formula 5.1 (PAC2002) Characteristic Curves",
            badge="PAC2002", badge_color="cyan", colors=c, pad_x=8, pad_y=8
        )
        plot_card.pack(fill="both", expand=True)

        tb = tk.Frame(plot_card.body, bg=c["panel"])
        tb.pack(fill="x", pady=(0, 6))
        self.lbl_status_model = tk.Label(
            tb, text="● MF 5.1 model ready for simulation and .TIR export",
            fg=c["text_muted"], bg=c["panel"], font=FONT_SMALL_BOLD
        )
        self.lbl_status_model.pack(side="left")

        self.fig = Figure(figsize=(7, 4.6), dpi=100, facecolor=c["panel"])
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_card.body)
        self.canvas.get_tk_widget().configure(bg=c["panel"], highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        kpi_frame = tk.Frame(right, bg=c["bg"])
        kpi_frame.pack(fill="x", pady=(8, 0))

        self.kpi_tiles: Dict[str, KpiTile] = {}
        for i, (kpi_id, title, color_key) in enumerate([
            ("grip", "PEAK FRICTION mu", "accent"),
            ("stiff", "CORNERING STIFFNESS", "cyan"),
            ("camber", "CAMBER THRUST", "ok"),
            ("peak_force", "PEAK FORCE @ Fz0", "warn"),
        ]):
            tile = KpiTile(kpi_frame, title, "--", accent=c[color_key], colors=c)
            tile.pack(side="left", fill="x", expand=True, padx=(0 if i == 0 else 4, 0))
            self.kpi_tiles[kpi_id] = tile

    # --------------------------------------------------------------------------
    # STATE SYNC
    # --------------------------------------------------------------------------
    def sync_from_model(self, redraw: bool = True):
        """Refresh UI fields and surface toggle from the active MF51Model."""
        m = self.model
        for key in DEFAULT_MF51_PARAMS:
            if key in self.param_vars:
                self.param_vars[key].set(f"{m.get(key):g}")
        lmuy = m.get("LMUY", 1.0)
        self._syncing_surface = True
        self.var_surface.set("asphalt" if lmuy < 0.85 else "ttc")
        self._syncing_surface = False
        if redraw:
            self.draw(notify=False)

    def _on_surface_change(self):
        if self._syncing_surface:
            return
        scale = 0.65 if self.var_surface.get() == "asphalt" else 1.00
        self.model.set("LMUX", scale)
        self.model.set("LMUY", scale)
        self.param_vars["LMUX"].set(f"{scale:g}")
        self.param_vars["LMUY"].set(f"{scale:g}")
        self.draw(notify=True)
        self._set_status(f"Surface multiplier set (LMUX/LMUY = {scale:.2f}).")

    def run_tire_scaling(self):
        """Scale longitudinal Fx parameters from Hoosier 18x6-10 R20 to 16x7.5-10 R20."""
        self._sync_model_from_ui()
        report = scale_18x6_to_16x75_r20(self.model)
        self.sync_from_model(redraw=False)
        self.lbl_scale_status.configure(
            text=f"✓ Scaled to 16x7.5 R20 (PDX1={report.pdx1_after:.3f}, PKX1={report.pkx1_after:.1f})",
            fg=self.colors["ok"],
        )
        self._set_status(
            f"Scaled Fx model from 18x6 to 16x7.5-10 R20 (PDX1={report.pdx1_after:.3f}, PKX1={report.pkx1_after:.1f})."
        )
        self.draw(notify=True)

        log_lines = "\n".join(f"• {line}" for line in report.log)
        messagebox.showinfo(
            "Tire Scaling 18x6 -> 16x7.5-10 R20",
            f"Applied physical tire scaling to MF 5.1 model:\n\n"
            f"Source tire: {report.source_tire}\n"
            f"Target tire: {report.target_tire}\n"
            f"Compound: {report.compound}\n\n"
            f"Changes applied:\n{log_lines}\n\n"
            f"Parameter Comparison:\n"
            f"• Peak Fx grip PDX1: {report.pdx1_before:.3f} -> {report.pdx1_after:.3f}\n"
            f"• Load sensitivity PDX2: {report.pdx2_before:.4f} -> {report.pdx2_after:.4f}\n"
            f"• Slip stiffness PKX1: {report.pkx1_before:.1f} -> {report.pkx1_after:.1f}\n"
            f"• Stiffness slope PKX2: {report.pkx2_before:.2f} -> {report.pkx2_after:.2f}",
        )

    def load_tir_file(self, path: Optional[str] = None):
        """Load an existing .tir file into the active MF 5.1 model."""
        if not path:
            path = filedialog.askopenfilename(
                title="Load Pacejka Tire Property File (.TIR)",
                filetypes=[("Pacejka Tire Property File (*.tir)", "*.tir"), ("All Files", "*.*")],
            )
        if not path:
            return
        try:
            loaded_model, meta = load_tir_to_model(path)
            for key, val in loaded_model.parameters.items():
                self.model.set(key, val)
            self.sync_from_model(redraw=False)
            self.draw(notify=True)
            name = meta.get("TIRE_NAME") or os.path.basename(path)
            self._set_status(f"Loaded .TIR model: {name}")
            messagebox.showinfo(".TIR Loaded", f"Successfully loaded tire parameters from:\n{os.path.basename(path)}")
        except Exception as exc:
            messagebox.showerror(".TIR Read Error", f"Failed to parse .tir file:\n{exc}")

    def save_plot(self, path: Optional[str] = None, dpi: int = 300):
        ch = self.var_channel.get().replace(" ", "_").replace("(", "").replace(")", "")
        if not path:
            path = filedialog.asksaveasfilename(
                title="Save MF 5.1 Simulation Plot",
                defaultextension=".png",
                initialfile=f"MF51_{ch}.png",
                filetypes=[("PNG Image (*.png)", "*.png"), ("PDF Document (*.pdf)", "*.pdf"), ("SVG (*.svg)", "*.svg")],
            )
        if not path:
            return
        try:
            self.fig.savefig(path, dpi=dpi, facecolor=self.fig.get_facecolor(), bbox_inches="tight")
            self._set_status(f"Saved MF 5.1 plot: {path}")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    # --------------------------------------------------------------------------
    # CHANNEL & SWEEP CHANGES
    # --------------------------------------------------------------------------
    def _on_channel_change(self):
        channel = self.var_channel.get()
        active_keys = set(CHANNEL_PARAM_KEYS.get(channel, list(DEFAULT_MF51_PARAMS.keys())))

        for cat_name, cat_card in self.category_cards.items():
            for slave in cat_card.body.grid_slaves():
                slave.grid_forget()

            matching_keys = [k for k in PARAM_CATEGORIES[cat_name] if k in active_keys]
            if not matching_keys:
                cat_card.pack_forget()
                continue

            cat_card.pack(fill="x", pady=(0, 6))
            for i, key in enumerate(matching_keys):
                r = i // 2
                col = i % 2
                cell = self.category_widgets[cat_name][key]
                cell.grid(row=r, column=col, sticky="ew", padx=3, pady=2)
            cat_card.body.columnconfigure(0, weight=1)
            cat_card.body.columnconfigure(1, weight=1)

        if "Fx" in channel:
            self.var_range.set("0.30")
        else:
            self.var_range.set("15.0")

        self.draw(notify=False)

    def _on_sweep_mode_change(self):
        mode = self.var_sweep_mode.get()
        if mode == SWEEP_MODES[0]:
            self.lbl_sweep_vals.configure(text="Fz Levels [N]:")
            self.var_sweep_levels.set("400, 800, 1200, 1600")
            self.lbl_const_param.configure(text="Camber IA [°]:")
            self.var_const_camber.set("0.0")
        else:
            self.lbl_sweep_vals.configure(text="Camber IA [°]:")
            self.var_sweep_levels.set("-4.0, -2.0, 0.0, 2.0, 4.0")
            self.lbl_const_param.configure(text="Fixed Fz [N]:")
            self.var_const_camber.set("1000.0")
        self.draw(notify=False)

    def _sync_model_from_ui(self):
        for key in DEFAULT_MF51_PARAMS:
            text = self.param_vars[key].get().strip().replace(",", ".")
            try:
                self.model.set(key, float(text))
            except ValueError:
                pass
        lmuy = self.model.get("LMUY", 1.0)
        self._syncing_surface = True
        self.var_surface.set("asphalt" if lmuy < 0.85 else "ttc")
        self._syncing_surface = False

    def _set_info(self, text: str):
        self._last_diagnostics_text = text

    def open_diagnostics_dialog(self):
        c = self.colors
        dlg = tk.Toplevel(self)
        dlg.title("MF 5.1 Model Calibration & Telemetry Report")
        dlg.geometry("740x520")
        dlg.configure(bg=c["bg"])
        dlg.transient(self.winfo_toplevel())

        hdr = tk.Frame(dlg, bg=c["bg_header"], padx=16, pady=12)
        hdr.pack(fill="x")
        tk.Label(hdr, text="MF 5.1 MODEL CALIBRATION & DIAGNOSTICS REPORT", fg=c["accent"], bg=c["bg_header"], font=("Segoe UI", 11, "bold")).pack(anchor="w")
        tk.Label(hdr, text="Detailed telemetry logs from TTC multi-load extraction and Pacejka curve evaluation.", fg=c["text_muted"], bg=c["bg_header"], font=self.font_small).pack(anchor="w")
        tk.Frame(dlg, bg=c["border"], height=1).pack(fill="x")

        body = tk.Frame(dlg, bg=c["bg"], padx=14, pady=12)
        body.pack(fill="both", expand=True)

        txt = tk.Text(
            body, font=self.font_mono, bg=c["input"], fg=c["text"],
            relief="flat", padx=12, pady=10, highlightbackground=c["border"], highlightthickness=1
        )
        sbar = ttk.Scrollbar(body, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=sbar.set)
        sbar.pack(side="right", fill="y")
        txt.pack(side="left", fill="both", expand=True)

        diag_text = getattr(self, "_last_diagnostics_text", "") or (
            "No extraction diagnostics recorded yet. Run Autopilot MF 5.1 from Step 2 or Unified Import."
        )
        txt.insert("1.0", diag_text)
        txt.configure(state="disabled")

        btns = tk.Frame(dlg, bg=c["bg_header"], padx=14, pady=10)
        btns.pack(fill="x")

        def copy_text():
            dlg.clipboard_clear()
            dlg.clipboard_append(diag_text)
            messagebox.showinfo("Copied", "Report copied to clipboard.", parent=dlg)

        ttk.Button(btns, text="Copy to Clipboard", command=copy_text, style="Cyan.TButton").pack(side="left")
        ttk.Button(btns, text="Close", command=dlg.destroy, style="TButton").pack(side="right")

    def _update_kpi_badges(self):
        if not hasattr(self, "kpi_tiles"):
            return
        channel = self.var_channel.get()
        fz0 = self.model.get("FNOMIN", 1000.0)
        lmuy = self.model.get("LMUY", 1.0)
        lmux = self.model.get("LMUX", 1.0)

        if "Fy" in channel:
            pdy1 = self.model.get("PDY1", 1.5) * lmuy
            pky1 = self.model.get("PKY1", 38.0)
            pvy3 = self.model.get("PVY3", 0.08)
            camber_thrust = pvy3 * fz0 * (np.pi / 180.0)
            cf_deg = pky1 * fz0 * (np.pi / 180.0)
            peak_fy = pdy1 * fz0

            self.kpi_tiles["grip"].lbl_title.configure(text="PEAK FRICTION mu_y")
            self.kpi_tiles["grip"].set_value(f"{pdy1:.3f}")
            self.kpi_tiles["stiff"].lbl_title.configure(text="STIFFNESS CFa")
            self.kpi_tiles["stiff"].set_value(f"{cf_deg:.0f} N/°")
            self.kpi_tiles["camber"].lbl_title.configure(text="CAMBER THRUST")
            self.kpi_tiles["camber"].set_value(f"{camber_thrust:.1f} N/°")
            self.kpi_tiles["peak_force"].lbl_title.configure(text="PEAK Fy @ Fz0")
            self.kpi_tiles["peak_force"].set_value(f"{peak_fy:.0f} N")

        elif "Fx" in channel:
            pdx1 = self.model.get("PDX1", 1.6) * lmux
            pkx1 = self.model.get("PKX1", 30.0)
            pdx2 = self.model.get("PDX2", -0.15)
            peak_fx = pdx1 * fz0
            kx = pkx1 * fz0

            self.kpi_tiles["grip"].lbl_title.configure(text="PEAK FRICTION mu_x")
            self.kpi_tiles["grip"].set_value(f"{pdx1:.3f}")
            self.kpi_tiles["stiff"].lbl_title.configure(text="SLIP STIFFNESS Kx")
            self.kpi_tiles["stiff"].set_value(f"{kx/1000:.1f} kN/slip")
            self.kpi_tiles["camber"].lbl_title.configure(text="LOAD SENS. PDX2")
            self.kpi_tiles["camber"].set_value(f"{pdx2:+.4f}")
            self.kpi_tiles["peak_force"].lbl_title.configure(text="PEAK Fx @ Fz0")
            self.kpi_tiles["peak_force"].set_value(f"{peak_fx:.0f} N")

        elif "Mz" in channel:
            r0 = self.model.get("UNLOADED_RADIUS", 0.203)
            qdz1 = self.model.get("QDZ1", 0.15)
            trail_mm = qdz1 * r0 * 1000.0
            pdy1 = self.model.get("PDY1", 1.5) * lmuy
            qcz1 = self.model.get("QCZ1", 1.2)
            qdz6 = self.model.get("QDZ6", 0.005)
            mzr = qdz6 * fz0 * r0

            self.kpi_tiles["grip"].lbl_title.configure(text="PNEUMATIC TRAIL t0")
            self.kpi_tiles["grip"].set_value(f"{trail_mm:.1f} mm")
            self.kpi_tiles["stiff"].lbl_title.configure(text="TRAIL SHAPE QCZ1")
            self.kpi_tiles["stiff"].set_value(f"{qcz1:.2f}")
            self.kpi_tiles["camber"].lbl_title.configure(text="RESIDUAL TORQUE Mzr")
            self.kpi_tiles["camber"].set_value(f"{mzr:.1f} Nm")
            self.kpi_tiles["peak_force"].lbl_title.configure(text="BASE FRICTION mu_y")
            self.kpi_tiles["peak_force"].set_value(f"{pdy1:.3f}")

        else:
            pdy1 = self.model.get("PDY1", 1.5) * lmuy
            pdx1 = self.model.get("PDX1", 1.6) * lmux
            pdy2 = self.model.get("PDY2", -0.15)
            pdx2 = self.model.get("PDX2", -0.18)
            self.kpi_tiles["grip"].lbl_title.configure(text="EFF. mu_y0")
            self.kpi_tiles["grip"].set_value(f"{pdy1:.3f}")
            self.kpi_tiles["stiff"].lbl_title.configure(text="EFF. mu_x0")
            self.kpi_tiles["stiff"].set_value(f"{pdx1:.3f}")
            self.kpi_tiles["camber"].lbl_title.configure(text="mu_y SLOPE (PDY2)")
            self.kpi_tiles["camber"].set_value(f"{pdy2:+.3f}")
            self.kpi_tiles["peak_force"].lbl_title.configure(text="mu_x SLOPE (PDX2)")
            self.kpi_tiles["peak_force"].set_value(f"{pdx2:+.3f}")

        if hasattr(self, "lbl_status_model"):
            surf_txt = "Asphalt 0.65" if lmuy < 0.85 else "Flat-Trac 1.00"
            self.lbl_status_model.configure(
                text=f"● MF 5.1 Ready · Fz0 = {fz0:.0f} N · R0 = {self.model.get('UNLOADED_RADIUS', 0.205):.3f} m · {surf_txt}"
            )

    # --------------------------------------------------------------------------
    # PLOTTING
    # --------------------------------------------------------------------------
    def draw(self, notify: bool = True):
        self._sync_model_from_ui()
        channel = self.var_channel.get()
        sweep_mode = self.var_sweep_mode.get()

        try:
            half_range = abs(float(self.var_range.get().replace(",", ".")))
        except ValueError:
            half_range = 15.0 if "Fx" not in channel else 0.30

        try:
            levels = [float(v.strip().replace(",", ".")) for v in self.var_sweep_levels.get().split(",") if v.strip()]
            if not levels:
                levels = [400.0, 800.0, 1200.0, 1600.0]
        except ValueError:
            levels = [400.0, 800.0, 1200.0, 1600.0]

        try:
            const_val = float(self.var_const_camber.get().replace(",", "."))
        except ValueError:
            const_val = 0.0

        self.fig.clear()
        c = self.colors
        line_colors = CHART_PALETTE

        if "Mz" in channel:
            ax1 = self.fig.add_subplot(211, facecolor=c["axes_bg"])
            ax2 = self.fig.add_subplot(212, facecolor=c["axes_bg"], sharex=ax1)

            alpha_deg = np.linspace(-half_range, half_range, 601)
            alpha_rad = np.deg2rad(alpha_deg)

            summary_lines = ["--- MF 5.1 EVALUATION (ALIGNING TORQUE Mz) ---"]
            for i, lvl in enumerate(levels):
                color = line_colors[i % len(line_colors)]
                if sweep_mode == SWEEP_MODES[0]:
                    fz, gamma_deg = lvl, const_val
                    lbl = f"Fz={fz:g} N (IA={gamma_deg:g}°)"
                else:
                    gamma_deg, fz = lvl, const_val
                    lbl = f"IA={gamma_deg:g}° (Fz={fz:g} N)"

                gamma_rad = np.deg2rad(gamma_deg)
                mz, trail, mzr, meta = self.model.evaluate_mz(alpha_rad, fz, gamma_rad)

                ax1.plot(alpha_deg, mz, color=color, linewidth=2.2, label=lbl)
                ax2.plot(alpha_deg, trail * 1000.0, color=color, linewidth=2.0, label=f"t(SA) {lbl}")
                ax2.plot(alpha_deg, mzr, color=color, linestyle="--", linewidth=1.3, alpha=0.7)

                summary_lines.append(
                    f"{lbl}: max |Mz| = {meta['max_mz_Nm']:.1f} Nm | peak trail = {meta['Dt_peak_trail_mm']:.1f} mm"
                )

            style_axes_2d(ax1, self.fig, title="Aligning Torque Mz(SA)", xlabel="Slip Angle SA [deg]", ylabel="Mz [Nm]", c=c)
            style_axes_2d(ax2, self.fig, title="Pneumatic Trail t [mm] (solid) & Residual Torque Mzr [Nm] (dashed)", xlabel="Slip Angle SA [deg]", ylabel="Trail [mm] / Mzr [Nm]", c=c)
            style_legend(ax1, c=c)
            style_legend(ax2, c=c)

            self.fig.tight_layout(pad=1.4)
            self.canvas.draw()
            self._update_kpi_badges()
            self._set_info("\n".join(summary_lines))
            if notify:
                self._on_model_updated()
            return

        if "Friction Coefficient" in channel:
            ax1 = self.fig.add_subplot(211, facecolor=c["axes_bg"])
            ax2 = self.fig.add_subplot(212, facecolor=c["axes_bg"], sharex=ax1)

            fz_arr = np.linspace(100.0, 2500.0, 401)
            fz0 = self.model.get("FNOMIN", 1000.0)
            dfz = (fz_arr - fz0) / fz0
            gamma_rad = np.deg2rad(const_val)
            lmuy = self.model.get("LMUY", 1.0)
            lmux = self.model.get("LMUX", 1.0)

            mu_y = (self.model.get("PDY1", 1.65) + self.model.get("PDY2", -0.15) * dfz) * (1.0 - self.model.get("PDY3", 0.0) * (gamma_rad**2)) * lmuy
            mu_x = (self.model.get("PDX1", 1.75) + self.model.get("PDX2", -0.15) * dfz) * (1.0 - self.model.get("PDX3", 0.0) * (gamma_rad**2)) * lmux

            Dy = mu_y * fz_arr
            Dx = mu_x * fz_arr

            ax1.plot(fz_arr, mu_y, color=c["accent"], linewidth=2.4, label=f"Lateral mu_y (PDY1={self.model.get('PDY1'):.2f}, LMUY={lmuy:.2f})")
            ax1.plot(fz_arr, mu_x, color=c["cyan"], linewidth=2.4, label=f"Long. mu_x (PDX1={self.model.get('PDX1'):.2f}, LMUX={lmux:.2f})")
            ax1.axvline(fz0, color=c["warn"], linestyle=":", alpha=0.85, label=f"Fz0 = {fz0:.0f} N")
            style_axes_2d(ax1, self.fig, title=f"Friction Load Sensitivity mu(Fz) — Camber={const_val:g}°", xlabel="Vertical Load Fz [N]", ylabel="Friction Coefficient mu [-]", zero_lines=False, c=c)
            style_legend(ax1, c=c)

            ax2.plot(fz_arr, Dy, color=c["accent"], linewidth=2.2, label="Peak Lateral Force Dy [N]")
            ax2.plot(fz_arr, Dx, color=c["cyan"], linewidth=2.2, label="Peak Long. Force Dx [N]")
            ax2.axvline(fz0, color=c["warn"], linestyle=":", alpha=0.85)
            style_axes_2d(ax2, self.fig, title="Peak Force Capacity Dy(Fz) & Dx(Fz) = mu · Fz", xlabel="Vertical Load Fz [N]", ylabel="Peak Force D [N]", zero_lines=False, c=c)
            style_legend(ax2, c=c)

            summary_lines = [
                "--- MF 5.1 FRICTION LOAD SENSITIVITY ---",
                f"Fz0 = {fz0:.0f} N | Camber = {const_val:g} deg | LMUY={lmuy:.2f} | LMUX={lmux:.2f}",
                f"Lateral:      PDY1={self.model.get('PDY1'):.3f}, PDY2={self.model.get('PDY2'):.4f}, PDY3={self.model.get('PDY3'):.3f}",
                f"Longitudinal: PDX1={self.model.get('PDX1'):.3f}, PDX2={self.model.get('PDX2'):.4f}, PDX3={self.model.get('PDX3'):.3f}",
                "",
                f"{'Fz [N]':>8} | {'mu_y':>8} | {'Dy [N]':>8} | {'mu_x':>8} | {'Dx [N]':>8}",
                "-" * 52,
            ]
            for check_fz in [400, 800, 1000, 1400, 1800, 2200]:
                c_dfz = (check_fz - fz0) / fz0
                c_my = (self.model.get("PDY1", 1.65) + self.model.get("PDY2", -0.15) * c_dfz) * (1.0 - self.model.get("PDY3", 0.0) * (gamma_rad**2)) * lmuy
                c_mx = (self.model.get("PDX1", 1.75) + self.model.get("PDX2", -0.15) * c_dfz) * (1.0 - self.model.get("PDX3", 0.0) * (gamma_rad**2)) * lmux
                summary_lines.append(f"{check_fz:8d} | {c_my:8.3f} | {c_my*check_fz:8.0f} | {c_mx:8.3f} | {c_mx*check_fz:8.0f}")

            self.fig.tight_layout(pad=1.4)
            self.canvas.draw()
            self._update_kpi_badges()
            self._set_info("\n".join(summary_lines))
            if notify:
                self._on_model_updated()
            return

        if "Combined" in channel:
            ax1 = self.fig.add_subplot(121, facecolor=c["axes_bg"])
            ax2 = self.fig.add_subplot(122, facecolor=c["axes_bg"])

            fz = float(self.model.get("FNOMIN", 1000.0))
            gamma_rad = np.deg2rad(const_val)

            alpha_deg = np.linspace(-half_range, half_range, 501)
            alpha_rad = np.deg2rad(alpha_deg)
            kappa_levels = [0.0, 0.05, 0.10, 0.20, 0.35]

            for i, kap in enumerate(kappa_levels):
                color = line_colors[i % len(line_colors)]
                fy_comb = self.model.evaluate_combined_fy(alpha_rad, kap, fz, gamma_rad)
                ax1.plot(alpha_deg, fy_comb, color=color, linewidth=2.0, label=f"SL = {kap:.2f}")

            style_axes_2d(ax1, self.fig, title=f"Fy(SA) Under Longitudinal Slip (Fz={fz:g} N)", xlabel="Slip Angle SA [deg]", ylabel="Lateral Force Fy [N]", c=c)
            style_legend(ax1, c=c)

            slips_k = np.linspace(-0.35, 0.35, 201)
            sa_levels_deg = [0.0, 2.0, 4.0, 8.0, 12.0]
            for i, sa_d in enumerate(sa_levels_deg):
                color = line_colors[i % len(line_colors)]
                sa_r = np.deg2rad(sa_d)
                fx_comb = self.model.evaluate_combined_fx(slips_k, sa_r, fz, gamma_rad)
                fy_comb = self.model.evaluate_combined_fy(np.full_like(slips_k, sa_r), slips_k, fz, gamma_rad)
                ax2.plot(fx_comb, fy_comb, color=color, linewidth=2.0, label=f"SA = {sa_d:g}°")

            style_axes_2d(ax2, self.fig, title=f"Friction Ellipse Fx vs Fy (Fz={fz:g} N)", xlabel="Longitudinal Force Fx [N]", ylabel="Lateral Force Fy [N]", c=c)
            style_legend(ax2, c=c)

            self.fig.tight_layout(pad=1.4)
            self.canvas.draw()
            self._update_kpi_badges()
            self._set_info(
                f"--- COMBINED SLIP MF 5.1 (Fz={fz:g} N) ---\n"
                f"Weighting functions: Gyk (lateral) and Gxa (longitudinal)\n"
                f"Parameters: RBY1={self.model.get('RBY1'):.2f}, RBY2={self.model.get('RBY2'):.2f}, RCY1={self.model.get('RCY1'):.2f}\n"
                f"            RBX1={self.model.get('RBX1'):.2f}, RBX2={self.model.get('RBX2'):.2f}, RCX1={self.model.get('RCX1'):.2f}"
            )
            if notify:
                self._on_model_updated()
            return

        # Pure slip Fy or Fx
        ax = self.fig.add_subplot(111, facecolor=c["axes_bg"])
        is_fy = "Fy" in channel

        if is_fy:
            x_shown = np.linspace(-half_range, half_range, 601)
            x_rad = np.deg2rad(x_shown)
            xlabel = "Slip Angle SA [deg]"
            ylabel = "Lateral Force Fy [N]"
            title_prefix = "Lateral Force Fy(SA)"
        else:
            x_shown = np.linspace(-half_range, half_range, 601)
            x_rad = x_shown
            xlabel = "Longitudinal Slip Ratio SL [-]"
            ylabel = "Longitudinal Force Fx [N]"
            title_prefix = "Longitudinal Force Fx(SL)"

        summary_lines = [f"--- MF 5.1 EVALUATION ({'LATERAL Fy' if is_fy else 'LONGITUDINAL Fx'}) ---"]

        for i, lvl in enumerate(levels):
            color = line_colors[i % len(line_colors)]
            if sweep_mode == SWEEP_MODES[0]:
                fz, gamma_deg = lvl, const_val
                lbl = f"Fz={fz:g} N (IA={gamma_deg:g}°)"
            else:
                gamma_deg, fz = lvl, const_val
                lbl = f"IA={gamma_deg:g}° (Fz={fz:g} N)"

            gamma_rad = np.deg2rad(gamma_deg)

            if is_fy:
                force, meta = self.model.evaluate_fy(x_rad, fz, gamma_rad)
                stiff = meta["Ky_alpha"] * (np.pi / 180.0)
                peak_f = meta["Dy"]
                mu = meta["mu_y"] * self.model.get("LMUY", 1.0)
                summary_lines.append(
                    f"{lbl}: Peak Fy = {peak_f:.1f} N | mu_eff = {mu:.3f} | CFa = {stiff:.1f} N/deg | By = {meta['By']:.2f}"
                )
            else:
                force, meta = self.model.evaluate_fx(x_rad, fz, gamma_rad)
                stiff = meta["Kx_kappa"]
                peak_f = meta["Dx"]
                mu = meta["mu_x"] * self.model.get("LMUX", 1.0)
                summary_lines.append(
                    f"{lbl}: Peak Fx = {peak_f:.1f} N | mu_eff = {mu:.3f} | Kx = {stiff:.1f} N/slip | Bx = {meta['Bx']:.2f}"
                )

            ax.plot(x_shown, force, color=color, linewidth=2.3, label=lbl)

        style_axes_2d(ax, self.fig, title=f"{title_prefix} — MF 5.1 / PAC2002", xlabel=xlabel, ylabel=ylabel, c=c)
        style_legend(ax, c=c)
        self.fig.tight_layout(pad=1.4)
        self.canvas.draw()
        self._update_kpi_badges()
        self._set_info("\n".join(summary_lines))
        if notify:
            self._on_model_updated()

    # --------------------------------------------------------------------------
    # EXTRACTION / FIT IMPORT / PRESET
    # --------------------------------------------------------------------------
    def auto_import_from_all_files(self):
        runs = self._get_runs() if self._get_runs else []
        df_corn = self._get_df_cornering() if self._get_df_cornering else None
        df_drive = self._get_df_drive() if self._get_df_drive else None

        has_data = (
            bool(runs)
            or (df_corn is not None and not df_corn.empty)
            or (df_drive is not None and not df_drive.empty)
        )
        if not has_data:
            return messagebox.showwarning(
                "No Telemetry Data",
                "No measurement files (.mat / .csv) or FSAE Demo data are loaded yet.\n\n"
                "Open Import... in the top bar or load data in Step 1 first.",
            )

        self._set_status("Running Multi-Fz extraction and MF 5.1 model optimization...")
        self.update_idletasks()

        try:
            result = AutoMF51Extractor.extract_from_app_data(
                self.model,
                runs=runs,
                df_cornering=df_corn,
                df_drive=df_drive,
            )
        except Exception as exc:
            messagebox.showerror("Multi-Fz Extraction Error", f"Unexpected error during data extraction:\n{exc}")
            self._set_status("MF 5.1 fitting failed.")
            return

        if not result.success:
            messagebox.showwarning("Extraction Unsuccessful", result.message)
            self._set_status("Could not extract MF 5.1 parameters from loaded data.")
            return

        self.sync_from_model(redraw=False)

        detected_fz = result.fz_levels_corn or result.fz_levels_drive
        if detected_fz:
            self.var_sweep_levels.set(", ".join(f"{x:.0f}" for x in detected_fz))

        diag_text = (
            "--- AUTOMATED MULTI-FZ EXTRACTION REPORT (MF 5.1) ---\n"
            + "\n".join(result.diagnostics_log)
        )
        self._set_info(diag_text)
        self.draw(notify=True)

        messagebox.showinfo("MF 5.1 Extraction Complete", result.message)
        self._set_status(f"MF 5.1 model fitted across {len(result.files_analyzed)} file(s).")

    def load_from_fit_results(self):
        fit_results = self._get_fit_results()
        if not fit_results:
            return messagebox.showinfo(
                "No Fit Results",
                "No curve fits are available from Step 2 yet.\nFit at least one model (e.g. Fy0, Fy_load) in Step 2 first.",
            )

        imported = self.model.import_from_fit_results(fit_results)
        if not imported:
            return messagebox.showwarning("Import", "No matching MF 5.1 parameters found in fit results.")

        self.sync_from_model(redraw=False)
        self.draw(notify=True)
        msg = "Mapped coefficients from Step 2 into active MF 5.1 model:\n\n• " + "\n• ".join(imported)
        messagebox.showinfo("Step 2 Fit Imported", msg)
        self._set_status("Updated MF 5.1 model from Step 2 fit results.")

    def reset_to_fsae_defaults(self):
        for key, val in DEFAULT_MF51_PARAMS.items():
            self.model.set(key, val)
        self.sync_from_model(redraw=False)
        self.lbl_scale_status.configure(
            text="● Scaling: Base Model (Hoosier R25B / FSAE Preset)",
            fg=self.colors["text_muted"],
        )
        self.draw(notify=True)
        self._set_status("Restored default Formula Student preset (Hoosier 18x6 / R25B base).")

    # --------------------------------------------------------------------------
    # PROGRAMMATIC EXPORTS (Unified Export Studio routes here cleanly)
    # --------------------------------------------------------------------------
    def export_tir(self, path: Optional[str] = None, tire_name: Optional[str] = None):
        self._sync_model_from_ui()
        if not path:
            path = filedialog.asksaveasfilename(
                defaultextension=".tir",
                filetypes=[("Pacejka Tire Property File (*.tir)", "*.tir"), ("All Files", "*.*")],
                initialfile="PUTM_Racing_MF51.tir",
            )
        if not path:
            return
        name = tire_name or os.path.splitext(os.path.basename(path))[0]
        content = self.model.to_tir_content(name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        self._set_status(f"Exported .TIR file: {path}")

    def export_json(self, path: Optional[str] = None):
        self._sync_model_from_ui()
        if not path:
            path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")], initialfile="PUTM_MF51_Params.json")
        if not path:
            return
        data = {
            "model_type": "Magic Formula 5.1 / PAC2002",
            "parameters": self.model.p,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self._set_status(f"Saved MF 5.1 JSON parameters to {path}")

    def build_curves_dataframe(self):
        import pandas as pd
        self._sync_model_from_ui()
        channel = self.var_channel.get()
        try:
            half_range = abs(float(self.var_range.get().replace(",", ".")))
        except ValueError:
            half_range = 15.0
        try:
            levels = [float(v.strip().replace(",", ".")) for v in self.var_sweep_levels.get().split(",") if v.strip()]
        except ValueError:
            levels = [400.0, 800.0, 1200.0, 1600.0]
        try:
            const_val = float(self.var_const_camber.get().replace(",", "."))
        except ValueError:
            const_val = 0.0

        x = np.linspace(-half_range, half_range, 301)
        x_rad = np.deg2rad(x) if "Fx" not in channel else x
        data = {"Slip_Angle_or_Ratio": x}
        for lvl in levels:
            fz, gamma = (lvl, const_val) if self.var_sweep_mode.get() == SWEEP_MODES[0] else (const_val, lvl)
            g_rad = np.deg2rad(gamma)
            if "Fy" in channel:
                y, _ = self.model.evaluate_fy(x_rad, fz, g_rad)
            elif "Fx" in channel:
                y, _ = self.model.evaluate_fx(x_rad, fz, g_rad)
            elif "Mz" in channel:
                y, _, _, _ = self.model.evaluate_mz(x_rad, fz, g_rad)
            else:
                y = self.model.evaluate_combined_fy(x_rad, 0.1, fz, g_rad)
            data[f"Level_{lvl:g}"] = np.round(y, 4)
        return pd.DataFrame(data)

    def export_csv(self, path: Optional[str] = None, *, sep: str = ","):
        df = self.build_curves_dataframe()
        if not path:
            path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")], initialfile="PUTM_MF51_Curves.csv")
        if not path:
            return
        df.to_csv(path, sep=sep, index=False, encoding="utf-8")
        self._set_status(f"Saved MF 5.1 curves to {path}")
