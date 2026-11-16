"""
Step 4 — 3D Simulator, Friction Cone & FSAE G-G Maneuver Analyzer (HubTab).

Carbon Motorsport Telemetry Edition:
- Operating envelope (Fz, Camber, Surface) + Live Maneuver Simulator (SA, SL)
- Debounced 3D/2D canvas updates for smooth slider interaction
- Shared MF 5.1 model sync with Step 3 & Step 5 + optional TTC telemetry overlay
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from auto_mf51_extractor import AutoMF51Extractor
from mf51_model import MF51Model
from tire_scaler import scale_18x6_to_16x75_r20
from ui_scroll import ScrollableFrame
from ui_theme import (
    CHART_PALETTE,
    FONT_MONO,
    FONT_MONO_BOLD,
    FONT_SMALL,
    FONT_SMALL_BOLD,
    PALETTE,
    CollapsibleCard,
    KpiTile,
    ModernCard,
    SegmentedToggle,
    StepHeader,
    style_axes_2d,
    style_colorbar,
    style_legend,
)


MODES_3D = [
    "3D: Fy(SA, Fz) — Lateral Force Surface",
    "3D: Fx(SL, Fz) — Longitudinal Force Surface",
    "3D: Mz(SA, Fz) — Aligning Torque Surface",
    "3D: Friction Cone (Combined Fx vs Fy)",
    "2D: Dynamic Curve at Active Fz & IA",
    "G-G: FSAE Maneuver Simulator (SA, SL -> Fx, Fy, mu)",
]


def _model_label(m: MF51Model) -> str:
    w = m.get("WIDTH", 0.1524)
    r0 = m.get("UNLOADED_RADIUS", 0.2286)
    if abs(w - 0.180) < 0.015 or abs(r0 - 0.205) < 0.012:
        return f"Hoosier 16x7.5-10 (W={w*1000:.0f}mm)"
    return f"MF 5.1 (R0={r0*1000:.0f}mm, W={w*1000:.0f}mm)"


def generate_fsae_demo_telemetry() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic TTC telemetry (Hoosier FSAE 18x6-10 R25B) matching SAE TTC sign conventions."""
    rng = np.random.default_rng(42)
    ref_model = MF51Model()

    sa_grid = np.linspace(-13.5, 13.5, 95)
    sa_rad = np.radians(sa_grid)
    c_rows = []
    for fz_val in (450.0, 900.0, 1350.0):
        for ia_val in (0.0, 2.0, 4.0):
            ia_rad = float(np.radians(ia_val))
            fy_clean, _ = ref_model.evaluate_fy(sa_rad, fz_val, ia_rad)
            mz_clean, _, _, _ = ref_model.evaluate_mz(sa_rad, fz_val, ia_rad)
            fy = -fy_clean + rng.normal(0.0, 28.0, size=sa_rad.size)
            mz = mz_clean + rng.normal(0.0, 1.5, size=sa_rad.size)
            fx = rng.normal(0.0, 12.0, size=sa_rad.size)
            fz_arr = fz_val + rng.normal(0.0, 6.0, size=sa_rad.size)
            c_rows.append(
                pd.DataFrame(
                    {
                        "SA": sa_grid,
                        "SL": np.zeros_like(sa_grid),
                        "FZ": np.maximum(100.0, fz_arr),
                        "IA": np.full_like(sa_grid, ia_val),
                        "P": np.full_like(sa_grid, 82.7),
                        "FY": fy,
                        "FX": fx,
                        "MZ": mz,
                        "V": np.full_like(sa_grid, 40.2),
                    }
                )
            )
    df_c = pd.concat(c_rows, ignore_index=True)

    sl_grid = np.linspace(-0.24, 0.24, 95)
    d_rows = []
    for fz_val in (450.0, 900.0, 1350.0):
        for ia_val in (0.0, 2.0):
            ia_rad = float(np.radians(ia_val))
            fx_clean, _ = ref_model.evaluate_fx(sl_grid, fz_val, ia_rad)
            fx = fx_clean + rng.normal(0.0, 32.0, size=sl_grid.size)
            fy = rng.normal(0.0, 16.0, size=sl_grid.size)
            mz = rng.normal(0.0, 1.1, size=sl_grid.size)
            fz_arr = fz_val + rng.normal(0.0, 6.0, size=sl_grid.size)
            d_rows.append(
                pd.DataFrame(
                    {
                        "SA": np.zeros_like(sl_grid),
                        "SL": sl_grid,
                        "FZ": np.maximum(100.0, fz_arr),
                        "IA": np.full_like(sl_grid, ia_val),
                        "P": np.full_like(sl_grid, 82.7),
                        "FY": fy,
                        "FX": fx,
                        "MZ": mz,
                        "V": np.full_like(sl_grid, 40.2),
                    }
                )
            )
    df_d = pd.concat(d_rows, ignore_index=True)
    return df_c, df_d


class HubTab(ttk.Frame):
    """Step 4 — 3D Tire Force Simulator & Live FSAE G-G Maneuver Analyzer."""

    def __init__(
        self,
        parent,
        *,
        colors=None,
        fonts=None,
        set_status: Optional[Callable[[str], None]] = None,
        get_model: Optional[Callable[[], MF51Model]] = None,
        set_model: Optional[Callable[[MF51Model], None]] = None,
        on_model_updated: Optional[Callable[[], None]] = None,
        get_df_cornering: Optional[Callable[[], Optional[pd.DataFrame]]] = None,
        get_df_drive: Optional[Callable[[], Optional[pd.DataFrame]]] = None,
        get_runs: Optional[Callable[[], list]] = None,
        on_data_loaded: Optional[Callable[[pd.DataFrame, pd.DataFrame], None]] = None,
        on_edit_model: Optional[Callable[[], None]] = None,
        on_next_step: Optional[Callable[[], None]] = None,
        show_step_header: bool = True,
    ):
        super().__init__(parent)
        self.c = colors or PALETTE
        self.fonts = fonts or {}
        self._set_status_cb = set_status or (lambda _msg: None)
        self._get_model_cb = get_model
        self._set_model_cb = set_model
        self._on_model_updated_cb = on_model_updated or (lambda: None)
        self._get_df_c_cb = get_df_cornering or (lambda: None)
        self._get_df_d_cb = get_df_drive or (lambda: None)
        self._get_runs_cb = get_runs or (lambda: [])
        self._on_data_loaded_cb = on_data_loaded
        self._on_edit_model_cb = on_edit_model
        self._on_next_step_cb = on_next_step
        self._show_step_header = show_step_header

        self._local_model = MF51Model()
        self._draw_job: Optional[str] = None
        self._cbar = None
        self._cam_bar_visible = True

        init_surf = "asphalt" if self.model.get("LMUY", 1.0) < 0.85 else "ttc"
        self.var_surface = tk.StringVar(value=init_surf)
        self.var_mode = tk.StringVar(value=MODES_3D[0])
        self.var_fz = tk.DoubleVar(value=900.0)
        self.var_camber = tk.DoubleVar(value=2.0)
        self.var_overlay_ttc = tk.BooleanVar(value=True)
        self.var_elev = tk.DoubleVar(value=26.0)
        self.var_azim = tk.DoubleVar(value=-48.0)

        self.var_sim_sa = tk.DoubleVar(value=6.0)
        self.var_sim_sl = tk.DoubleVar(value=0.05)

        self._build_ui()
        self.sync_from_model(redraw=True)

    @property
    def model(self) -> MF51Model:
        if self._get_model_cb is not None:
            return self._get_model_cb()
        return self._local_model

    @model.setter
    def model(self, value: MF51Model):
        if self._set_model_cb is not None:
            self._set_model_cb(value)
        else:
            self._local_model = value

    def _status(self, msg: str):
        self._set_status_cb(msg)

    # ------------------------------------------------------------------
    # UI CONSTRUCTION
    # ------------------------------------------------------------------
    def _build_ui(self):
        c = self.c
        self.configure(style="TFrame")

        if self._show_step_header:
            StepHeader(
                self,
                "STEP 4 / 5",
                "3D Force Simulator & G-G Maneuver Envelope",
                "Explore 3D MF 5.1 force surfaces, combined-slip friction cone, and live FSAE cornering/braking grip utilization.",
                badge_kind="cyan",
                next_text="Next: .TIR Comparator (Step 5) ->" if self._on_next_step_cb else None,
                next_command=self._on_next_step_cb,
                colors=c,
            ).pack(fill="x", pady=(0, 8))

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)

        left_wrap = tk.Frame(body, bg=c["bg"], width=425)
        left_wrap.pack(side="left", fill="y", padx=(0, 10))
        left_wrap.pack_propagate(False)

        self._left_scroll = ScrollableFrame(left_wrap, bg=c["bg"], padx=2, pady=2)
        self._left_scroll.pack(fill="both", expand=True)
        ctrl = self._left_scroll.inner

        # 1. Surface & Mode card
        card_view = ModernCard(
            ctrl,
            "1. Track Surface & Visualization Mode",
            badge="LIVE",
            badge_color="cyan",
            accent_color=c["cyan"],
            colors=c,
            pad_x=10,
            pad_y=8,
        )
        card_view.pack(fill="x", pady=(0, 8))

        self._surface_seg = SegmentedToggle(
            card_view.body,
            self.var_surface,
            [
                ("asphalt", "Asphalt (mu x0.65)"),
                ("ttc", "TTC Flat-Trac (1.00)"),
            ],
            command=self._on_surface_change,
            accent_key="accent",
            colors=c,
        )
        self._surface_seg.pack(fill="x", pady=(0, 6))

        self.lbl_surface_info = tk.Label(
            card_view.body,
            text="",
            bg=c["panel"],
            fg=c["text_muted"],
            font=FONT_SMALL,
            anchor="w",
        )
        self.lbl_surface_info.pack(fill="x", pady=(0, 6))

        tk.Label(
            card_view.body,
            text="Plot view:",
            bg=c["panel"],
            fg=c["text_secondary"],
            font=FONT_SMALL_BOLD,
            anchor="w",
        ).pack(fill="x")

        mode_cb = ttk.Combobox(
            card_view.body,
            textvariable=self.var_mode,
            values=MODES_3D,
            state="readonly",
        )
        mode_cb.pack(fill="x", pady=(3, 6))
        mode_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_mode_selected())

        opt_row = tk.Frame(card_view.body, bg=c["panel"])
        opt_row.pack(fill="x")
        ttk.Checkbutton(
            opt_row,
            text="Overlay TTC telemetry",
            variable=self.var_overlay_ttc,
            command=lambda: self._schedule_draw(10),
        ).pack(side="left")

        ttk.Button(
            opt_row,
            text="Reset 3D Camera",
            style="Pill.TButton",
            command=self.reset_camera,
        ).pack(side="right")

        # 2. Load & Camber card
        card_env = ModernCard(
            ctrl,
            "2. Operating Load & Camber",
            accent_color=c["accent"],
            colors=c,
            pad_x=10,
            pad_y=8,
        )
        card_env.pack(fill="x", pady=(0, 8))

        fz_hdr = tk.Frame(card_env.body, bg=c["panel"])
        fz_hdr.pack(fill="x")
        tk.Label(fz_hdr, text="Vertical Load Fz [N]:", bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL_BOLD).pack(side="left")
        self.lbl_fz_val = tk.Label(fz_hdr, text="900 N", bg=c["panel"], fg=c["accent_hover"], font=FONT_MONO_BOLD)
        self.lbl_fz_val.pack(side="right")

        self.fz_scale = ttk.Scale(
            card_env.body,
            from_=150.0,
            to=2000.0,
            variable=self.var_fz,
            command=self._on_fz_scale,
        )
        self.fz_scale.pack(fill="x", pady=(3, 4))

        fz_pills = tk.Frame(card_env.body, bg=c["panel"])
        fz_pills.pack(fill="x", pady=(0, 8))
        for val in (250, 500, 900, 1350):
            ttk.Button(fz_pills, text=f"{val} N", style="Pill.TButton", command=lambda v=val: self._set_fz(v)).pack(side="left", padx=(0, 4))

        ia_hdr = tk.Frame(card_env.body, bg=c["panel"])
        ia_hdr.pack(fill="x")
        tk.Label(ia_hdr, text="Camber IA [deg]:", bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL_BOLD).pack(side="left")
        self.lbl_camber_val = tk.Label(ia_hdr, text="+2.0°", bg=c["panel"], fg=c["cyan"], font=FONT_MONO_BOLD)
        self.lbl_camber_val.pack(side="right")

        self.camber_scale = ttk.Scale(
            card_env.body,
            from_=-5.0,
            to=5.0,
            variable=self.var_camber,
            command=self._on_camber_scale,
        )
        self.camber_scale.pack(fill="x", pady=(3, 4))

        ia_pills = tk.Frame(card_env.body, bg=c["panel"])
        ia_pills.pack(fill="x")
        for val in (0.0, 2.0, 4.0, -2.0):
            ttk.Button(
                ia_pills,
                text=f"{val:+.0f}°" if val != 0 else "0°",
                style="Pill.TButton",
                command=lambda v=val: self._set_camber(v),
            ).pack(side="left", padx=(0, 4))

        # 3. Maneuver Simulator card
        card_sim = ModernCard(
            ctrl,
            "3. FSAE Maneuver Simulator (SA x SL)",
            badge="G-G",
            badge_color="ok",
            accent_color=c["ok"],
            colors=c,
            pad_x=10,
            pad_y=8,
        )
        card_sim.pack(fill="x", pady=(0, 8))

        ttk.Button(
            card_sim.body,
            text="Show G-G Friction Circle View ->",
            style="Cyan.TButton",
            command=self._switch_to_maneuver_sim,
        ).pack(fill="x", pady=(0, 6))

        sa_hdr = tk.Frame(card_sim.body, bg=c["panel"])
        sa_hdr.pack(fill="x")
        tk.Label(sa_hdr, text="Slip Angle SA [deg]:", bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL_BOLD).pack(side="left")
        self.lbl_sim_sa = tk.Label(sa_hdr, text="+6.0°", bg=c["panel"], fg=c["text"], font=FONT_MONO_BOLD)
        self.lbl_sim_sa.pack(side="right")

        ttk.Scale(card_sim.body, from_=-14.0, to=14.0, variable=self.var_sim_sa, command=self._on_sim_sa_scale).pack(fill="x", pady=(2, 4))

        sa_pills = tk.Frame(card_sim.body, bg=c["panel"])
        sa_pills.pack(fill="x", pady=(0, 6))
        for val in (0.0, 4.0, 8.0, 12.0):
            ttk.Button(sa_pills, text=f"{val:.0f}°", style="Pill.TButton", command=lambda v=val: self._set_sim_sa(v)).pack(side="left", padx=(0, 4))

        sl_hdr = tk.Frame(card_sim.body, bg=c["panel"])
        sl_hdr.pack(fill="x")
        tk.Label(sl_hdr, text="Longitudinal Slip SL [-]:", bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL_BOLD).pack(side="left")
        self.lbl_sim_sl = tk.Label(sl_hdr, text="+0.050", bg=c["panel"], fg=c["text"], font=FONT_MONO_BOLD)
        self.lbl_sim_sl.pack(side="right")

        ttk.Scale(card_sim.body, from_=-0.25, to=0.25, variable=self.var_sim_sl, command=self._on_sim_sl_scale).pack(fill="x", pady=(2, 4))

        sl_pills = tk.Frame(card_sim.body, bg=c["panel"])
        sl_pills.pack(fill="x", pady=(0, 8))
        for val in (-0.12, 0.0, 0.06, 0.14):
            ttk.Button(sl_pills, text=f"{val:+.2f}" if val != 0 else "0.00", style="Pill.TButton", command=lambda v=val: self._set_sim_sl(v)).pack(side="left", padx=(0, 4))

        telem_box = tk.Frame(
            card_sim.body,
            bg=c["panel_alt"],
            highlightthickness=1,
            highlightbackground=c["border"],
            padx=8,
            pady=6,
        )
        telem_box.pack(fill="x")

        self.lbl_sim_regime = tk.Label(telem_box, text="--", bg=c["panel_alt"], fg=c["cyan"], font=FONT_SMALL_BOLD, anchor="w")
        self.lbl_sim_regime.pack(fill="x")

        self.lbl_sim_telemetry = tk.Label(
            telem_box,
            text="Fx = -- | Fy = -- | mu = --",
            bg=c["panel_alt"],
            fg=c["text"],
            font=FONT_MONO,
            justify="left",
            anchor="w",
        )
        self.lbl_sim_telemetry.pack(fill="x", pady=(3, 4))

        util_row = tk.Frame(telem_box, bg=c["panel_alt"])
        util_row.pack(fill="x")
        tk.Label(util_row, text="Grip utilization:", bg=c["panel_alt"], fg=c["text_muted"], font=FONT_SMALL).pack(side="left")
        self.lbl_sim_util_pct = tk.Label(util_row, text="0%", bg=c["panel_alt"], fg=c["ok"], font=FONT_MONO_BOLD)
        self.lbl_sim_util_pct.pack(side="right")

        self.prog_grip = ttk.Progressbar(
            telem_box,
            orient="horizontal",
            mode="determinate",
            maximum=100.0,
            style="Horizontal.TProgressbar",
        )
        self.prog_grip.pack(fill="x", pady=(3, 0))

        # 4. Collapsible Model Snapshot (read-only + quick link back to Editor)
        card_model = CollapsibleCard(
            ctrl,
            "Active MF 5.1 Snapshot",
            is_open=False,
            badge="MF51",
            badge_color="purple",
            colors=c,
            pad_x=10,
            pad_y=8,
        )
        card_model.pack(fill="x", pady=(0, 8))

        self.lbl_model_snapshot = tk.Label(
            card_model.body,
            text="",
            bg=c["panel"],
            fg=c["text_secondary"],
            font=FONT_MONO,
            justify="left",
            anchor="w",
        )
        self.lbl_model_snapshot.pack(fill="x", pady=(0, 6))

        if self._on_edit_model_cb is not None:
            ttk.Button(card_model.body, text="<- Edit Parameters in Step 3", style="Pill.TButton", command=self._on_edit_model_cb).pack(fill="x")

        # ------------------------------------------------------------------
        # RIGHT PANEL: KPI Tiles + Plot Canvas
        # ------------------------------------------------------------------
        right = ttk.Frame(body)
        right.pack(side="right", fill="both", expand=True)

        kpi_grid = tk.Frame(right, bg=c["bg"])
        kpi_grid.pack(fill="x", pady=(0, 8))
        for col in range(6):
            kpi_grid.columnconfigure(col, weight=1)

        self.tile_peak_fy = KpiTile(kpi_grid, "PEAK FY", "--", accent=c["accent"], colors=c)
        self.tile_peak_fy.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.tile_peak_fx = KpiTile(kpi_grid, "PEAK FX", "--", accent=c["cyan"], colors=c)
        self.tile_peak_fx.grid(row=0, column=1, sticky="ew", padx=(0, 5))

        self.tile_cfa = KpiTile(kpi_grid, "STIFFNESS CFa", "--", accent=c["ok"], colors=c)
        self.tile_cfa.grid(row=0, column=2, sticky="ew", padx=(0, 5))

        self.tile_peak_mz = KpiTile(kpi_grid, "PEAK |MZ|", "--", accent=c["purple"], colors=c)
        self.tile_peak_mz.grid(row=0, column=3, sticky="ew", padx=(0, 5))

        self.tile_mu = KpiTile(kpi_grid, "MAX MU (Fy/Fz)", "--", accent=c["warn"], colors=c)
        self.tile_mu.grid(row=0, column=4, sticky="ew", padx=(0, 5))

        self.tile_sim_grip = KpiTile(kpi_grid, "MANEUVER GRIP", "--", accent=c["cyan"], colors=c)
        self.tile_sim_grip.grid(row=0, column=5, sticky="ew")

        plot_card = ModernCard(right, "3D & G-G Tire Performance Visualization", colors=c, pad_x=8, pad_y=8)
        plot_card.pack(fill="both", expand=True)

        toolbar = tk.Frame(plot_card.body, bg=c["panel"])
        toolbar.pack(fill="x", pady=(0, 4))

        self.cam_bar = tk.Frame(toolbar, bg=c["panel"])
        self.cam_bar.pack(side="left", fill="x", expand=True)

        tk.Label(self.cam_bar, text="3D Camera · Elev:", bg=c["panel"], fg=c["text_muted"], font=FONT_SMALL).pack(side="left", padx=(0, 4))
        ttk.Scale(self.cam_bar, from_=5.0, to=80.0, variable=self.var_elev, length=110, command=lambda _v: self._schedule_draw(25)).pack(side="left", padx=(0, 10))

        tk.Label(self.cam_bar, text="Azim:", bg=c["panel"], fg=c["text_muted"], font=FONT_SMALL).pack(side="left", padx=(0, 4))
        ttk.Scale(self.cam_bar, from_=-160.0, to=160.0, variable=self.var_azim, length=110, command=lambda _v: self._schedule_draw(25)).pack(side="left")

        self.fig = Figure(figsize=(7.6, 5.2), dpi=100, facecolor=c["panel"])
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_card.body)
        self.canvas.get_tk_widget().configure(bg=c["panel"], highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # MODEL SYNC & CALLBACKS
    # ------------------------------------------------------------------
    def sync_from_model(self, redraw: bool = True):
        m = self.model
        lmuy = float(m.get("LMUY", 1.0))
        lmux = float(m.get("LMUX", 1.0))

        target_surf = "asphalt" if lmuy < 0.85 else "ttc"
        if self.var_surface.get() != target_surf:
            self.var_surface.set(target_surf)
            self._surface_seg.refresh()

        lbl = _model_label(m)
        self.lbl_surface_info.config(text=f"Friction Scale: LMUX={lmux:.3f} | LMUY={lmuy:.3f} | {lbl}")
        self.lbl_model_snapshot.config(
            text=(
                f"{lbl}\n"
                f"PDY1={m.get('PDY1'):.3f}  PDY2={m.get('PDY2'):.3f}  PKY1={m.get('PKY1'):.2f}\n"
                f"PDX1={m.get('PDX1'):.3f}  PDX2={m.get('PDX2'):.3f}  QDZ1={m.get('QDZ1'):.3f}"
            )
        )
        self.update_kpis()
        if redraw:
            self._schedule_draw(10)

    def _on_mode_selected(self):
        mode = self.var_mode.get()
        is_3d = mode.startswith("3D:")
        if is_3d and not self._cam_bar_visible:
            self.cam_bar.pack(side="left", fill="x", expand=True)
            self._cam_bar_visible = True
        elif not is_3d and self._cam_bar_visible:
            self.cam_bar.pack_forget()
            self._cam_bar_visible = False
        self._schedule_draw(10)

    def _on_fz_scale(self, val):
        self.lbl_fz_val.config(text=f"{float(val):.0f} N")
        self.update_kpis()
        self._schedule_draw(30)

    def _set_fz(self, val: float):
        self.var_fz.set(float(val))
        self.lbl_fz_val.config(text=f"{float(val):.0f} N")
        self.update_kpis()
        self._schedule_draw(10)

    def _on_camber_scale(self, val):
        self.lbl_camber_val.config(text=f"{float(val):+.1f}°")
        self.update_kpis()
        self._schedule_draw(30)

    def _set_camber(self, val: float):
        self.var_camber.set(float(val))
        self.lbl_camber_val.config(text=f"{float(val):+.1f}°")
        self.update_kpis()
        self._schedule_draw(10)

    def _on_sim_sa_scale(self, val):
        self.lbl_sim_sa.config(text=f"{float(val):+.1f}°")
        self._update_sim_telemetry()
        if self.var_mode.get().startswith("G-G"):
            self._schedule_draw(25)

    def _set_sim_sa(self, val: float):
        self.var_sim_sa.set(float(val))
        self.lbl_sim_sa.config(text=f"{float(val):+.1f}°")
        self._update_sim_telemetry()
        if self.var_mode.get().startswith("G-G"):
            self._schedule_draw(10)

    def _on_sim_sl_scale(self, val):
        self.lbl_sim_sl.config(text=f"{float(val):+.3f}")
        self._update_sim_telemetry()
        if self.var_mode.get().startswith("G-G"):
            self._schedule_draw(25)

    def _set_sim_sl(self, val: float):
        self.var_sim_sl.set(float(val))
        self.lbl_sim_sl.config(text=f"{float(val):+.3f}")
        self._update_sim_telemetry()
        if self.var_mode.get().startswith("G-G"):
            self._schedule_draw(10)

    def _switch_to_maneuver_sim(self):
        self.var_mode.set(MODES_3D[-1])
        self._on_mode_selected()

    def _on_surface_change(self):
        surf = self.var_surface.get()
        scale = 0.65 if surf == "asphalt" else 1.00
        self.model.set("LMUX", scale)
        self.model.set("LMUY", scale)
        self._status(f"Surface set: {'Asphalt (x0.65)' if surf == 'asphalt' else 'TTC Flat-Trac (1.00)'}")
        self._on_model_updated_cb()
        self.sync_from_model(redraw=True)

    def reset_camera(self):
        self.var_elev.set(26.0)
        self.var_azim.set(-48.0)
        self._schedule_draw(10)

    # ------------------------------------------------------------------
    # TELEMETRY & KPIs
    # ------------------------------------------------------------------
    def _update_sim_telemetry(self):
        fz = float(self.var_fz.get())
        ia = float(np.radians(float(self.var_camber.get())))
        sa_deg = float(self.var_sim_sa.get())
        sa = float(np.radians(sa_deg))
        sl = float(self.var_sim_sl.get())

        m = self.model
        fx = float(m.evaluate_combined_fx(np.array([sl]), sa, fz, ia)[0])
        fy = float(m.evaluate_combined_fy(np.array([sa]), sl, fz, ia)[0])
        mz_arr, _, _, _ = m.evaluate_mz(np.array([sa]), fz, ia)
        mz = float(mz_arr[0])
        f_res = float(np.hypot(fx, fy))
        mu_use = f_res / max(fz, 1.0)

        sa_grid = np.radians(np.linspace(-14.0, 14.0, 101))
        sl_grid = np.linspace(-0.25, 0.25, 101)
        fy_all, _ = m.evaluate_fy(sa_grid, fz, ia)
        fx_all, _ = m.evaluate_fx(sl_grid, fz, ia)
        f_cap = max(float(np.max(np.abs(fy_all))), float(np.max(np.abs(fx_all))), 1.0)
        util_pct = min(100.0, 100.0 * f_res / f_cap)

        if abs(sa_deg) < 0.3 and abs(sl) < 0.005:
            regime = "Straight running (free rolling)"
        elif abs(sl) < 0.01:
            regime = "Pure Cornering"
        elif abs(sa_deg) < 0.5:
            regime = "Straight-line Drive / Braking"
        elif sl > 0:
            regime = "Combined Slip: Corner Exit (Traction + SA)"
        else:
            regime = "Combined Slip: Corner Entry (Trail Braking + SA)"

        self.lbl_sim_regime.config(text=regime)
        self.lbl_sim_telemetry.config(
            text=(
                f"Fx = {fx:+7.0f} N   |  Fy = {fy:+7.0f} N\n"
                f"|F| = {f_res:6.0f} N   |  Mz = {mz:+6.1f} Nm  |  mu = {mu_use:.3f}"
            )
        )
        self.lbl_sim_util_pct.config(
            text=f"{util_pct:.1f}%",
            fg=self.c["danger"] if util_pct > 92 else (self.c["warn"] if util_pct > 75 else self.c["ok"]),
        )
        self.prog_grip["value"] = util_pct
        self.tile_sim_grip.set_value(f"{util_pct:.0f}%")

    def update_kpis(self):
        fz = float(self.var_fz.get())
        ia = float(np.radians(float(self.var_camber.get())))
        m = self.model

        sa_rad = np.radians(np.linspace(-14.0, 14.0, 281))
        sl_arr = np.linspace(-0.25, 0.25, 201)

        fy, _ = m.evaluate_fy(sa_rad, fz, ia)
        fx, _ = m.evaluate_fx(sl_arr, fz, ia)
        mz, _, _, _ = m.evaluate_mz(sa_rad, fz, ia)

        peak_fy = float(np.max(np.abs(fy)))
        peak_fx = float(np.max(np.abs(fx)))
        peak_mz = float(np.max(np.abs(mz)))

        sa_sm = np.radians(np.linspace(-0.5, 0.5, 21))
        fy_sm, _ = m.evaluate_fy(sa_sm, fz, ia)
        cfa = abs(float(np.polyfit(np.degrees(sa_sm), fy_sm, 1)[0]))

        self.tile_peak_fy.set_value(f"{peak_fy:.0f} N")
        self.tile_peak_fx.set_value(f"{peak_fx:.0f} N")
        self.tile_cfa.set_value(f"{cfa:.0f} N/°")
        self.tile_peak_mz.set_value(f"{peak_mz:.1f} Nm")
        self.tile_mu.set_value(f"{peak_fy / max(fz, 1.0):.2f}")

        self._update_sim_telemetry()

    # ------------------------------------------------------------------
    # COMPATIBILITY ACTIONS
    # ------------------------------------------------------------------
    def run_tire_scaling(self):
        m = self.model.copy()
        rep = scale_18x6_to_16x75_r20(m)
        self.model = m
        self._on_model_updated_cb()
        self.sync_from_model(redraw=True)
        self._status(f"Scaled tire model: {rep.source_tire} -> {rep.target_tire}")
        return rep

    def run_autopilot(self):
        runs = self._get_runs_cb()
        df_c = self._get_df_c_cb()
        df_d = self._get_df_d_cb()
        if df_c is None and df_d is None and not runs:
            messagebox.showwarning("No Data", "Load TTC data in Step 1 or run the FSAE Demo first.")
            return None
        m = self.model.copy()
        res = AutoMF51Extractor.extract_from_app_data(m, runs, df_cornering=df_c, df_drive=df_d)
        if res.success:
            self.model = m
            self._on_model_updated_cb()
            self.sync_from_model(redraw=True)
        self._status(res.message.splitlines()[0] if res.message else "Autopilot complete.")
        return res

    def load_fsae_demo(self):
        df_c, df_d = generate_fsae_demo_telemetry()
        if self._on_data_loaded_cb is not None:
            self._on_data_loaded_cb(df_c, df_d)
        self._status(f"Loaded FSAE Demo TTC ({len(df_c)} cornering pts, {len(df_d)} drive pts).")
        self.sync_from_model(redraw=True)
        return df_c, df_d

    def reset_to_preset(self):
        surf_scale = 0.65 if self.var_surface.get() == "asphalt" else 1.0
        preset = MF51Model()
        preset.set("LMUX", surf_scale)
        preset.set("LMUY", surf_scale)
        self.model = preset
        self._on_model_updated_cb()
        self.sync_from_model(redraw=True)
        self._status("Restored default Hoosier 18x6-10 R25B preset.")

    def export_tir(self, path: Optional[str] = None):
        if not path:
            path = filedialog.asksaveasfilename(
                title="Export .TIR Tire Property File (MF 5.1)",
                defaultextension=".tir",
                filetypes=[("ADAMS / CarSim TIR", "*.tir"), ("All Files", "*.*")],
                initialfile="PUTM_MF51_Tire.tir",
            )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model.to_tir_content())
        self._status(f"Exported MF 5.1 model to {path}")

    def save_plot(self, path: Optional[str] = None, dpi: int = 220):
        if not path:
            path = filedialog.asksaveasfilename(
                title="Save 3D / G-G Simulator Plot",
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
                initialfile="PUTM_Simulator3D.png",
            )
        if not path:
            return
        self.fig.savefig(path, dpi=dpi, facecolor=self.fig.get_facecolor())
        self._status(f"Saved plot: {path}")

    # ------------------------------------------------------------------
    # PLOTTING (DEBOUNCED)
    # ------------------------------------------------------------------
    def _schedule_draw(self, delay_ms: int = 25):
        if self._draw_job is not None:
            try:
                self.after_cancel(self._draw_job)
            except tk.TclError:
                pass
            self._draw_job = None
        self._draw_job = self.after(delay_ms, self.draw)

    def _style_3d_ax(self, ax, title: str, xlabel: str, ylabel: str, zlabel: str):
        c = self.c
        ax.set_facecolor(c["panel_alt"])
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_facecolor((0.08, 0.11, 0.16, 0.85))
            pane.set_edgecolor(c["border"])
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"]["color"] = c["border"]
            axis._axinfo["grid"]["linewidth"] = 0.6
        ax.tick_params(colors=c["text_secondary"], labelsize=8)
        ax.set_xlabel(xlabel, color=c["text_secondary"], fontsize=8.5, labelpad=6)
        ax.set_ylabel(ylabel, color=c["text_secondary"], fontsize=8.5, labelpad=6)
        ax.set_zlabel(zlabel, color=c["text_secondary"], fontsize=8.5, labelpad=6)
        ax.set_title(title, color=c["text"], fontsize=10.5, fontweight="bold", pad=8)
        ax.view_init(elev=float(self.var_elev.get()), azim=float(self.var_azim.get()))

    def draw(self):
        self._draw_job = None
        c = self.c
        m = self.model
        lbl = _model_label(m)
        mode = self.var_mode.get()
        fz_sel = float(self.var_fz.get())
        ia_deg = float(self.var_camber.get())
        ia_rad = float(np.radians(ia_deg))

        self.fig.clear()
        self.fig.patch.set_facecolor(c["panel"])
        self._cbar = None

        df_c = self._get_df_c_cb()
        df_d = self._get_df_d_cb()
        show_ttc = bool(self.var_overlay_ttc.get())

        if mode.startswith("3D: Fy"):
            ax = self.fig.add_subplot(111, projection="3d")
            sa_v = np.linspace(-14.0, 14.0, 55)
            sa_r = np.radians(sa_v)
            fz_v = np.linspace(200.0, 1800.0, 32)
            SA, FZ = np.meshgrid(sa_v, fz_v)
            FY = np.vstack([m.evaluate_fy(sa_r, float(fv), ia_rad)[0] for fv in fz_v])

            surf = ax.plot_surface(SA, FZ, FY, cmap="plasma", alpha=0.90, edgecolor="none", antialiased=True)
            fy_slice, _ = m.evaluate_fy(sa_r, fz_sel, ia_rad)
            ax.plot(sa_v, np.full_like(sa_v, fz_sel), fy_slice, color=c["cyan"], linewidth=2.6)

            if show_ttc and df_c is not None and not df_c.empty and "FY" in df_c and "SA" in df_c and "FZ" in df_c:
                sub = df_c.iloc[::max(1, len(df_c) // 450)]
                fy_sign = -1.0 if float((sub["SA"] * sub["FY"]).mean()) < 0.0 else 1.0
                ax.scatter(sub["SA"], sub["FZ"], fy_sign * sub["FY"], color=c["cyan"], s=10, alpha=0.55)

            self._style_3d_ax(ax, f"Surface Fy(SA, Fz) | IA={ia_deg:+.1f}° | {lbl}", "SA [deg]", "Fz [N]", "Fy [N]")
            self._cbar = self.fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08)
            style_colorbar(self._cbar, "Fy [N]", c)

        elif mode.startswith("3D: Fx"):
            ax = self.fig.add_subplot(111, projection="3d")
            sl_v = np.linspace(-0.25, 0.25, 55)
            fz_v = np.linspace(200.0, 1800.0, 32)
            SL, FZ = np.meshgrid(sl_v, fz_v)
            FX = np.vstack([m.evaluate_fx(sl_v, float(fv), ia_rad)[0] for fv in fz_v])

            surf = ax.plot_surface(SL, FZ, FX, cmap="viridis", alpha=0.90, edgecolor="none", antialiased=True)
            fx_slice, _ = m.evaluate_fx(sl_v, fz_sel, ia_rad)
            ax.plot(sl_v, np.full_like(sl_v, fz_sel), fx_slice, color=c["warn"], linewidth=2.6)

            if show_ttc and df_d is not None and not df_d.empty and "FX" in df_d and "SL" in df_d and "FZ" in df_d:
                sub = df_d.iloc[::max(1, len(df_d) // 450)]
                ax.scatter(sub["SL"], sub["FZ"], sub["FX"], color=c["warn"], s=10, alpha=0.55)

            self._style_3d_ax(ax, f"Surface Fx(SL, Fz) | IA={ia_deg:+.1f}° | {lbl}", "Slip Ratio SL [-]", "Fz [N]", "Fx [N]")
            self._cbar = self.fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08)
            style_colorbar(self._cbar, "Fx [N]", c)

        elif mode.startswith("3D: Mz"):
            ax = self.fig.add_subplot(111, projection="3d")
            sa_v = np.linspace(-14.0, 14.0, 55)
            sa_r = np.radians(sa_v)
            fz_v = np.linspace(200.0, 1800.0, 32)
            SA, FZ = np.meshgrid(sa_v, fz_v)
            MZ = np.vstack([m.evaluate_mz(sa_r, float(fv), ia_rad)[0] for fv in fz_v])

            surf = ax.plot_surface(SA, FZ, MZ, cmap="coolwarm", alpha=0.90, edgecolor="none", antialiased=True)
            if show_ttc and df_c is not None and not df_c.empty and "MZ" in df_c and "SA" in df_c and "FZ" in df_c:
                sub = df_c.iloc[::max(1, len(df_c) // 450)]
                ax.scatter(sub["SA"], sub["FZ"], sub["MZ"], color=c["cyan"], s=10, alpha=0.55)

            self._style_3d_ax(ax, f"Aligning Torque Mz(SA, Fz) | IA={ia_deg:+.1f}°", "SA [deg]", "Fz [N]", "Mz [Nm]")
            self._cbar = self.fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08)
            style_colorbar(self._cbar, "Mz [Nm]", c)

        elif mode.startswith("3D: Friction"):
            ax = self.fig.add_subplot(111, projection="3d")
            theta = np.linspace(0.0, 2.0 * np.pi, 64)
            fz_levels = np.linspace(250.0, 1800.0, 24)
            sa_ring = np.radians(12.0 * np.sin(theta))
            sl_ring = 0.16 * np.cos(theta)
            FZ = np.repeat(fz_levels[:, None], len(theta), axis=1)
            FX = np.vstack([m.evaluate_combined_fx(sl_ring, sa_ring, float(fv), ia_rad) for fv in fz_levels])
            FY = np.vstack([m.evaluate_combined_fy(sa_ring, sl_ring, float(fv), ia_rad) for fv in fz_levels])

            surf = ax.plot_surface(FX, FY, FZ, cmap="magma", alpha=0.86, edgecolor="none")
            fx_r = m.evaluate_combined_fx(sl_ring, sa_ring, fz_sel, ia_rad)
            fy_r = m.evaluate_combined_fy(sa_ring, sl_ring, fz_sel, ia_rad)
            ax.plot(fx_r, fy_r, np.full_like(theta, fz_sel), color=c["cyan"], linewidth=2.6)

            self._style_3d_ax(ax, f"3D Friction Cone (Combined Slip) | IA={ia_deg:+.1f}°", "Fx [N]", "Fy [N]", "Fz [N]")
            self._cbar = self.fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08)
            style_colorbar(self._cbar, "Fz [N]", c)

        elif mode.startswith("2D:"):
            ax1 = self.fig.add_subplot(121)
            ax2 = self.fig.add_subplot(122)

            sa_v = np.linspace(-14.0, 14.0, 220)
            sa_r = np.radians(sa_v)
            for idx, (f_mult, lbl_suffix) in enumerate(((0.55, "-45%"), (1.0, "Active"), (1.45, "+45%"))):
                fz_cur = float(np.clip(fz_sel * f_mult, 150.0, 2200.0))
                fy_cur, _ = m.evaluate_fy(sa_r, fz_cur, ia_rad)
                lw = 2.6 if f_mult == 1.0 else 1.5
                ls = "-" if f_mult == 1.0 else "--"
                col = CHART_PALETTE[0] if f_mult == 1.0 else CHART_PALETTE[idx + 1]
                ax1.plot(sa_v, fy_cur, color=col, linewidth=lw, linestyle=ls, label=f"Fz={fz_cur:.0f} N ({lbl_suffix})")

            if show_ttc and df_c is not None and not df_c.empty:
                tol_fz = max(120.0, 0.20 * fz_sel)
                mask = (np.abs(df_c["FZ"] - fz_sel) <= tol_fz) & (np.abs(df_c["IA"] - ia_deg) <= 1.2)
                sub = df_c[mask]
                if not sub.empty:
                    fy_sign = -1.0 if float((sub["SA"] * sub["FY"]).mean()) < 0.0 else 1.0
                    ax1.scatter(sub["SA"], fy_sign * sub["FY"], color=c["cyan"], s=12, alpha=0.45, label="TTC (±20% Fz)")

            style_axes_2d(ax1, title=f"Fy(SA) @ Fz={fz_sel:.0f} N, IA={ia_deg:+.1f}°", xlabel="SA [deg]", ylabel="Fy [N]", c=c)
            style_legend(ax1, c=c)

            sl_v = np.linspace(-0.25, 0.25, 220)
            for idx, (f_mult, lbl_suffix) in enumerate(((0.55, "-45%"), (1.0, "Active"), (1.45, "+45%"))):
                fz_cur = float(np.clip(fz_sel * f_mult, 150.0, 2200.0))
                fx_cur, _ = m.evaluate_fx(sl_v, fz_cur, ia_rad)
                lw = 2.6 if f_mult == 1.0 else 1.5
                ls = "-" if f_mult == 1.0 else "--"
                col = CHART_PALETTE[1] if f_mult == 1.0 else CHART_PALETTE[idx + 2]
                ax2.plot(sl_v, fx_cur, color=col, linewidth=lw, linestyle=ls, label=f"Fz={fz_cur:.0f} N")

            if show_ttc and df_d is not None and not df_d.empty:
                tol_fz = max(120.0, 0.20 * fz_sel)
                mask_d = np.abs(df_d["FZ"] - fz_sel) <= tol_fz
                sub_d = df_d[mask_d]
                if not sub_d.empty:
                    ax2.scatter(sub_d["SL"], sub_d["FX"], color=c["warn"], s=12, alpha=0.45, label="TTC Drive/Brake")

            style_axes_2d(ax2, title=f"Fx(SL) @ Fz={fz_sel:.0f} N, IA={ia_deg:+.1f}°", xlabel="Slip Ratio SL [-]", ylabel="Fx [N]", c=c)
            style_legend(ax2, c=c)

        else:
            self._draw_maneuver_simulator(fz_sel, ia_deg)

        self.fig.tight_layout(pad=1.6)
        self.canvas.draw_idle()

    def _draw_maneuver_simulator(self, fz_sel: float, ia_deg: float):
        c = self.c
        m = self.model
        ia_rad = float(np.radians(ia_deg))
        sa_pt = float(self.var_sim_sa.get())
        sl_pt = float(self.var_sim_sl.get())

        ax_gg = self.fig.add_subplot(121)
        ax_sl = self.fig.add_subplot(122)

        sa_sweep = np.linspace(-14.0, 14.0, 121)
        sa_sweep_rad = np.radians(sa_sweep)
        for sl_val in (-0.14, -0.06, 0.0, 0.06, 0.14):
            fx_c = m.evaluate_combined_fx(np.full_like(sa_sweep, sl_val), sa_sweep_rad, fz_sel, ia_rad)
            fy_c = m.evaluate_combined_fy(sa_sweep_rad, sl_val, fz_sel, ia_rad)
            col = c["cyan"] if abs(sl_val) < 1e-6 else c["border_bright"]
            lw = 1.9 if abs(sl_val) < 1e-6 else 1.0
            alpha = 0.9 if abs(sl_val) < 1e-6 else 0.55
            ax_gg.plot(fy_c, fx_c, color=col, linewidth=lw, alpha=alpha, label=f"SL={sl_val:+.2f}" if sl_val in (-0.14, 0.0, 0.14) else None)

        theta = np.linspace(0, 2 * np.pi, 180)
        sa_th = np.radians(13.0 * np.sin(theta))
        sl_th = 0.18 * np.cos(theta)
        fx_env = m.evaluate_combined_fx(sl_th, sa_th, fz_sel, ia_rad)
        fy_env = m.evaluate_combined_fy(sa_th, sl_th, fz_sel, ia_rad)
        ax_gg.plot(fy_env, fx_env, color=c["warn"], linewidth=1.6, linestyle="--", label="Grip Envelope")

        fx_op = float(m.evaluate_combined_fx(np.array([sl_pt]), float(np.radians(sa_pt)), fz_sel, ia_rad)[0])
        fy_op = float(m.evaluate_combined_fy(np.array([np.radians(sa_pt)]), sl_pt, fz_sel, ia_rad)[0])
        ax_gg.plot([0.0, fy_op], [0.0, fx_op], color=c["accent"], linewidth=2.2)
        ax_gg.scatter([fy_op], [fx_op], color=c["accent"], edgecolor=c["text"], s=90, zorder=8, label=f"Operating (SA={sa_pt:+.1f}°, SL={sl_pt:+.2f})")

        ax_gg.set_aspect("equal", adjustable="datalim")
        style_axes_2d(ax_gg, title=f"G-G Friction Circle @ Fz={fz_sel:.0f} N, IA={ia_deg:+.1f}°", xlabel="Fy [N] (Lateral)", ylabel="Fx [N] (Longitudinal)", c=c)
        style_legend(ax_gg, c=c)

        fy_pure, _ = m.evaluate_fy(sa_sweep_rad, fz_sel, ia_rad)
        fy_comb = m.evaluate_combined_fy(sa_sweep_rad, sl_pt, fz_sel, ia_rad)
        ax_sl.plot(sa_sweep, fy_pure, color=c["text_muted"], linestyle="--", linewidth=1.5, label="Pure Fy (SL=0)")
        ax_sl.plot(sa_sweep, fy_comb, color=c["cyan"], linewidth=2.4, label=f"Combined Fy @ SL={sl_pt:+.2f}")
        ax_sl.scatter([sa_pt], [fy_op], color=c["accent"], edgecolor=c["text"], s=80, zorder=8)

        style_axes_2d(ax_sl, title="Lateral Grip Drop Under Longitudinal Slip SL", xlabel="SA [deg]", ylabel="Fy [N]", c=c)
        style_legend(ax_sl, c=c)
