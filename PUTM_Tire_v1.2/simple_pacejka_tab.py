"""
Tool 7 — 2D Magic Formula Curve Sketcher (SimplePacejkaTab).

Carbon Motorsport Telemetry Edition:
- SegmentedToggle (Physical vs Direct B/C/D/E parameterization)
- KPI tiles + structured summary table + physical sanity warnings
- Sync coefficients from Step 2 Curve Fitting
"""

from __future__ import annotations

import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Dict, List, Optional

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from simple_pacejka import (
    LOAD_PARAMS,
    FrictionVsFz,
    LoadSensitiveFy,
    SimpleMagicFormula,
    evaluate,
    from_fit_result,
    from_physical,
    load_family,
    parse_fz_levels,
    project_dict,
    summarize,
    summarize_family,
    summarize_friction,
)
from ui_scroll import ScrollableFrame
from ui_theme import (
    CHART_PALETTE,
    FONT_SMALL,
    FONT_SMALL_BOLD,
    PALETTE,
    KpiTile,
    ModernCard,
    SegmentedToggle,
    StepHeader,
    style_axes_2d,
    style_legend,
    style_treeview_zebra,
)


MODE_PHYSICAL = "Physical (mu, Fz, stiffness)"
MODE_DIRECT = "Direct (B, C, D, E)"
LOAD_CHANNEL = "Fy (Load-Sensitive)"
MU_CHANNEL = "mu(Fz)"
CHANNEL_CHOICES = ["Fy", "Fx", "Mz", LOAD_CHANNEL, MU_CHANNEL]

FIT_KEY = {
    "Fy": "Fy0",
    "Fx": "Fx0",
    "Mz": "Mz0",
    LOAD_CHANNEL: "Fy_load",
    MU_CHANNEL: "mu_vs_fz",
}

DEFAULTS = {
    "Fy": {"mu": "1.65", "fz": "900", "stiffness": "420", "B": "10.5", "C": "1.35", "D": "1485", "E": "-0.45", "Sh": "0.0", "Sv": "0.0", "range": "15"},
    "Fx": {"mu": "1.75", "fz": "900", "stiffness": "22000", "B": "12.0", "C": "1.55", "D": "1575", "E": "-0.35", "Sh": "0.0", "Sv": "0.0", "range": "0.30"},
    "Mz": {"mu": "1.60", "fz": "900", "stiffness": "10", "B": "9.0", "C": "2.1", "D": "55", "E": "-1.2", "Sh": "0.0", "Sv": "0.0", "range": "15"},
    LOAD_CHANNEL: {
        "pCy1": "1.35", "pDy1": "1.65", "pDy2": "-0.18", "pEy1": "-0.45",
        "pKy1": "-16.5", "pKy2": "1.8", "pUy1": "0.0", "pVy1": "0.0",
        "fz0": "1000", "fz_levels": "450, 750, 1050, 1350", "range": "15",
    },
    MU_CHANNEL: {
        "pDy1": "1.65", "pDy2": "-0.18", "pDx1": "1.75", "pDx2": "-0.20",
        "fz0": "1000", "range": "2200",
    },
}


class SimplePacejkaTab(ttk.Frame):
    """Tool 7 — Fast 2D Magic Formula Curve Sketcher."""

    def __init__(
        self,
        parent,
        *,
        colors=None,
        fonts=None,
        set_status: Optional[Callable[[str], None]] = None,
        get_fit_results: Optional[Callable[[], dict]] = None,
        get_runs: Optional[Callable[[], list]] = None,
        show_step_header: bool = True,
    ):
        super().__init__(parent)
        self.c = colors or PALETTE
        self.fonts = fonts or {}
        self._set_status_cb = set_status or (lambda _msg: None)
        self._get_fit_results_cb = get_fit_results or (lambda: {})
        self._get_runs_cb = get_runs or (lambda: [])
        self._show_step_header = show_step_header

        self._saved_models: Dict[str, object] = {}
        self._last_model: Optional[object] = None

        self.var_channel = tk.StringVar(value="Fy")
        self.var_mode = tk.StringVar(value=MODE_PHYSICAL)
        self.var_range = tk.StringVar(value="15")

        self.vars: Dict[str, tk.StringVar] = {}
        for key in ("mu", "fz", "stiffness", "B", "C", "D", "E", "Sh", "Sv", *LOAD_PARAMS, "fz0", "fz_levels", "pDx1", "pDx2"):
            self.vars[key] = tk.StringVar(value="")

        self._build_ui()
        self._load_channel_defaults("Fy")
        self._rebuild_param_fields()
        self.draw()

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
                "TOOL 7",
                "2D Magic Formula Curve Sketcher",
                "Quick interactive calculation of Fy, Fx, Mz, Fy(SA, Fz) family, and mu(Fz) using physical tire targets or B/C/D/E.",
                badge_kind="cyan",
                colors=c,
            ).pack(fill="x", pady=(0, 8))

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)

        left_wrap = tk.Frame(body, bg=c["bg"], width=420)
        left_wrap.pack(side="left", fill="y", padx=(0, 10))
        left_wrap.pack_propagate(False)

        self._left_scroll = ScrollableFrame(left_wrap, bg=c["bg"], padx=2, pady=2)
        self._left_scroll.pack(fill="both", expand=True)
        ctrl = self._left_scroll.inner

        # 1. Channel & Mode
        card_top = ModernCard(
            ctrl,
            "1. Channel & Parameterization Mode",
            accent_color=c["cyan"],
            colors=c,
            pad_x=10,
            pad_y=8,
        )
        card_top.pack(fill="x", pady=(0, 8))

        tk.Label(card_top.body, text="Output channel:", bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL_BOLD, anchor="w").pack(fill="x")
        cb_ch = ttk.Combobox(card_top.body, textvariable=self.var_channel, values=CHANNEL_CHOICES, state="readonly")
        cb_ch.pack(fill="x", pady=(3, 8))
        cb_ch.bind("<<ComboboxSelected>>", lambda _e: self._on_channel_change())

        self._mode_seg = SegmentedToggle(
            card_top.body,
            self.var_mode,
            [
                (MODE_PHYSICAL, "Physical (mu, Fz)"),
                (MODE_DIRECT, "Coefficients B/C/D/E"),
            ],
            command=self._on_mode_change,
            accent_key="accent",
            colors=c,
        )
        self._mode_seg.pack(fill="x", pady=(0, 8))

        rng_row = tk.Frame(card_top.body, bg=c["panel"])
        rng_row.pack(fill="x", pady=(0, 6))
        tk.Label(rng_row, text="Half slip range (or max Fz):", bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL).pack(side="left")
        ttk.Entry(rng_row, textvariable=self.var_range, width=9).pack(side="right")

        ttk.Button(
            card_top.body,
            text="Sync Parameters from Step 2 Fit",
            style="Pill.TButton",
            command=self.import_from_fit,
        ).pack(fill="x")

        # 2. Parameter form
        self.card_params = ModernCard(
            ctrl,
            "2. Curve Parameters",
            accent_color=c["accent"],
            colors=c,
            pad_x=10,
            pad_y=8,
        )
        self.card_params.pack(fill="x", pady=(0, 8))

        self._form_container = tk.Frame(self.card_params.body, bg=c["panel"])
        self._form_container.pack(fill="x", pady=(0, 8))

        ttk.Button(
            self.card_params.body,
            text="Compute & Draw Curve",
            style="Accent.TButton",
            command=self.draw,
        ).pack(fill="x")

        # ------------------------------------------------------------------
        # RIGHT PANEL: KPI Tiles + Plot + Summary Treeview & Warnings
        # ------------------------------------------------------------------
        right = ttk.Frame(body)
        right.pack(side="right", fill="both", expand=True)

        kpi_strip = tk.Frame(right, bg=c["bg"])
        kpi_strip.pack(fill="x", pady=(0, 8))
        for i in range(4):
            kpi_strip.columnconfigure(i, weight=1)

        self.tile_peak = KpiTile(kpi_strip, "PEAK VALUE", "--", accent=c["accent"], colors=c)
        self.tile_peak.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.tile_slip = KpiTile(kpi_strip, "SLIP @ PEAK", "--", accent=c["cyan"], colors=c)
        self.tile_slip.grid(row=0, column=1, sticky="ew", padx=(0, 6))

        self.tile_stiff = KpiTile(kpi_strip, "INITIAL STIFFNESS", "--", accent=c["ok"], colors=c)
        self.tile_stiff.grid(row=0, column=2, sticky="ew", padx=(0, 6))

        self.tile_diag = KpiTile(kpi_strip, "DIAGNOSTICS", "OK", accent=c["purple"], colors=c)
        self.tile_diag.grid(row=0, column=3, sticky="ew")

        plot_card = ModernCard(right, "2D Magic Formula Characteristic", colors=c, pad_x=8, pad_y=6)
        plot_card.pack(fill="both", expand=True, pady=(0, 8))

        self.fig = Figure(figsize=(7.5, 4.0), dpi=100, facecolor=c["panel"])
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_card.body)
        self.canvas.get_tk_widget().configure(bg=c["panel"], highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        sum_card = ModernCard(right, "Summary Points & Physical Checks", colors=c, pad_x=8, pad_y=6)
        sum_card.pack(fill="x")

        cols = ("c1", "c2", "c3", "c4", "c5")
        self.sum_tree = ttk.Treeview(sum_card.body, columns=cols, show="headings", height=4)
        self.sum_tree.pack(fill="x", pady=(0, 4))
        style_treeview_zebra(self.sum_tree, c)

        self.lbl_warnings = tk.Label(
            sum_card.body,
            text="",
            bg=c["panel"],
            fg=c["warn"],
            font=FONT_SMALL,
            anchor="w",
            justify="left",
        )
        self.lbl_warnings.pack(fill="x")

    # ------------------------------------------------------------------
    # FORM MANAGEMENT
    # ------------------------------------------------------------------
    def _load_channel_defaults(self, ch: str):
        defs = DEFAULTS[ch]
        for k, v in defs.items():
            if k == "range":
                self.var_range.set(v)
            elif k in self.vars:
                self.vars[k].set(v)

    def _on_channel_change(self):
        ch = self.var_channel.get()
        if ch in ("Mz", LOAD_CHANNEL, MU_CHANNEL):
            self.var_mode.set(MODE_DIRECT)
            self._mode_seg.refresh()
        self._load_channel_defaults(ch)
        self._rebuild_param_fields()
        self.draw()

    def _on_mode_change(self):
        ch = self.var_channel.get()
        if ch in ("Mz", LOAD_CHANNEL, MU_CHANNEL) and self.var_mode.get() == MODE_PHYSICAL:
            self.var_mode.set(MODE_DIRECT)
            self._mode_seg.refresh()
            messagebox.showinfo("Direct Mode Required", f"Channel '{ch}' uses direct parameters.")
            return
        self._rebuild_param_fields()
        self.draw()

    def _rebuild_param_fields(self):
        c = self.c
        for child in self._form_container.winfo_children():
            child.destroy()

        ch = self.var_channel.get()
        mode = self.var_mode.get()

        if ch in ("Fy", "Fx") and mode == MODE_PHYSICAL:
            stiff_lbl = "Cornering stiffness CFa [N/deg]:" if ch == "Fy" else "Long. stiffness [N / 1.0 SL]:"
            fields = [
                ("Friction coefficient mu:", "mu"),
                ("Vertical load Fz [N]:", "fz"),
                (stiff_lbl, "stiffness"),
                ("Shape factor C (1.3-1.7):", "C"),
                ("Curvature factor E (<= 1.0):", "E"),
                ("Horizontal shift Sh:", "Sh"),
                ("Vertical shift Sv:", "Sv"),
            ]
        elif ch in ("Fy", "Fx", "Mz"):
            fields = [
                ("Stiffness factor B:", "B"),
                ("Shape factor C:", "C"),
                ("Peak factor D:", "D"),
                ("Curvature factor E:", "E"),
                ("Horizontal shift Sh:", "Sh"),
                ("Vertical shift Sv:", "Sv"),
            ]
        elif ch == LOAD_CHANNEL:
            fields = [(name + ":", name) for name in LOAD_PARAMS] + [
                ("Nominal Fz0 [N]:", "fz0"),
                ("Fz levels [N] (list):", "fz_levels"),
            ]
        else:
            fields = [
                ("pDy1 (mu_y @ Fz0):", "pDy1"),
                ("pDy2 (load slope):", "pDy2"),
                ("pDx1 (mu_x @ Fz0):", "pDx1"),
                ("pDx2 (load slope):", "pDx2"),
                ("Nominal Fz0 [N]:", "fz0"),
            ]

        self._form_container.columnconfigure(1, weight=1)
        for row_idx, (lbl, key) in enumerate(fields):
            tk.Label(
                self._form_container,
                text=lbl,
                bg=c["panel"],
                fg=c["text_secondary"],
                font=FONT_SMALL,
                anchor="w",
            ).grid(row=row_idx, column=0, sticky="w", pady=2, padx=(0, 8))
            ttk.Entry(self._form_container, textvariable=self.vars[key]).grid(row=row_idx, column=1, sticky="ew", pady=2)

    def _fval(self, key: str) -> float:
        return float(self.vars[key].get().strip().replace(",", "."))

    def import_from_fit(self):
        fit_results = self._get_fit_results_cb()
        ch = self.var_channel.get()
        fit_key = FIT_KEY.get(ch)
        if not fit_results or fit_key not in fit_results:
            messagebox.showinfo(
                "No Fit Available",
                f"Fit '{fit_key}' has not been run yet in Step 2 (Model Fitting).",
            )
            return
        try:
            model = from_fit_result(fit_results[fit_key])
        except Exception as exc:
            messagebox.showerror("Import Error", str(exc))
            return

        self.var_mode.set(MODE_DIRECT)
        self._mode_seg.refresh()
        self._rebuild_param_fields()

        if isinstance(model, SimpleMagicFormula):
            for k in ("B", "C", "D", "E", "Sh", "Sv"):
                self.vars[k].set(f"{getattr(model, k):.5g}")
        elif isinstance(model, LoadSensitiveFy):
            for k in LOAD_PARAMS:
                self.vars[k].set(f"{getattr(model, k):.5g}")
            self.vars["fz0"].set(f"{model.fz0:.1f}")
        elif isinstance(model, FrictionVsFz):
            for k in ("pDy1", "pDy2", "pDx1", "pDx2", "fz0"):
                self.vars[k].set(f"{getattr(model, k):.5g}")

        self._status(f"Imported '{fit_key}' fit parameters from Step 2.")
        self.draw()

    # ------------------------------------------------------------------
    # EVALUATION & PLOTTING
    # ------------------------------------------------------------------
    def _configure_summary_columns(self, cols: List[tuple]):
        for iid in self.sum_tree.get_children():
            self.sum_tree.delete(iid)
        for col_id, (title, width, anchor) in zip(("c1", "c2", "c3", "c4", "c5"), cols):
            self.sum_tree.heading(col_id, text=title)
            self.sum_tree.column(col_id, width=width, anchor=anchor, stretch=(col_id == "c1"))

    def draw(self):
        c = self.c
        ch = self.var_channel.get()
        mode = self.var_mode.get()

        try:
            half_rng = float(self.var_range.get().strip().replace(",", "."))
        except ValueError:
            half_rng = 15.0

        self.fig.clear()
        self.fig.patch.set_facecolor(c["panel"])
        ax = self.fig.add_subplot(111)
        warnings_list: List[str] = []

        try:
            if ch in ("Fy", "Fx", "Mz"):
                if ch in ("Fy", "Fx") and mode == MODE_PHYSICAL:
                    model = from_physical(
                        ch,
                        mu=self._fval("mu"),
                        fz=self._fval("fz"),
                        stiffness=self._fval("stiffness"),
                        C=self._fval("C"),
                        E=self._fval("E"),
                        Sh=self._fval("Sh"),
                        Sv=self._fval("Sv"),
                    )
                else:
                    model = SimpleMagicFormula(
                        ch,
                        B=self._fval("B"),
                        C=self._fval("C"),
                        D=self._fval("D"),
                        E=self._fval("E"),
                        Sh=self._fval("Sh"),
                        Sv=self._fval("Sv"),
                    )
                self._last_model = model
                self._saved_models[model.label] = model

                xs, ys = evaluate(model, half_range=half_rng)
                info = summarize(model, half_range=half_rng)
                warnings_list = info["warnings"]

                ax.plot(xs, ys, color=CHART_PALETTE[0], linewidth=2.5, label=f"{model.label} (B={model.B:.2f}, C={model.C:.2f}, D={model.D:.0f})")
                xlabel = "SA [deg]" if ch in ("Fy", "Mz") else "Slip Ratio SL [-]"
                ylabel = f"{ch} [{'Nm' if ch == 'Mz' else 'N'}]"
                style_axes_2d(ax, title=f"2D Sketch: {ch} Magic Formula", xlabel=xlabel, ylabel=ylabel, c=c)
                style_legend(ax, c=c)

                self.tile_peak.set_value(f"{info['peak_force']:.1f} {info['force_unit']}")
                self.tile_slip.set_value(f"{info['slip_at_peak']:.2f}")
                self.tile_stiff.set_value(f"{info['initial_stiffness']:.1f} {info['stiffness_unit']}")

                self._configure_summary_columns([
                    ("Channel", 120, "w"),
                    ("B", 90, "e"),
                    ("C", 90, "e"),
                    ("D", 100, "e"),
                    ("E", 90, "e"),
                ])
                self.sum_tree.insert("", "end", values=(model.label, f"{model.B:.4f}", f"{model.C:.4f}", f"{model.D:.2f}", f"{model.E:.4f}"), tags=("even",))

            elif ch == LOAD_CHANNEL:
                model = LoadSensitiveFy(
                    **{k: self._fval(k) for k in LOAD_PARAMS},
                    fz0=self._fval("fz0"),
                )
                levels = parse_fz_levels(self.vars["fz_levels"].get())
                self._last_model = model
                self._saved_models[model.label] = model

                xs, fam = load_family(model, levels, half_range=half_rng)
                info = summarize_family(model, levels, half_range=half_rng)
                warnings_list = info["warnings"]

                for idx, (fz_v, fy_arr) in enumerate(fam.items()):
                    ax.plot(xs, fy_arr, color=CHART_PALETTE[idx % len(CHART_PALETTE)], linewidth=2.2, label=f"Fz = {fz_v:.0f} N")
                style_axes_2d(ax, title="Load-Sensitive Family Fy(SA, Fz)", xlabel="SA [deg]", ylabel="Fy [N]", c=c)
                style_legend(ax, c=c)

                rows = info["rows"]
                best = max(rows, key=lambda r: r["peak_force"])
                self.tile_peak.set_value(f"{best['peak_force']:.0f} N")
                self.tile_slip.set_value(f"{best['slip_at_peak']:.1f}°")
                self.tile_stiff.set_value(f"{best['stiffness_n_per_deg']:.0f} N/°")

                self._configure_summary_columns([
                    ("Fz [N]", 100, "e"),
                    ("Peak Fy [N]", 110, "e"),
                    ("SA @ Peak [°]", 110, "e"),
                    ("Peak mu [-]", 100, "e"),
                    ("CFa [N/deg]", 110, "e"),
                ])
                for i, r in enumerate(rows):
                    self.sum_tree.insert(
                        "",
                        "end",
                        values=(f"{r['fz']:.0f}", f"{r['peak_force']:.1f}", f"{r['slip_at_peak']:.2f}", f"{r['mu_peak']:.3f}", f"{r['stiffness_n_per_deg']:.1f}"),
                        tags=("even" if i % 2 == 0 else "odd",),
                    )

            else:
                model = FrictionVsFz(
                    pDy1=self._fval("pDy1"),
                    pDy2=self._fval("pDy2"),
                    pDx1=self._fval("pDx1"),
                    pDx2=self._fval("pDx2"),
                    fz0=self._fval("fz0"),
                )
                self._last_model = model
                self._saved_models[model.label] = model

                fz_arr = np.linspace(200.0, max(600.0, half_rng), 200)
                mu_y, mu_x, _dy, _dx = model.evaluate(fz_arr)
                info = summarize_friction(model, fz_max=max(600.0, half_rng))
                warnings_list = info["warnings"]

                ax.plot(fz_arr, mu_y, color=CHART_PALETTE[0], linewidth=2.4, label="mu_y (Lateral)")
                ax.plot(fz_arr, mu_x, color=CHART_PALETTE[1], linewidth=2.4, linestyle="--", label="mu_x (Longitudinal)")
                style_axes_2d(ax, title="Friction Load Sensitivity mu(Fz)", xlabel="Vertical Load Fz [N]", ylabel="Friction Coefficient mu [-]", zero_lines=False, c=c)
                style_legend(ax, c=c)

                self.tile_peak.set_value(f"{model.pDy1:.2f} (mu_y0)")
                self.tile_slip.set_value(f"{model.pDx1:.2f} (mu_x0)")
                self.tile_stiff.set_value(f"{model.pDy2:+.3f}")

                self._configure_summary_columns([
                    ("Fz [N]", 100, "e"),
                    ("mu_y [-]", 100, "e"),
                    ("mu_x [-]", 100, "e"),
                    ("Dy [N]", 110, "e"),
                    ("Dx [N]", 110, "e"),
                ])
                for i, r in enumerate(info["rows"]):
                    self.sum_tree.insert(
                        "",
                        "end",
                        values=(f"{r['fz']:.0f}", f"{r['mu_y']:.3f}", f"{r['mu_x']:.3f}", f"{r['Dy']:.0f}", f"{r['Dx']:.0f}"),
                        tags=("even" if i % 2 == 0 else "odd",),
                    )

        except Exception as exc:
            warnings_list = [f"Parameter error: {exc}"]
            style_axes_2d(ax, title="Check parameter inputs on the left", c=c)

        if warnings_list:
            self.tile_diag.set_value(f"{len(warnings_list)} warn", c["warn"])
            self.lbl_warnings.config(text=" | ".join(warnings_list), fg=c["warn"])
        else:
            self.tile_diag.set_value("OK", c["ok"])
            self.lbl_warnings.config(text="All curve parameters are within physical bounds.", fg=c["ok"])

        self.fig.tight_layout(pad=1.5)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # EXPORT HELPERS
    # ------------------------------------------------------------------
    def get_export_payload(self) -> Optional[dict]:
        if not self._saved_models:
            return None
        return project_dict(self._saved_models)

    def save_plot(self, path: Optional[str] = None, dpi: int = 220):
        if not path:
            path = filedialog.asksaveasfilename(
                title="Save 2D Sketcher Plot",
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
                initialfile="PUTM_Sketch2D.png",
            )
        if not path:
            return
        self.fig.savefig(path, dpi=dpi, facecolor=self.fig.get_facecolor())
        self._status(f"Saved plot: {path}")

    def export_json(self, path: Optional[str] = None):
        payload = self.get_export_payload()
        if payload is None:
            messagebox.showwarning("No Model", "Compute and draw a curve first.")
            return
        if not path:
            path = filedialog.asksaveasfilename(
                title="Export 2D Sketcher JSON (Schema v2)",
                defaultextension=".json",
                filetypes=[("JSON", "*.json")],
                initialfile="PUTM_SimplePacejka_v2.json",
            )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        self._status(f"Exported JSON (schema v2): {path}")
