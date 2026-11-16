"""Configuration & preview dialog for exporting Fy(SA) Excel (.xlsx) engineering reports."""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Iterable, Optional

import numpy as np

from condition_export import (
    DEFAULT_FZ_CENTERS,
    DEFAULT_FZ_TOL,
    DEFAULT_IA_CENTERS,
    DEFAULT_IA_TOL,
    DEFAULT_P_RANGES,
    PRESSURE_UNITS,
    center_bins,
    parse_float,
    parse_number_list,
    parse_range_list,
    range_bins,
)
from fy_excel_export import (
    DEFAULT_SA_MAX,
    DEFAULT_SA_MIN,
    DEFAULT_SA_STEP,
    FyExcelExportResult,
    FyExcelSpec,
    build_fy_excel_data,
    save_fy_excel,
)
from ui_theme import (
    FONT_MONO,
    FONT_SMALL,
    FONT_SMALL_BOLD,
    PALETTE,
    ModernCard,
    apply_ttk_theme,
    load_brand_logo,
    style_treeview_zebra,
)


class FyExcelDialog(tk.Toplevel):
    """Fy(SA) Excel (.xlsx) Report Generator with native scatter charts & KPI sheets."""

    def __init__(self, parent, runs: Iterable, raw_cache: Optional[dict] = None, colors=None):
        super().__init__(parent)
        self.runs = list(runs)
        self.cache = raw_cache if raw_cache is not None else {}
        self.c = colors or PALETTE
        self._result: Optional[FyExcelExportResult] = None

        self.title("PUTM Tire Lab — Fy(SA) Excel Workbook Export (.xlsx)")
        self.geometry("860x690")
        self.minsize(720, 560)
        self.configure(bg=self.c["bg"])
        apply_ttk_theme(self, self.c)

        self.fz_centers = tk.StringVar(value=", ".join(f"{v:g}" for v in DEFAULT_FZ_CENTERS))
        self.fz_tol = tk.StringVar(value=f"{DEFAULT_FZ_TOL:g}")
        self.ia_centers = tk.StringVar(value=", ".join(f"{v:g}" for v in DEFAULT_IA_CENTERS))
        self.ia_tol = tk.StringVar(value=f"{DEFAULT_IA_TOL:g}")
        self.p_ranges = tk.StringVar(value=", ".join(f"{a:g}-{b:g}" for a, b in DEFAULT_P_RANGES))
        self.p_unit = tk.StringVar(value="kPa")
        self.nearest = tk.BooleanVar(value=True)

        self.sa_min = tk.StringVar(value=f"{DEFAULT_SA_MIN:g}")
        self.sa_max = tk.StringVar(value=f"{DEFAULT_SA_MAX:g}")
        self.sa_step = tk.StringVar(value=f"{DEFAULT_SA_STEP:g}")

        self.include_fits = tk.BooleanVar(value=True)
        self.include_binned = tk.BooleanVar(value=True)
        self.include_charts = tk.BooleanVar(value=True)
        self.include_kpi = tk.BooleanVar(value=True)

        self._build_ui()
        if self.runs:
            self._preview()

    def _build_ui(self):
        c = self.c
        hdr = tk.Frame(self, bg=c["bg_header"], height=46, padx=14)
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
            text="EXCEL REPORT BUILDER (.XLSX) — Fy(SA) CURVES, KPI & NATIVE CHARTS",
            bg=c["bg_header"],
            fg=c["text"],
            font=("Segoe UI Semibold", 11, "bold"),
        ).pack(side="left")

        body = ttk.Frame(self, padding=12)
        body.pack(fill="both", expand=True)

        card_cfg = ModernCard(body, "Binning & Slip Angle Grid Configuration", accent_color=c["ok"], colors=c, pad_x=10, pad_y=8)
        card_cfg.pack(fill="x", pady=(0, 10))

        grid = tk.Frame(card_cfg.body, bg=c["panel"])
        grid.pack(fill="x")
        grid.columnconfigure(1, weight=1)

        for r_idx, (l1, v1, l2, v2, combo_vals) in enumerate([
            ("Fz centers [N]:", self.fz_centers, "Tol ± [N]:", self.fz_tol, None),
            ("Camber IA [deg]:", self.ia_centers, "Tol ± [deg]:", self.ia_tol, None),
            ("Pressure P:", self.p_ranges, "Unit:", self.p_unit, list(PRESSURE_UNITS.keys())),
        ]):
            tk.Label(grid, text=l1, bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL_BOLD).grid(row=r_idx, column=0, sticky="w", pady=3, padx=(0, 8))
            ttk.Entry(grid, textvariable=v1).grid(row=r_idx, column=1, sticky="ew", pady=3, padx=(0, 12))
            tk.Label(grid, text=l2, bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL).grid(row=r_idx, column=2, sticky="w", pady=3, padx=(0, 6))
            if combo_vals:
                ttk.Combobox(grid, textvariable=v2, values=combo_vals, state="readonly", width=8).grid(row=r_idx, column=3, sticky="w", pady=3)
            else:
                ttk.Entry(grid, textvariable=v2, width=10).grid(row=r_idx, column=3, sticky="w", pady=3)

        sa_row = tk.Frame(card_cfg.body, bg=c["panel"])
        sa_row.pack(fill="x", pady=(8, 4))
        for lbl, var in (("SA min [deg]:", self.sa_min), ("SA max [deg]:", self.sa_max), ("Step [deg]:", self.sa_step)):
            tk.Label(sa_row, text=lbl, bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL).pack(side="left")
            ttk.Entry(sa_row, textvariable=var, width=7).pack(side="left", padx=(4, 14))
        ttk.Checkbutton(sa_row, text="Nearest Fz/IA center", variable=self.nearest).pack(side="left")

        opt_row = tk.Frame(card_cfg.body, bg=c["panel"])
        opt_row.pack(fill="x", pady=(4, 0))
        ttk.Checkbutton(opt_row, text="Magic Formula Fits (Fy0)", variable=self.include_fits).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(opt_row, text="Binned Experimental Points", variable=self.include_binned).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(opt_row, text="Native Excel Charts", variable=self.include_charts).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(opt_row, text="KPI Summary Sheet", variable=self.include_kpi).pack(side="left")

        btn_bar = tk.Frame(body, bg=c["bg"])
        btn_bar.pack(fill="x", pady=(0, 10))
        ttk.Button(btn_bar, text="Compute & Preview Fy(SA)", style="Cyan.TButton", command=self._preview).pack(side="left", padx=(0, 8))
        ttk.Button(btn_bar, text="Generate Excel (.xlsx)...", style="Accent.TButton", command=self._export).pack(side="left")

        card_prev = ModernCard(body, "Cells & Pacejka Fit Preview", colors=c, pad_x=10, pad_y=8)
        card_prev.pack(fill="both", expand=True)

        self.lbl_stats = tk.Label(
            card_prev.body,
            text="Click 'Compute & Preview Fy(SA)'.",
            bg=c["panel"],
            fg=c["text_secondary"],
            font=FONT_SMALL_BOLD,
            anchor="w",
        )
        self.lbl_stats.pack(fill="x", pady=(0, 6))

        cols = ("fz", "ia", "p", "n", "peak", "cfa", "mu", "status")
        self.tree = ttk.Treeview(card_prev.body, columns=cols, show="headings", height=9)
        for key, title, width, anchor in [
            ("fz", "Fz [N]", 74, "center"),
            ("ia", "IA [°]", 68, "center"),
            ("p", "P [kPa]", 88, "center"),
            ("n", "Samples", 74, "e"),
            ("peak", "Peak Fy [N]", 95, "e"),
            ("cfa", "CFa [N/°]", 90, "e"),
            ("mu", "mu peak", 78, "e"),
            ("status", "Fit Status", 140, "w"),
        ]:
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width, anchor=anchor, stretch=(key == "status"))
        self.tree.pack(fill="both", expand=True, pady=(0, 6))
        style_treeview_zebra(self.tree, c)

        self.txt_notes = tk.Text(
            card_prev.body,
            height=4,
            bg=c["input"],
            fg=c["text_muted"],
            font=FONT_MONO,
            borderwidth=0,
            highlightthickness=0,
        )
        self.txt_notes.pack(fill="x")
        self.txt_notes.config(state="disabled")

    def _build_spec(self) -> FyExcelSpec:
        return FyExcelSpec(
            fz_bins=center_bins(parse_number_list(self.fz_centers.get(), "Fz centers"), parse_float(self.fz_tol.get(), "Fz tol")),
            ia_bins=center_bins(parse_number_list(self.ia_centers.get(), "IA centers"), parse_float(self.ia_tol.get(), "IA tol")),
            p_bins=range_bins(parse_range_list(self.p_ranges.get(), "Pressure P")),
            p_to_kpa=PRESSURE_UNITS[self.p_unit.get()],
            p_unit_label=self.p_unit.get(),
            nearest_only=bool(self.nearest.get()),
            sa_min=parse_float(self.sa_min.get(), "SA min"),
            sa_max=parse_float(self.sa_max.get(), "SA max"),
            sa_step=parse_float(self.sa_step.get(), "SA step"),
            include_fits=bool(self.include_fits.get()),
            include_binned=bool(self.include_binned.get()),
            include_charts=bool(self.include_charts.get()),
            include_kpi=bool(self.include_kpi.get()),
        )

    def _preview(self) -> Optional[FyExcelExportResult]:
        if not self.runs:
            self.lbl_stats.config(text="No telemetry files queued in Step 1.", fg=self.c["warn"])
            return None
        try:
            spec = self._build_spec()
            res = build_fy_excel_data(self.runs, spec, cache=self.cache)
        except Exception as exc:
            messagebox.showerror("Processing Error", str(exc), parent=self)
            return None

        self._result = res
        self.lbl_stats.config(
            text=(
                f"Files: {res.files_used} | Samples: {res.rows_used:,} | "
                f"Cells: {len(res.conditions)} (Fitted: {res.cells_fitted}, Empty: {res.cells_empty})"
            ),
            fg=self.c["ok"] if res.cells_fitted > 0 else self.c["warn"],
        )

        for iid in self.tree.get_children():
            self.tree.delete(iid)
        for idx, cond in enumerate(res.conditions):
            if cond.n_samples == 0:
                continue
            peak_s = f"{cond.fy_peak:.1f}" if np.isfinite(cond.fy_peak) else "--"
            cfa_s = f"{cond.cfa_deg:.1f}" if np.isfinite(cond.cfa_deg) else "--"
            mu_s = f"{cond.mu_peak:.2f}" if np.isfinite(cond.mu_peak) else "--"
            st_s = "OK" if cond.fit_success else (cond.fit_message or "--")
            self.tree.insert(
                "",
                "end",
                values=(cond.fz_bin, cond.ia_bin, cond.p_bin, cond.n_samples, peak_s, cfa_s, mu_s, st_s),
                tags=("even" if idx % 2 == 0 else "odd",),
            )

        self.txt_notes.config(state="normal")
        self.txt_notes.delete("1.0", "end")
        self.txt_notes.insert("1.0", "\n".join(res.notes) if res.notes else "Ready to generate Excel workbook.")
        self.txt_notes.config(state="disabled")
        return res

    def _export(self):
        res = self._preview()
        if res is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Save Fy(SA) Excel Report",
            defaultextension=".xlsx",
            filetypes=[("Excel Workbook", "*.xlsx")],
            initialfile="PUTM_Fy_SA_Report.xlsx",
        )
        if not path:
            return
        try:
            save_fy_excel(path, res)
        except Exception as exc:
            messagebox.showerror("Excel Write Error", str(exc), parent=self)
            return
        messagebox.showinfo("Excel Exported", f"Saved workbook:\n{path}", parent=self)
