"""Batch Auto-Fit dialog across the full Fz x IA x P operating condition matrix."""

from __future__ import annotations

import csv
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Iterable, List, Optional

import numpy as np

from condition_export import (
    DEFAULT_FZ_CENTERS,
    DEFAULT_FZ_TOL,
    DEFAULT_IA_CENTERS,
    DEFAULT_IA_TOL,
    DEFAULT_P_RANGES,
    PRESSURE_UNITS,
    ConditionSpec,
    center_bins,
    collect_condition_data,
    parse_float,
    parse_number_list,
    parse_range_list,
    range_bins,
)
from fitting import fit_pure_magic
from ui_theme import (
    FONT_SMALL,
    FONT_SMALL_BOLD,
    PALETTE,
    ModernCard,
    apply_ttk_theme,
    load_brand_logo,
    style_treeview_zebra,
)


class MatrixAutoFitDialog(tk.Toplevel):
    """Fit Pacejka Magic Formula (Fy0 or Fx0) across every Fz x IA x Pressure cell."""

    def __init__(
        self,
        parent,
        runs: Iterable,
        raw_cache: Optional[dict] = None,
        on_fit_selected: Optional[Callable[[str, dict], None]] = None,
        colors=None,
    ):
        super().__init__(parent)
        self.runs = list(runs)
        self.cache = raw_cache if raw_cache is not None else {}
        self._on_fit_selected = on_fit_selected
        self.c = colors or PALETTE
        self._rows_data: List[dict] = []

        self.title("PUTM Tire Lab — Matrix Auto-Fit (Batch Fz x IA x P)")
        self.geometry("980x690")
        self.minsize(780, 540)
        self.configure(bg=self.c["bg"])
        apply_ttk_theme(self, self.c)

        self.var_target = tk.StringVar(value="Fy0")
        self.var_min_pts = tk.StringVar(value="20")
        self.fz_centers = tk.StringVar(value=", ".join(f"{v:g}" for v in DEFAULT_FZ_CENTERS))
        self.fz_tol = tk.StringVar(value=f"{DEFAULT_FZ_TOL:g}")
        self.ia_centers = tk.StringVar(value=", ".join(f"{v:g}" for v in DEFAULT_IA_CENTERS))
        self.ia_tol = tk.StringVar(value=f"{DEFAULT_IA_TOL:g}")
        self.p_ranges = tk.StringVar(value=", ".join(f"{a:g}-{b:g}" for a, b in DEFAULT_P_RANGES))
        self.p_unit = tk.StringVar(value="kPa")
        self.nearest = tk.BooleanVar(value=True)

        self._build_ui()

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
            text="MATRIX AUTO-FIT (BATCH Fz x IA x P FITTING)",
            bg=c["bg_header"],
            fg=c["text"],
            font=("Segoe UI Semibold", 11, "bold"),
        ).pack(side="left")

        body = ttk.Frame(self, padding=12)
        body.pack(fill="both", expand=True)

        card_cfg = ModernCard(body, "Grid & Model Parameters", accent_color=c["accent"], colors=c, pad_x=10, pad_y=8)
        card_cfg.pack(fill="x", pady=(0, 10))

        top_row = tk.Frame(card_cfg.body, bg=c["panel"])
        top_row.pack(fill="x", pady=(0, 6))
        tk.Label(top_row, text="Target model:", bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL_BOLD).pack(side="left")
        ttk.Combobox(top_row, textvariable=self.var_target, values=["Fy0", "Fx0"], state="readonly", width=10).pack(side="left", padx=(6, 14))
        tk.Label(top_row, text="Min samples/cell:", bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL_BOLD).pack(side="left")
        ttk.Entry(top_row, textvariable=self.var_min_pts, width=7).pack(side="left", padx=(6, 14))
        ttk.Checkbutton(top_row, text="Nearest bin center only", variable=self.nearest).pack(side="left")

        grid = tk.Frame(card_cfg.body, bg=c["panel"])
        grid.pack(fill="x")
        grid.columnconfigure(1, weight=1)

        for r_idx, (l1, v1, l2, v2, combo_vals) in enumerate([
            ("Fz centers [N]:", self.fz_centers, "Tol ±:", self.fz_tol, None),
            ("IA centers [deg]:", self.ia_centers, "Tol ±:", self.ia_tol, None),
            ("Pressure P:", self.p_ranges, "Unit:", self.p_unit, list(PRESSURE_UNITS.keys())),
        ]):
            tk.Label(grid, text=l1, bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL_BOLD).grid(row=r_idx, column=0, sticky="w", pady=3, padx=(0, 8))
            ttk.Entry(grid, textvariable=v1).grid(row=r_idx, column=1, sticky="ew", pady=3, padx=(0, 12))
            tk.Label(grid, text=l2, bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL).grid(row=r_idx, column=2, sticky="w", pady=3, padx=(0, 6))
            if combo_vals:
                ttk.Combobox(grid, textvariable=v2, values=combo_vals, state="readonly", width=8).grid(row=r_idx, column=3, sticky="w", pady=3)
            else:
                ttk.Entry(grid, textvariable=v2, width=10).grid(row=r_idx, column=3, sticky="w", pady=3)

        btn_bar = tk.Frame(body, bg=c["bg"])
        btn_bar.pack(fill="x", pady=(0, 10))
        ttk.Button(btn_bar, text="Run Matrix Auto-Fit", style="Accent.TButton", command=self.run_matrix_fit).pack(side="left", padx=(0, 8))
        ttk.Button(btn_bar, text="Apply Selected Row -> Step 2 Fit", style="Cyan.TButton", command=self.apply_selected).pack(side="left", padx=(0, 8))
        ttk.Button(btn_bar, text="Export Table CSV", command=self.export_csv).pack(side="left")

        card_res = ModernCard(body, "Fitted Matrix Cells (B, C, D, E, Sh, Sv)", colors=c, pad_x=10, pad_y=8)
        card_res.pack(fill="both", expand=True)

        self.lbl_status = tk.Label(
            card_res.body,
            text="Ready. Click 'Run Matrix Auto-Fit'.",
            bg=c["panel"],
            fg=c["text_secondary"],
            font=FONT_SMALL_BOLD,
            anchor="w",
        )
        self.lbl_status.pack(fill="x", pady=(0, 6))

        cols = ("fz", "ia", "p", "n", "r2", "rmse", "b", "c", "d", "e", "sh", "sv")
        self.tree = ttk.Treeview(card_res.body, columns=cols, show="headings", height=14)
        for key, title, width, anchor in [
            ("fz", "Fz [N]", 72, "center"),
            ("ia", "IA [°]", 60, "center"),
            ("p", "P [kPa]", 74, "center"),
            ("n", "Samples", 62, "e"),
            ("r2", "R^2", 68, "e"),
            ("rmse", "RMSE [N]", 74, "e"),
            ("b", "B", 72, "e"),
            ("c", "C", 64, "e"),
            ("d", "D", 78, "e"),
            ("e", "E", 68, "e"),
            ("sh", "Sh", 72, "e"),
            ("sv", "Sv", 72, "e"),
        ]:
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width, anchor=anchor)
        self.tree.pack(fill="both", expand=True)
        style_treeview_zebra(self.tree, c)

    def _build_spec(self) -> ConditionSpec:
        return ConditionSpec(
            fz_bins=center_bins(parse_number_list(self.fz_centers.get(), "Fz centers"), parse_float(self.fz_tol.get(), "Fz tol")),
            ia_bins=center_bins(parse_number_list(self.ia_centers.get(), "IA centers"), parse_float(self.ia_tol.get(), "IA tol")),
            p_bins=range_bins(parse_range_list(self.p_ranges.get(), "Pressure P")),
            p_to_kpa=PRESSURE_UNITS[self.p_unit.get()],
            nearest_only=bool(self.nearest.get()),
        )

    def run_matrix_fit(self):
        if not self.runs:
            messagebox.showwarning("No Files", "Load telemetry files in Step 1 first.", parent=self)
            return
        try:
            spec = self._build_spec()
            min_pts = int(parse_float(self.var_min_pts.get(), "Min samples"))
        except Exception as exc:
            messagebox.showerror("Invalid Parameters", str(exc), parent=self)
            return

        target = self.var_target.get()
        test_type = "Cornering" if target == "Fy0" else "Drive/Brake"
        filtered_runs = [r for r in self.runs if getattr(r, "test_type", "") == test_type]
        if not filtered_runs:
            filtered_runs = self.runs

        cond = collect_condition_data(filtered_runs, spec, cache=self.cache)
        df = cond.data
        if df.empty:
            self.lbl_status.config(text="No samples found in configured Fz x IA x P bins.", fg=self.c["warn"])
            return

        if "__type__" in df.columns:
            df = df[df["__type__"] == test_type]

        for iid in self.tree.get_children():
            self.tree.delete(iid)
        self._rows_data.clear()

        ok_cnt = 0
        for (fz_b, ia_b, p_b), grp in df.groupby(["Fz_bin_N", "IA_bin_deg", "P_bin_kPa"], dropna=False, sort=False):
            if len(grp) < min_pts:
                continue
            try:
                if target == "Fy0":
                    x = np.radians(grp["SA"].to_numpy(dtype=float))
                    y = -grp["FY"].to_numpy(dtype=float)
                    slip_unit = "rad"
                else:
                    x = grp["SL"].to_numpy(dtype=float)
                    y = grp["FX"].to_numpy(dtype=float)
                    slip_unit = "ratio"

                out = fit_pure_magic(x, y, model_name=target, slip_unit=slip_unit)
                p = out.parameters
                diag = out.diagnostics
                if target == "Fy0":
                    vals = (p["B"], p["C"], p["D"], p["E"], p["Sh"], p["Sv"])
                else:
                    vals = (p["Bx"], p["Cx"], p["Dx"], p["Ex"], p["Shx"], p["Svx"])

                row_idx = len(self._rows_data)
                fit_dict = {
                    "model": target,
                    **{k: float(v) for k, v in p.items()},
                    "slip_unit": slip_unit,
                    "force_unit": "N",
                    "r2": float(diag.get("r2", np.nan)),
                    "rmse": float(diag.get("rmse", np.nan)),
                    "cell": f"Fz={fz_b}, IA={ia_b}, P={p_b}",
                }
                self._rows_data.append(fit_dict)
                self.tree.insert(
                    "",
                    "end",
                    iid=str(row_idx),
                    values=(
                        fz_b,
                        ia_b,
                        p_b,
                        len(grp),
                        f"{diag.get('r2', np.nan):.4f}",
                        f"{diag.get('rmse', np.nan):.1f}",
                        *(f"{v:.4f}" for v in vals[:4]),
                        *(f"{v:.5f}" for v in vals[4:]),
                    ),
                    tags=("even" if row_idx % 2 == 0 else "odd",),
                )
                ok_cnt += 1
            except Exception:
                continue

        self.lbl_status.config(
            text=f"Completed Auto-Fit ({target}): {ok_cnt} fitted cell(s).",
            fg=self.c["ok"] if ok_cnt > 0 else self.c["warn"],
        )

    def apply_selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Select Row", "Select a row from the table first.", parent=self)
            return
        idx = int(sel[0])
        fit_dict = self._rows_data[idx]
        if self._on_fit_selected is not None:
            self._on_fit_selected(self.var_target.get(), fit_dict)
            messagebox.showinfo("Applied", f"Loaded fit parameters from cell:\n{fit_dict['cell']}", parent=self)

    def export_csv(self):
        if not self._rows_data:
            messagebox.showwarning("No Results", "Run Matrix Auto-Fit first.", parent=self)
            return
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Export Fitted Matrix (CSV)",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=f"PUTM_MatrixAutoFit_{self.var_target.get()}.csv",
        )
        if not path:
            return
        keys = list(self._rows_data[0].keys())
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, delimiter=";")
            writer.writeheader()
            writer.writerows(self._rows_data)
        messagebox.showinfo("Export Complete", f"Saved {len(self._rows_data)} fitted cells to:\n{path}", parent=self)
