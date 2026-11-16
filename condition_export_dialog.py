"""Configuration & preview dialog for exporting TTC data binned into Fz x IA x Pressure CSV."""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Iterable, Optional

from condition_export import (
    DEFAULT_FZ_CENTERS,
    DEFAULT_FZ_TOL,
    DEFAULT_IA_CENTERS,
    DEFAULT_IA_TOL,
    DEFAULT_P_RANGES,
    PRESSURE_UNITS,
    ConditionResult,
    ConditionSpec,
    center_bins,
    collect_condition_data,
    parse_float,
    parse_number_list,
    parse_range_list,
    range_bins,
    write_csv,
)
from fy_excel_dialog import FyExcelDialog
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


def open_excel_export(parent, runs: Iterable, raw_cache: Optional[dict] = None, colors=None):
    """Open the Fy(SA) -> Excel (.xlsx) configuration and export window."""
    return FyExcelDialog(parent, list(runs), raw_cache=raw_cache, colors=colors)


class ConditionExportDialog(tk.Toplevel):
    """Dialog for Fz x IA x Pressure binning and deduplicated CSV export."""

    def __init__(self, parent, runs: Iterable, raw_cache: Optional[dict] = None, colors=None):
        super().__init__(parent)
        self.runs = list(runs)
        self.cache = raw_cache if raw_cache is not None else {}
        self.c = colors or PALETTE
        self._result: Optional[ConditionResult] = None

        self.title("PUTM Tire Lab — Condition Matrix Export (Fz x IA x P -> CSV)")
        self.geometry("760x660")
        self.minsize(640, 520)
        self.configure(bg=self.c["bg"])
        apply_ttk_theme(self, self.c)

        self.fz_centers = tk.StringVar(value=", ".join(f"{v:g}" for v in DEFAULT_FZ_CENTERS))
        self.fz_tol = tk.StringVar(value=f"{DEFAULT_FZ_TOL:g}")
        self.ia_centers = tk.StringVar(value=", ".join(f"{v:g}" for v in DEFAULT_IA_CENTERS))
        self.ia_tol = tk.StringVar(value=f"{DEFAULT_IA_TOL:g}")
        self.p_ranges = tk.StringVar(value=", ".join(f"{a:g}-{b:g}" for a, b in DEFAULT_P_RANGES))
        self.p_unit = tk.StringVar(value="kPa")
        self.nearest = tk.BooleanVar(value=True)
        self.csv_sep = tk.StringVar(value=", (Comma)")

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
            text="CONDITION MATRIX EXPORT (Fz x IA x P -> CSV)",
            bg=c["bg_header"],
            fg=c["text"],
            font=("Segoe UI Semibold", 11, "bold"),
        ).pack(side="left")

        body = ttk.Frame(self, padding=12)
        body.pack(fill="both", expand=True)

        card_cfg = ModernCard(body, "Bin Definitions (Condition Grid)", accent_color=c["cyan"], colors=c, pad_x=10, pad_y=8)
        card_cfg.pack(fill="x", pady=(0, 10))

        grid = tk.Frame(card_cfg.body, bg=c["panel"])
        grid.pack(fill="x")
        grid.columnconfigure(1, weight=1)

        rows = [
            ("Fz centers [N]:", self.fz_centers, "Tol ± [N]:", self.fz_tol, None),
            ("Camber IA [deg]:", self.ia_centers, "Tol ± [deg]:", self.ia_tol, None),
            ("Pressure P:", self.p_ranges, "Unit:", self.p_unit, list(PRESSURE_UNITS.keys())),
        ]
        for r_idx, (l1, v1, l2, v2, combo_vals) in enumerate(rows):
            tk.Label(grid, text=l1, bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL_BOLD, anchor="w").grid(
                row=r_idx, column=0, sticky="w", pady=4, padx=(0, 8)
            )
            ttk.Entry(grid, textvariable=v1).grid(row=r_idx, column=1, sticky="ew", pady=4, padx=(0, 12))
            tk.Label(grid, text=l2, bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL).grid(
                row=r_idx, column=2, sticky="w", pady=4, padx=(0, 6)
            )
            if combo_vals is not None:
                ttk.Combobox(grid, textvariable=v2, values=combo_vals, state="readonly", width=8).grid(row=r_idx, column=3, sticky="w", pady=4)
            else:
                ttk.Entry(grid, textvariable=v2, width=10).grid(row=r_idx, column=3, sticky="w", pady=4)

        opt_row = tk.Frame(card_cfg.body, bg=c["panel"])
        opt_row.pack(fill="x", pady=(6, 0))
        ttk.Checkbutton(
            opt_row,
            text="Assign each sample only to nearest Fz/IA center (no duplicates)",
            variable=self.nearest,
        ).pack(side="left")
        tk.Label(opt_row, text="Delimiter:", bg=c["panel"], fg=c["text_muted"], font=FONT_SMALL).pack(side="right", padx=(8, 4))
        ttk.Combobox(opt_row, textvariable=self.csv_sep, values=[", (Comma)", "; (Semicolon)"], state="readonly", width=13).pack(side="right")

        btn_bar = tk.Frame(body, bg=c["bg"])
        btn_bar.pack(fill="x", pady=(0, 10))
        ttk.Button(btn_bar, text="Preview Cell Counts", style="Cyan.TButton", command=self._preview).pack(side="left", padx=(0, 8))
        ttk.Button(btn_bar, text="Export Matrix CSV...", style="Accent.TButton", command=self._export).pack(side="left")

        card_prev = ModernCard(body, "Cell Occupancy Preview", colors=c, pad_x=10, pad_y=8)
        card_prev.pack(fill="both", expand=True)

        self.lbl_stats = tk.Label(
            card_prev.body,
            text="Click 'Preview Cell Counts'.",
            bg=c["panel"],
            fg=c["text_secondary"],
            font=FONT_SMALL_BOLD,
            anchor="w",
        )
        self.lbl_stats.pack(fill="x", pady=(0, 6))

        cols = ("fz", "ia", "p", "n")
        self.tree = ttk.Treeview(card_prev.body, columns=cols, show="headings", height=9)
        for key, title, width, anchor in [
            ("fz", "Fz [N]", 120, "center"),
            ("ia", "IA [deg]", 120, "center"),
            ("p", "Pressure P [kPa]", 140, "center"),
            ("n", "Sample Count", 130, "e"),
        ]:
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width, anchor=anchor)
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

    def _build_spec(self) -> ConditionSpec:
        return ConditionSpec(
            fz_bins=center_bins(parse_number_list(self.fz_centers.get(), "Fz centers"), parse_float(self.fz_tol.get(), "Fz tol")),
            ia_bins=center_bins(parse_number_list(self.ia_centers.get(), "IA centers"), parse_float(self.ia_tol.get(), "IA tol")),
            p_bins=range_bins(parse_range_list(self.p_ranges.get(), "Pressure P")),
            p_to_kpa=PRESSURE_UNITS[self.p_unit.get()],
            nearest_only=bool(self.nearest.get()),
        )

    def _preview(self) -> Optional[ConditionResult]:
        if not self.runs:
            self.lbl_stats.config(text="No telemetry files queued in Step 1.", fg=self.c["warn"])
            return None
        try:
            spec = self._build_spec()
            res = collect_condition_data(self.runs, spec, cache=self.cache)
        except Exception as exc:
            messagebox.showerror("Configuration Error", str(exc), parent=self)
            return None

        self._result = res
        self.lbl_stats.config(
            text=(
                f"Files: {res.files_used} | Read: {res.rows_read:,} | "
                f"Export rows: {res.rows_exported:,} (unique: {res.rows_unique:,}) | Outside bins: {res.rows_outside:,}"
            ),
            fg=self.c["cyan"] if res.rows_exported > 0 else self.c["warn"],
        )

        for iid in self.tree.get_children():
            self.tree.delete(iid)
        filled = res.summary[res.summary["n_samples"] > 0]
        for idx, row in enumerate(filled.itertuples(index=False)):
            self.tree.insert("", "end", values=(row[0], row[1], row[2], int(row[3])), tags=("even" if idx % 2 == 0 else "odd",))

        self.txt_notes.config(state="normal")
        self.txt_notes.delete("1.0", "end")
        self.txt_notes.insert("1.0", "\n".join(res.notes) if res.notes else "All cells ready without warnings.")
        self.txt_notes.config(state="disabled")
        return res

    def _export(self):
        res = self._preview()
        if res is None:
            return
        if res.data.empty:
            messagebox.showwarning("Empty Output", "No samples match the configured condition bins.", parent=self)
            return

        path = filedialog.asksaveasfilename(
            parent=self,
            title="Save Condition Matrix CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All Files", "*.*")],
            initialfile="PUTM_Condition_Matrix_Fz_IA_P.csv",
        )
        if not path:
            return
        sep = ";" if self.csv_sep.get().startswith(";") else ","
        try:
            write_csv(path, res, sep=sep)
        except Exception as exc:
            messagebox.showerror("Write Error", str(exc), parent=self)
            return
        messagebox.showinfo("Export Complete", f"Saved {res.rows_exported:,} rows to:\n{path}", parent=self)
