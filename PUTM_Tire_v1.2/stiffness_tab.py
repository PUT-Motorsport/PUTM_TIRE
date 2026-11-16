"""
Tool 6 — Lateral Cornering Stiffness Analyzer CFa (dFy / dSA in linear zone).

Carbon Motorsport Telemetry Edition:
- Resilient multi-file loading (.mat, .csv, .txt, .dat, .tsv) & full Step 1 Cornering integration
- Histogram-plateau O(N) auto-detection for Fz x IA x P bins (TTC + continuous track logs)
- Signal conditioning (stationary dwell V~0 & combined longitudinal slip rejection + deg/rad safeguard)
- Interactive table-to-regression bin inspector, B vs A comparison table, numeric heatmap
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from condition_export import (
    CELL_COLUMNS,
    DEFAULT_FZ_CENTERS,
    DEFAULT_IA_CENTERS,
    DEFAULT_P_RANGES,
    NO_DATA,
    PRESSURE_UNITS,
    ConditionSpec,
    center_bins,
    classify_frame,
    parse_float,
    parse_number_list,
    parse_range_list,
    range_bins,
)
from tire_model import RunConfig, TireDataManager
from ui_scroll import ScrollableFrame
from ui_theme import (
    CHART_PALETTE,
    FONT_MONO,
    FONT_SMALL,
    FONT_SMALL_BOLD,
    PALETTE,
    ModernCard,
    StepHeader,
    style_axes_2d,
    style_colorbar,
    style_legend,
    style_treeview_zebra,
)


PLOT_MODES = [
    "CFa(Fz) vs Vertical Load (A vs B)",
    "CFa(IA) vs Camber Angle (A vs B)",
    "CFa(P) vs Inflation Pressure (A vs B)",
    "Heatmap CFa(Fz, IA) for Tire A / B",
    "Linear Regression Fy(SA) in Selected Bin",
]

BIN_BY_XCOL = {
    "Fz_mean_N": "Fz_bin_N",
    "IA_mean_deg": "IA_bin_deg",
    "P_mean_kPa": "P_bin_kPa",
}


def detect_pressure_multiplier(df: Optional[pd.DataFrame], unit_choice: str) -> Tuple[float, str]:
    """Return multiplier converting channel P to kPa and a readable unit summary."""
    if unit_choice in PRESSURE_UNITS:
        return float(PRESSURE_UNITS[unit_choice]), unit_choice
    if df is None or "P" not in df:
        return 1.0, "no P"

    hints = getattr(df, "attrs", {}).get("unit_hints", {}) or {}
    for k, u in hints.items():
        if str(k).strip().upper().startswith("P"):
            ul = str(u).strip().lower()
            if ul == "bar":
                return 100.0, "Header: bar"
            if ul == "psi":
                return 6.894757, "Header: psi"
            if ul == "kpa":
                return 1.0, "Header: kPa"

    vals = pd.to_numeric(df["P"], errors="coerce").dropna().to_numpy(dtype=float)
    vals = np.abs(vals[np.isfinite(vals)])
    if vals.size == 0:
        return 1.0, "no P"
    med = float(np.median(vals))
    if med < 3.5:
        return 100.0, f"Auto: bar (~{med:.2f} bar = {med * 100.0:.0f} kPa)"
    if med < 30.0:
        return 6.894757, f"Auto: psi (~{med:.1f} psi = {med * 6.894757:.1f} kPa)"
    return 1.0, f"Auto: kPa (~{med:.1f} kPa)"


def _cluster_values(values: np.ndarray, threshold: float, round_step: float) -> List[float]:
    """Fast O(N) plateau/cluster detector using histogram density peaks with continuous fallback."""
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return []
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    span = vmax - vmin
    if span <= max(threshold * 0.5, round_step * 0.5):
        center = round(float(np.median(vals)) / round_step) * round_step
        return [float(center)]

    # Build histogram with fine bins (~ threshold / 3) to separate test plateaus from ramp transitions
    bin_w = max(round_step * 0.5, threshold / 3.0)
    n_bins = int(np.clip(np.ceil(span / bin_w), 5, 400))
    counts, edges = np.histogram(vals, bins=n_bins, range=(vmin, vmax))
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Keep populated plateau bins (>= 3% of densest bin, or >= 5 samples)
    peak_thresh = max(3, int(0.035 * int(np.max(counts))))
    dense_mask = counts >= peak_thresh

    levels: List[float] = []
    if np.any(dense_mask):
        # Group adjacent dense bins and compute exact sample-weighted centroid inside [lo, hi]
        weighted: List[Tuple[float, int]] = []
        i = 0
        while i < len(counts):
            if dense_mask[i]:
                j = i
                while j < len(counts) and dense_mask[j]:
                    j += 1
                lo_e, hi_e = float(edges[i]), float(edges[j])
                in_band = vals[(vals >= lo_e) & (vals <= hi_e if j == len(counts) else vals < hi_e)]
                if in_band.size > 0:
                    weighted.append((float(np.mean(in_band)), int(in_band.size)))
                else:
                    weighted.append((0.5 * (lo_e + hi_e), int(np.sum(counts[i:j]))))
                i = j
            else:
                i += 1

        # Greedy merge closest weighted plateaus within threshold
        weighted.sort(key=lambda item: item[0])
        merged_groups: List[List[Tuple[float, int]]] = [[weighted[0]]]
        for val_c, cnt_c in weighted[1:]:
            prev_mean = sum(v * n for v, n in merged_groups[-1]) / sum(n for _, n in merged_groups[-1])
            if abs(val_c - prev_mean) <= threshold:
                merged_groups[-1].append((val_c, cnt_c))
            else:
                merged_groups.append([(val_c, cnt_c)])

        for grp in merged_groups:
            mean_val = sum(v * n for v, n in grp) / sum(n for _, n in grp)
            levels.append(float(round(mean_val / round_step) * round_step))

    levels = sorted(list(dict.fromkeys(levels)))
    if not levels:
        # Continuous track-log fallback: 4 quantile centroids
        qs = np.unique(np.round(np.quantile(vals, [0.15, 0.4, 0.65, 0.9]) / round_step) * round_step)
        levels = [float(x) for x in qs]
    return levels


def _sort_bin_labels(df: pd.DataFrame, bin_col: str, mean_col: str) -> List[str]:
    """Sort bin labels numerically according to channel mean."""
    if df is None or df.empty or bin_col not in df:
        return []
    pairs: Dict[str, float] = {}
    for b_lbl, grp in df.groupby(bin_col, sort=False):
        lbl = str(b_lbl)
        if mean_col in grp and grp[mean_col].notna().any():
            pairs[lbl] = float(grp[mean_col].mean())
        else:
            try:
                pairs[lbl] = float(lbl.split("-")[0])
            except ValueError:
                pairs[lbl] = float("inf")
    return [lbl for lbl, _ in sorted(pairs.items(), key=lambda kv: kv[1])]


def _condition_hygiene_mask(
    df: pd.DataFrame,
    *,
    filter_stationary: bool = True,
    min_speed: float = 3.0,
    filter_long_slip: bool = True,
    max_abs_sl: float = 0.04,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Build boolean mask rejecting stationary dwell (V~0) and Drive/Brake (|SL|>0) samples safely."""
    n = len(df)
    mask = np.ones(n, dtype=bool)
    stats = {"dropped_dwell": 0, "dropped_sl": 0}

    if filter_stationary and "V" in df:
        v = pd.to_numeric(df["V"], errors="coerce").to_numpy(dtype=float)
        moving = np.isfinite(v) & (np.abs(v) >= float(min_speed))
        if np.any(moving):
            keep_v = (~np.isfinite(v)) | (np.abs(v) >= float(min_speed))
            stats["dropped_dwell"] = int(np.count_nonzero(mask & ~keep_v))
            mask &= keep_v

    if filter_long_slip and "SL" in df:
        sl = pd.to_numeric(df["SL"], errors="coerce").to_numpy(dtype=float)
        pure_sl = np.isfinite(sl) & (np.abs(sl) <= float(max_abs_sl))
        if np.any(pure_sl):
            keep_sl = (~np.isfinite(sl)) | (np.abs(sl) <= float(max_abs_sl))
            stats["dropped_sl"] = int(np.count_nonzero(mask & ~keep_sl))
            mask &= keep_sl

    return mask, stats


def compute_linear_stiffness(
    df: pd.DataFrame,
    *,
    sa_max_deg: float = 1.5,
    min_points: int = 6,
    min_sa_span_deg: float = 0.4,
    filter_stationary: bool = True,
    min_speed: float = 3.0,
    filter_long_slip: bool = True,
    max_abs_sl: float = 0.04,
) -> pd.DataFrame:
    """Compute Cornering Stiffness CFa (linear regression Fy = a*SA + b for |SA| <= sa_max_deg)
    across every (Fz_bin_N, IA_bin_deg, P_bin_kPa) condition cell.
    """
    cols = [
        *CELL_COLUMNS,
        "Fz_mean_N",
        "IA_mean_deg",
        "P_mean_kPa",
        "n_points",
        "slope_N_deg",
        "cfa_N_deg",
        "cfa_N_rad",
        "cfa_norm_deg",
        "intercept_N",
        "r2",
        "rmse_N",
    ]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)

    hygiene_mask, _ = _condition_hygiene_mask(
        df,
        filter_stationary=filter_stationary,
        min_speed=min_speed,
        filter_long_slip=filter_long_slip,
        max_abs_sl=max_abs_sl,
    )
    sub_df = df[hygiene_mask & (np.abs(pd.to_numeric(df["SA"], errors="coerce")) <= float(sa_max_deg))].copy()
    if sub_df.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for (fz_bin, ia_bin, p_bin), grp in sub_df.groupby(CELL_COLUMNS, dropna=False, sort=False):
        if len(grp) < int(min_points):
            continue
        x = pd.to_numeric(grp["SA"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(grp["FY"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if len(x) < int(min_points) or float(np.std(x)) < 1e-6 or float(np.ptp(x)) < float(min_sa_span_deg):
            continue

        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        cfa_deg = abs(float(slope))
        cfa_rad = float(np.degrees(cfa_deg))

        fz_vals = pd.to_numeric(grp["FZ"], errors="coerce").to_numpy(dtype=float)
        fz_mean = float(np.nanmean(fz_vals)) if np.isfinite(fz_vals).any() else np.nan
        ia_vals = pd.to_numeric(grp["IA"], errors="coerce").to_numpy(dtype=float) if "IA" in grp else np.array([np.nan])
        ia_mean = float(np.nanmean(ia_vals)) if np.isfinite(ia_vals).any() else np.nan
        p_vals = pd.to_numeric(grp["P_kPa"], errors="coerce").to_numpy(dtype=float) if "P_kPa" in grp else np.array([np.nan])
        p_mean = float(np.nanmean(p_vals)) if np.isfinite(p_vals).any() else np.nan
        cfa_norm = (cfa_deg / fz_mean) if (np.isfinite(fz_mean) and fz_mean > 1.0) else np.nan

        rows.append(
            {
                "Fz_bin_N": str(fz_bin),
                "IA_bin_deg": str(ia_bin),
                "P_bin_kPa": str(p_bin),
                "Fz_mean_N": fz_mean,
                "IA_mean_deg": ia_mean,
                "P_mean_kPa": p_mean,
                "n_points": int(len(x)),
                "slope_N_deg": float(slope),
                "cfa_N_deg": cfa_deg,
                "cfa_N_rad": cfa_rad,
                "cfa_norm_deg": cfa_norm,
                "intercept_N": float(intercept),
                "r2": float(r2),
                "rmse_N": rmse,
            }
        )

    out = pd.DataFrame(rows, columns=cols)
    if not out.empty:
        out = out.sort_values(["Fz_mean_N", "IA_mean_deg", "P_mean_kPa"], na_position="last").reset_index(drop=True)
    return out


def _normalize_per_file_pressure(raw: pd.DataFrame, unit_choice: str) -> Tuple[pd.Series, List[str]]:
    """Compute P_kPa per source file so mixing psi/bar/kPa runs within Tire A or B works accurately."""
    notes: List[str] = []
    if "P" not in raw or not np.isfinite(pd.to_numeric(raw["P"], errors="coerce")).any():
        if "P_kPa" in raw and np.isfinite(pd.to_numeric(raw["P_kPa"], errors="coerce")).any():
            return pd.to_numeric(raw["P_kPa"], errors="coerce"), ["Using existing P_kPa column."]
        return pd.Series(np.nan, index=raw.index, dtype=float), [f"Missing P channel -> assigned '{NO_DATA}'."]

    p_kpa = pd.Series(np.nan, index=raw.index, dtype=float)
    if unit_choice in PRESSURE_UNITS:
        mult = float(PRESSURE_UNITS[unit_choice])
        p_kpa = pd.to_numeric(raw["P"], errors="coerce") * mult
        return p_kpa, [f"Pressure converted using explicit {unit_choice} (x{mult:.4g})."]

    if "__source__" in raw.columns:
        for src, idx in raw.groupby("__source__", sort=False).groups.items():
            sub = raw.loc[idx]
            mult, desc = detect_pressure_multiplier(sub, "Auto")
            p_kpa.loc[idx] = pd.to_numeric(sub["P"], errors="coerce") * mult
            notes.append(f"{src}: {desc}")
    else:
        mult, desc = detect_pressure_multiplier(raw, "Auto")
        p_kpa = pd.to_numeric(raw["P"], errors="coerce") * mult
        notes.append(desc)
    return p_kpa, notes


class StiffnessTab(ttk.Frame):
    """Tool 6 — Linear Cornering Stiffness CFa Analysis & A vs B Comparison."""

    def __init__(
        self,
        parent,
        *,
        colors=None,
        fonts=None,
        set_status: Optional[Callable[[str], None]] = None,
        get_df_cornering: Optional[Callable[[], Optional[pd.DataFrame]]] = None,
        get_runs: Optional[Callable[[], list]] = None,
        show_step_header: bool = True,
    ):
        super().__init__(parent)
        self.c = colors or PALETTE
        self.fonts = fonts or {}
        self._set_status_cb = set_status or (lambda _msg: None)
        self._get_df_c_cb = get_df_cornering
        self._get_runs_cb = get_runs
        self._show_step_header = show_step_header

        self._raw_a: Optional[pd.DataFrame] = None
        self._raw_b: Optional[pd.DataFrame] = None
        self._class_a: Optional[pd.DataFrame] = None
        self._class_b: Optional[pd.DataFrame] = None
        self._res_a: pd.DataFrame = pd.DataFrame()
        self._res_b: pd.DataFrame = pd.DataFrame()
        self._gain_df: pd.DataFrame = pd.DataFrame()
        self._selected_cell: Optional[Tuple[str, str, str]] = None
        self._row_cell_map: Dict[str, Tuple[str, str, str]] = {}

        self.var_path_a = tk.StringVar(value="")
        self.var_path_b = tk.StringVar(value="")
        self.var_name_a = tk.StringVar(value="Tire A")
        self.var_name_b = tk.StringVar(value="Tire B")

        self.var_sa_max = tk.StringVar(value="1.5")
        self.var_min_pts = tk.StringVar(value="6")
        self.var_filter_hygiene = tk.BooleanVar(value=True)
        self.var_sa_in_rad = tk.BooleanVar(value=False)
        self.var_lock_bins = tk.BooleanVar(value=False)

        self.var_fz_centers = tk.StringVar(value=", ".join(f"{v:g}" for v in DEFAULT_FZ_CENTERS))
        self.var_fz_tol = tk.StringVar(value="200")
        self.var_ia_centers = tk.StringVar(value=", ".join(f"{v:g}" for v in DEFAULT_IA_CENTERS))
        self.var_ia_tol = tk.StringVar(value="0.5")
        self.var_p_ranges = tk.StringVar(value=", ".join(f"{a:g}-{b:g}" for a, b in DEFAULT_P_RANGES))
        self.var_p_unit = tk.StringVar(value="Auto")
        self.var_nearest = tk.BooleanVar(value=True)

        self.var_plot_mode = tk.StringVar(value=PLOT_MODES[0])
        self.var_sel_fz = tk.StringVar(value="All")
        self.var_sel_ia = tk.StringVar(value="All")
        self.var_sel_p = tk.StringVar(value="All")
        self.var_heatmap_target = tk.StringVar(value="A")

        self._build_ui()

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
                "TOOL 6",
                "Cornering Stiffness Analyzer (CFa)",
                "Linear Fy(SA) slope extraction inside the small-slip zone with dwell/SL filtering, plateau auto-binning, and A vs B delta.",
                badge_kind="ok",
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

        # 1. Input datasets
        card_files = ModernCard(
            ctrl,
            "1. Cornering Telemetry Sources (Tire A & B)",
            accent_color=c["cyan"],
            colors=c,
            pad_x=10,
            pad_y=8,
        )
        card_files.pack(fill="x", pady=(0, 8))

        if self._get_df_c_cb is not None or self._get_runs_cb is not None:
            ttk.Button(
                card_files.body,
                text="Pull Step 1 Cornering Runs into Tire A",
                style="Cyan.TButton",
                command=self.use_app_cornering_as_a,
            ).pack(fill="x", pady=(0, 8))

        self._build_tire_picker(
            card_files.body, "Tire A (Base):", self.var_name_a, self.var_path_a,
            lambda: self._pick_mat("A"), lambda: self._clear_tire("A"),
        )
        self._build_tire_picker(
            card_files.body, "Tire B (Comp):", self.var_name_b, self.var_path_b,
            lambda: self._pick_mat("B"), lambda: self._clear_tire("B"),
        )

        # 2. Regression & Bins
        card_bins = ModernCard(
            ctrl,
            "2. Linear Window & Condition Bins",
            accent_color=c["accent"],
            colors=c,
            pad_x=10,
            pad_y=8,
        )
        card_bins.pack(fill="x", pady=(0, 8))

        row_lin = tk.Frame(card_bins.body, bg=c["panel"])
        row_lin.pack(fill="x", pady=(0, 4))
        tk.Label(row_lin, text="|SA| max [deg]:", bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL_BOLD).pack(side="left")
        ttk.Entry(row_lin, textvariable=self.var_sa_max, width=6).pack(side="left", padx=(4, 10))
        tk.Label(row_lin, text="Min pts:", bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL_BOLD).pack(side="left")
        ttk.Entry(row_lin, textvariable=self.var_min_pts, width=6).pack(side="left", padx=(4, 0))

        sa_pills = tk.Frame(card_bins.body, bg=c["panel"])
        sa_pills.pack(fill="x", pady=(0, 6))
        tk.Label(sa_pills, text="Quick |SA|:", bg=c["panel"], fg=c["text_muted"], font=FONT_SMALL).pack(side="left", padx=(0, 6))
        for val in ("1.0", "1.5", "2.0", "3.0"):
            ttk.Button(
                sa_pills,
                text=f"±{val}°",
                style="Pill.TButton",
                command=lambda v=val: self._set_sa_max(v),
            ).pack(side="left", padx=(0, 4))

        ttk.Checkbutton(
            card_bins.body,
            text="Reject stationary dwell (V~0) & combined slip (|SL|>0.04)",
            variable=self.var_filter_hygiene,
            command=lambda: self.compute_and_refresh(allow_auto_bins=False) if self._raw_a is not None else None,
        ).pack(anchor="w", pady=(0, 2))
        ttk.Checkbutton(
            card_bins.body,
            text="Input SA column is in radians (convert -> deg)",
            variable=self.var_sa_in_rad,
            command=lambda: self.compute_and_refresh(allow_auto_bins=False) if self._raw_a is not None else None,
        ).pack(anchor="w", pady=(0, 6))

        self._entry_row(card_bins.body, "Fz centers [N]:", self.var_fz_centers, "Tol ±:", self.var_fz_tol)
        self._entry_row(card_bins.body, "IA centers [°]:", self.var_ia_centers, "Tol ±:", self.var_ia_tol)
        self._entry_row(
            card_bins.body, "P ranges:", self.var_p_ranges, "Unit:",
            self.var_p_unit, is_combo=["Auto", *PRESSURE_UNITS.keys()],
        )

        opt_row = tk.Frame(card_bins.body, bg=c["panel"])
        opt_row.pack(fill="x", pady=(4, 6))
        ttk.Checkbutton(
            opt_row,
            text="Nearest Fz/IA bin",
            variable=self.var_nearest,
        ).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(
            opt_row,
            text="Lock manual bins",
            variable=self.var_lock_bins,
        ).pack(side="left")

        btn_calc_row = tk.Frame(card_bins.body, bg=c["panel"])
        btn_calc_row.pack(fill="x")
        ttk.Button(
            btn_calc_row,
            text="Auto-Detect Bins",
            style="Pill.TButton",
            command=self.auto_detect_bins_and_refresh,
        ).pack(side="left", padx=(0, 6))
        ttk.Button(
            btn_calc_row,
            text="Compute CFa & Compare A vs B",
            style="Accent.TButton",
            command=lambda: self.compute_and_refresh(allow_auto_bins=False),
        ).pack(side="left", fill="x", expand=True)

        # 3. View filters (Exports moved cleanly to global Header Export button)
        card_view = ModernCard(
            ctrl,
            "3. Chart Mode & Condition Filters",
            colors=c,
            pad_x=10,
            pad_y=8,
        )
        card_view.pack(fill="x", pady=(0, 8))

        tk.Label(card_view.body, text="Plot mode:", bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL_BOLD, anchor="w").pack(fill="x")
        cb_mode = ttk.Combobox(card_view.body, textvariable=self.var_plot_mode, values=PLOT_MODES, state="readonly")
        cb_mode.pack(fill="x", pady=(3, 6))
        cb_mode.bind("<<ComboboxSelected>>", lambda _e: self.draw())

        flt_grid = tk.Frame(card_view.body, bg=c["panel"])
        flt_grid.pack(fill="x")
        flt_grid.columnconfigure(1, weight=1)
        flt_grid.columnconfigure(3, weight=1)

        tk.Label(flt_grid, text="Fz:", bg=c["panel"], fg=c["text_muted"], font=FONT_SMALL).grid(row=0, column=0, sticky="w")
        self.cb_fz = ttk.Combobox(flt_grid, textvariable=self.var_sel_fz, state="readonly", width=11)
        self.cb_fz.grid(row=0, column=1, sticky="ew", padx=(4, 8), pady=2)
        self.cb_fz.bind("<<ComboboxSelected>>", lambda _e: self._on_filter_change())

        tk.Label(flt_grid, text="IA:", bg=c["panel"], fg=c["text_muted"], font=FONT_SMALL).grid(row=0, column=2, sticky="w")
        self.cb_ia = ttk.Combobox(flt_grid, textvariable=self.var_sel_ia, state="readonly", width=11)
        self.cb_ia.grid(row=0, column=3, sticky="ew", padx=(4, 0), pady=2)
        self.cb_ia.bind("<<ComboboxSelected>>", lambda _e: self._on_filter_change())

        tk.Label(flt_grid, text="P:", bg=c["panel"], fg=c["text_muted"], font=FONT_SMALL).grid(row=1, column=0, sticky="w")
        self.cb_p = ttk.Combobox(flt_grid, textvariable=self.var_sel_p, state="readonly", width=11)
        self.cb_p.grid(row=1, column=1, sticky="ew", padx=(4, 8), pady=2)
        self.cb_p.bind("<<ComboboxSelected>>", lambda _e: self._on_filter_change())

        tk.Label(flt_grid, text="Map:", bg=c["panel"], fg=c["text_muted"], font=FONT_SMALL).grid(row=1, column=2, sticky="w")
        cb_hm = ttk.Combobox(flt_grid, textvariable=self.var_heatmap_target, values=["A", "B"], state="readonly", width=6)
        cb_hm.grid(row=1, column=3, sticky="w", padx=(4, 0), pady=2)
        cb_hm.bind("<<ComboboxSelected>>", lambda _e: self.draw())

        # ------------------------------------------------------------------
        # RIGHT PANEL: Chart + Notebook Tables
        # ------------------------------------------------------------------
        right = ttk.Frame(body)
        right.pack(side="right", fill="both", expand=True)

        plot_card = ModernCard(right, "Cornering Stiffness CFa Visualization", colors=c, pad_x=8, pad_y=6)
        plot_card.pack(fill="both", expand=True, pady=(0, 8))

        self.fig = Figure(figsize=(7.5, 4.0), dpi=100, facecolor=c["panel"])
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_card.body)
        self.canvas.get_tk_widget().configure(bg=c["panel"], highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        tbl_card = ModernCard(
            right,
            "Regression Results & Tire Comparison (click row to inspect fit)",
            colors=c,
            pad_x=8,
            pad_y=6,
        )
        tbl_card.pack(fill="x")

        self.nb_tables = ttk.Notebook(tbl_card.body)
        self.nb_tables.pack(fill="both", expand=True)

        # Tab 1: Results Treeview
        tab_res = ttk.Frame(self.nb_tables)
        self.nb_tables.add(tab_res, text="CFa Results")
        cols_r = ("tire", "fz", "ia", "p", "cfa_deg", "cfa_rad", "cfa_norm", "r2", "rmse", "n")
        self.tree_res = ttk.Treeview(tab_res, columns=cols_r, show="headings", height=5)
        for key, title, width, anchor in [
            ("tire", "Tire", 112, "w"),
            ("fz", "Fz [N]", 70, "center"),
            ("ia", "IA [deg]", 68, "center"),
            ("p", "P [kPa]", 78, "center"),
            ("cfa_deg", "CFa [N/deg]", 88, "e"),
            ("cfa_rad", "CFa [kN/rad]", 88, "e"),
            ("cfa_norm", "CFa/Fz [1/°]", 82, "e"),
            ("r2", "R^2", 68, "e"),
            ("rmse", "RMSE [N]", 76, "e"),
            ("n", "Samples", 66, "e"),
        ]:
            self.tree_res.heading(key, text=title)
            self.tree_res.column(key, width=width, anchor=anchor, stretch=(key == "tire"))
        self.tree_res.pack(fill="both", expand=True)
        self.tree_res.bind("<<TreeviewSelect>>", self._on_table_select)
        self.tree_res.bind("<Double-1>", lambda _e: self._open_selected_regression())
        style_treeview_zebra(self.tree_res, c)

        # Tab 2: Gain B vs A Treeview
        tab_gain = ttk.Frame(self.nb_tables)
        self.nb_tables.add(tab_gain, text="Gain B vs A [%]")
        cols_g = ("fz", "ia", "p", "cfa_a", "cfa_b", "delta", "pct")
        self.tree_gain = ttk.Treeview(tab_gain, columns=cols_g, show="headings", height=5)
        for key, title, width, anchor in [
            ("fz", "Fz [N]", 85, "center"),
            ("ia", "IA [deg]", 80, "center"),
            ("p", "P [kPa]", 95, "center"),
            ("cfa_a", "CFa A [N/deg]", 105, "e"),
            ("cfa_b", "CFa B [N/deg]", 105, "e"),
            ("delta", "Delta [N/deg]", 105, "e"),
            ("pct", "Gain B vs A", 95, "e"),
        ]:
            self.tree_gain.heading(key, text=title)
            self.tree_gain.column(key, width=width, anchor=anchor, stretch=False)
        self.tree_gain.pack(fill="both", expand=True)
        style_treeview_zebra(self.tree_gain, c)

        # Tab 3: Diagnostics log
        tab_log = ttk.Frame(self.nb_tables)
        self.nb_tables.add(tab_log, text="Diagnostics & Notes")
        self.txt_notes = tk.Text(
            tab_log,
            height=5,
            bg=c["input"],
            fg=c["text_secondary"],
            font=FONT_MONO,
            borderwidth=0,
            highlightthickness=0,
            wrap="word",
        )
        self.txt_notes.pack(fill="both", expand=True, padx=4, pady=4)
        self.txt_notes.insert("1.0", "Load .MAT / .CSV / .TXT files for Tire A (and optionally B), or pull Cornering data from Step 1.")
        self.txt_notes.config(state="disabled")

        self.draw()

    def _build_tire_picker(self, parent, label: str, var_name: tk.StringVar, var_path: tk.StringVar, cmd_pick, cmd_clear):
        c = self.c
        box = tk.Frame(parent, bg=c["panel_alt"], highlightthickness=1, highlightbackground=c["border"], padx=8, pady=6)
        box.pack(fill="x", pady=(0, 6))

        top = tk.Frame(box, bg=c["panel_alt"])
        top.pack(fill="x")
        tk.Label(top, text=label, bg=c["panel_alt"], fg=c["text_secondary"], font=FONT_SMALL_BOLD).pack(side="left")
        ttk.Entry(top, textvariable=var_name, width=13).pack(side="left", padx=(6, 6), fill="x", expand=True)
        ttk.Button(top, text="Browse...", style="Pill.TButton", command=cmd_pick).pack(side="left", padx=(0, 4))
        ttk.Button(top, text="✕", style="Pill.TButton", width=3, command=cmd_clear).pack(side="right")

        tk.Label(box, textvariable=var_path, bg=c["panel_alt"], fg=c["text_muted"], font=FONT_SMALL, anchor="w").pack(fill="x", pady=(3, 0))

    def _entry_row(self, parent, l1, v1, l2, v2, is_combo=None):
        c = self.c
        row = tk.Frame(parent, bg=c["panel"])
        row.pack(fill="x", pady=2)
        tk.Label(row, text=l1, bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL, width=13, anchor="w").pack(side="left")
        ttk.Entry(row, textvariable=v1).pack(side="left", fill="x", expand=True, padx=(0, 6))
        tk.Label(row, text=l2, bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL).pack(side="left")
        if is_combo:
            ttk.Combobox(row, textvariable=v2, values=is_combo, state="readonly", width=6).pack(side="left", padx=(4, 0))
        else:
            ttk.Entry(row, textvariable=v2, width=6).pack(side="left", padx=(4, 0))

    def _set_sa_max(self, value: str):
        self.var_sa_max.set(value)
        if self._raw_a is not None or self._raw_b is not None:
            self.compute_and_refresh(allow_auto_bins=False)

    def _on_filter_change(self):
        self._selected_cell = None
        self.draw()

    def _on_table_select(self, _event=None):
        sel = self.tree_res.selection()
        if not sel:
            return
        cell = self._row_cell_map.get(sel[0])
        if cell is not None:
            self._selected_cell = cell
            if self.var_plot_mode.get() == PLOT_MODES[4]:
                self.draw()

    def _open_selected_regression(self):
        sel = self.tree_res.selection()
        if sel and sel[0] in self._row_cell_map:
            self._selected_cell = self._row_cell_map[sel[0]]
        self.var_plot_mode.set(PLOT_MODES[4])
        self.draw()

    # ------------------------------------------------------------------
    # DATA & COMPUTE
    # ------------------------------------------------------------------
    def _clear_tire(self, target: str):
        if target == "A":
            self._raw_a = None
            self._class_a = None
            self._res_a = pd.DataFrame()
            self.var_path_a.set("")
            self.var_name_a.set("Tire A")
        else:
            self._raw_b = None
            self._class_b = None
            self._res_b = pd.DataFrame()
            self.var_path_b.set("")
            self.var_name_b.set("Tire B")
        self._selected_cell = None
        if self._raw_a is not None or self._raw_b is not None:
            self.compute_and_refresh(allow_auto_bins=not self.var_lock_bins.get())
        else:
            self._gain_df = pd.DataFrame()
            self._populate_tables()
            self.draw()

    def load_files(self, target: str, paths: Iterable[str], *, show_errors: bool = True) -> Tuple[int, List[str]]:
        """Load one or more .mat/.csv/.txt/.dat files into Tire A or Tire B."""
        frames: List[pd.DataFrame] = []
        loaded_names: List[str] = []
        errors: List[str] = []

        for path in paths:
            try:
                raw = TireDataManager.load_unfiltered(RunConfig(filepath=path, test_type="Cornering"))
                raw["__source__"] = os.path.basename(path)
                p_kpa, _p_notes = _normalize_per_file_pressure(raw, self.var_p_unit.get())
                raw["P_kPa"] = p_kpa
                frames.append(raw)
                loaded_names.append(os.path.basename(path))
            except Exception as exc:
                errors.append(f"{os.path.basename(path)}: {exc}")

        if not frames:
            if show_errors and errors:
                messagebox.showerror("File Import Error", "Could not load any telemetry file:\n" + "\n".join(errors[:5]))
            return 0, errors

        df = pd.concat(frames, ignore_index=True)
        first_base = os.path.splitext(loaded_names[0])[0]
        summary_path = (
            f"{loaded_names[0]} ({len(df):,} pts)"
            if len(loaded_names) == 1
            else f"{loaded_names[0]} (+{len(loaded_names) - 1} files, {len(df):,} pts)"
        )

        if target == "A":
            self._raw_a = df
            self.var_path_a.set(summary_path)
            self.var_name_a.set(first_base[:20])
        else:
            self._raw_b = df
            self.var_path_b.set(summary_path)
            self.var_name_b.set(first_base[:20])

        if show_errors and errors:
            messagebox.showwarning("Partial Import", "Some files could not be parsed:\n" + "\n".join(errors[:5]))

        self._status(f"Tire {target}: loaded {len(loaded_names)} file(s), {len(df):,} samples.")
        self.compute_and_refresh(allow_auto_bins=not self.var_lock_bins.get(), extra_notes=errors)
        return len(loaded_names), errors

    def use_app_cornering_as_a(self):
        frames: List[pd.DataFrame] = []
        used_files: List[str] = []
        load_errors: List[str] = []

        if self._get_runs_cb is not None:
            seen_paths = set()
            for r in self._get_runs_cb():
                if getattr(r, "test_type", "") != "Cornering":
                    continue
                ap = os.path.abspath(r.filepath)
                if ap in seen_paths or not os.path.exists(ap):
                    continue
                seen_paths.add(ap)
                try:
                    raw = TireDataManager.load_unfiltered(RunConfig(filepath=ap, test_type="Cornering"))
                    raw["__source__"] = os.path.basename(ap)
                    p_kpa, _ = _normalize_per_file_pressure(raw, self.var_p_unit.get())
                    raw["P_kPa"] = p_kpa
                    frames.append(raw)
                    used_files.append(os.path.basename(ap))
                except Exception as exc:
                    load_errors.append(f"{os.path.basename(ap)}: {exc}")

        if frames:
            df = pd.concat(frames, ignore_index=True)
            src_label = f"Step 1 Raw ({len(used_files)} file(s), {len(df):,} pts)"
        elif self._get_df_c_cb is not None:
            df_c = self._get_df_c_cb()
            if df_c is None or df_c.empty:
                msg = "No Cornering data or FSAE Demo loaded in Step 1."
                if load_errors:
                    msg += "\n\nRaw file read errors:\n" + "\n".join(load_errors[:4])
                messagebox.showwarning("No Data", msg)
                return
            df = df_c.copy()
            src_label = f"Step 1 Filtered Cornering ({len(df):,} pts)"
        else:
            return

        self._raw_a = df
        self.var_path_a.set(src_label)
        self.var_name_a.set("TTC Step 1")
        self.compute_and_refresh(allow_auto_bins=not self.var_lock_bins.get(), extra_notes=load_errors)

    def _pick_mat(self, target: str):
        paths = filedialog.askopenfilenames(
            title=f"Select Cornering Files (.MAT / .CSV / .TXT / .DAT) for Tire {target}",
            filetypes=[
                ("Tire Telemetry (*.mat;*.csv;*.txt;*.dat;*.tsv)", "*.mat *.csv *.txt *.dat *.tsv"),
                ("MATLAB TTC (*.mat)", "*.mat"),
                ("CSV / ASCII (*.csv;*.txt;*.dat)", "*.csv *.txt *.dat *.tsv"),
                ("All Files", "*.*"),
            ],
        )
        if paths:
            self.load_files(target, paths, show_errors=True)

    def _auto_detect_bins_from_raw(self) -> List[str]:
        """Detect Fz, IA, and P test plateaus across all currently loaded datasets."""
        raws = [d for d in (self._raw_a, self._raw_b) if d is not None and not d.empty]
        if not raws:
            return []
        cat = pd.concat(raws, ignore_index=True)
        unit_choice = self.var_p_unit.get()
        detected_notes: List[str] = []

        # Filter stationary dwell / combined slip before detecting test plateaus so ramp/park noise doesn't pollute bins
        mask_h, _ = _condition_hygiene_mask(cat, filter_stationary=bool(self.var_filter_hygiene.get()), filter_long_slip=bool(self.var_filter_hygiene.get()))
        cat_clean = cat[mask_h] if np.count_nonzero(mask_h) >= 6 else cat

        # Fz plateaus
        fz_vals = np.abs(pd.to_numeric(cat_clean["FZ"], errors="coerce").dropna().to_numpy(dtype=float))
        fz_levels = _cluster_values(fz_vals, threshold=110.0, round_step=50.0)
        if fz_levels:
            fz_tol = 200.0
            if len(fz_levels) > 1:
                gaps = [b - a for a, b in zip(fz_levels, fz_levels[1:]) if (b - a) > 10.0]
                if gaps:
                    fz_tol = max(75.0, min(250.0, 0.45 * min(gaps)))
            self.var_fz_centers.set(", ".join(f"{v:g}" for v in fz_levels))
            self.var_fz_tol.set(f"{fz_tol:g}")
            detected_notes.append(f"Fz={fz_levels} (±{fz_tol:g} N)")

        # IA plateaus (supports both negative -1..-4 and positive +2/+4 camber conventions)
        if "IA" in cat_clean and np.isfinite(pd.to_numeric(cat_clean["IA"], errors="coerce")).any():
            ia_vals = pd.to_numeric(cat_clean["IA"], errors="coerce").dropna().to_numpy(dtype=float)
            ia_levels = _cluster_values(ia_vals, threshold=0.45, round_step=0.5)
            if ia_levels:
                ia_tol = 0.5
                if len(ia_levels) > 1:
                    ia_gaps = [b - a for a, b in zip(ia_levels, ia_levels[1:]) if (b - a) > 0.1]
                    if ia_gaps:
                        ia_tol = max(0.25, min(0.6, 0.45 * min(ia_gaps)))
                self.var_ia_centers.set(", ".join(f"{v:g}" for v in ia_levels))
                self.var_ia_tol.set(f"{ia_tol:g}")
                detected_notes.append(f"IA={ia_levels} deg")

        # P plateaus in kPa
        p_kpa_chunks = []
        for d in raws:
            if "P_kPa" in d and np.isfinite(pd.to_numeric(d["P_kPa"], errors="coerce")).any():
                p_kpa_chunks.append(pd.to_numeric(d["P_kPa"], errors="coerce").dropna().to_numpy(dtype=float))
            elif "P" in d and np.isfinite(pd.to_numeric(d["P"], errors="coerce")).any():
                mult, _ = detect_pressure_multiplier(d, unit_choice)
                p_kpa_chunks.append(pd.to_numeric(d["P"], errors="coerce").dropna().to_numpy(dtype=float) * mult)

        if p_kpa_chunks:
            p_all = np.concatenate(p_kpa_chunks)
            p_all = p_all[np.isfinite(p_all)]
            p_levels = _cluster_values(p_all, threshold=5.0, round_step=1.0)
            if p_levels:
                ranges = []
                for c_val in p_levels:
                    lo = max(0.0, float(c_val) - 6.0)
                    hi = max(lo + 2.0, float(c_val) + 6.0)
                    ranges.append((lo, hi))
                merged: List[Tuple[float, float]] = []
                for lo, hi in sorted(ranges):
                    if merged and lo <= merged[-1][1] + 1.0:
                        merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
                    else:
                        merged.append((lo, hi))
                self.var_p_ranges.set(", ".join(f"{a:g}-{b:g}" for a, b in merged))
                detected_notes.append("P=" + ", ".join(f"{a:g}-{b:g} kPa" for a, b in merged))

        return detected_notes

    def auto_detect_bins_and_refresh(self):
        if self._raw_a is None and self._raw_b is None:
            messagebox.showwarning("No Data", "Load Tire A telemetry (.mat / .csv / Step 1) first.")
            return
        self._auto_detect_bins_from_raw()
        self.compute_and_refresh(allow_auto_bins=False)

    def _build_spec(self) -> Tuple[ConditionSpec, float, int]:
        fz_c = parse_number_list(self.var_fz_centers.get(), "Fz centers")
        fz_t = parse_float(self.var_fz_tol.get(), "Fz tol")
        ia_c = parse_number_list(self.var_ia_centers.get(), "IA centers")
        ia_t = parse_float(self.var_ia_tol.get(), "IA tol")
        p_r = parse_range_list(self.var_p_ranges.get(), "P ranges")
        sa_max = parse_float(self.var_sa_max.get(), "SA max")
        min_pts = max(2, int(parse_float(self.var_min_pts.get(), "Min pts")))
        spec = ConditionSpec(
            fz_bins=center_bins(fz_c, fz_t),
            ia_bins=center_bins(ia_c, ia_t),
            p_bins=range_bins(p_r),
            p_to_kpa=1.0,
            nearest_only=bool(self.var_nearest.get()),
        )
        return spec, sa_max, min_pts

    def compute_and_refresh(self, allow_auto_bins: bool = True, extra_notes: Optional[List[str]] = None):
        if self._raw_a is None and self._raw_b is None:
            messagebox.showwarning("No Data", "Select .MAT/.CSV files for Tire A or pull data from Step 1.")
            return

        notes: List[str] = list(extra_notes or [])
        if allow_auto_bins and not self.var_lock_bins.get():
            adapted = self._auto_detect_bins_from_raw()
            if adapted:
                notes.append("Auto-detected bins: " + "; ".join(adapted))

        try:
            spec, sa_max, min_pts = self._build_spec()
        except Exception as exc:
            messagebox.showerror("Invalid Parameters", str(exc))
            return

        unit_choice = self.var_p_unit.get()
        use_hygiene = bool(self.var_filter_hygiene.get())
        sa_in_rad = bool(self.var_sa_in_rad.get())

        def prepare(raw: pd.DataFrame, tire_tag: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
            df = raw.copy()
            df["SA"] = pd.to_numeric(df["SA"], errors="coerce")
            if sa_in_rad:
                df["SA"] = np.degrees(df["SA"])
            df["FY"] = pd.to_numeric(df["FY"], errors="coerce")
            df["FZ"] = np.abs(pd.to_numeric(df["FZ"], errors="coerce"))
            df = df[np.isfinite(df["SA"]) & np.isfinite(df["FY"]) & np.isfinite(df["FZ"])].copy()

            if df.empty:
                notes.append(f"{tire_tag}: 0 finite rows after reading SA/FY/FZ.")
                return df, {"raw": 0, "hygiene": 0, "linear_sa": 0, "binned": 0}

            max_sa = float(np.nanmax(np.abs(df["SA"])))
            if not sa_in_rad and 0.0 < max_sa < 0.5:
                notes.append(
                    f"WARNING {tire_tag}: max|SA|={max_sa:.3f}. If SA was exported in radians, check 'Input SA column is in radians'."
                )

            p_kpa, p_notes = _normalize_per_file_pressure(df, unit_choice)
            df["P_kPa"] = p_kpa
            notes.extend(f"{tire_tag}: {pn}" for pn in p_notes[:3])

            hyg_mask, hyg_stats = _condition_hygiene_mask(
                df,
                filter_stationary=use_hygiene,
                filter_long_slip=use_hygiene,
            )
            if hyg_stats["dropped_dwell"] or hyg_stats["dropped_sl"]:
                notes.append(
                    f"{tire_tag}: signal filter removed {hyg_stats['dropped_dwell']:,} dwell (V~0) and {hyg_stats['dropped_sl']:,} combined-slip points."
                )
            sa_lin_cnt = int(np.count_nonzero(hyg_mask & (np.abs(df["SA"].to_numpy(dtype=float)) <= sa_max)))

            classed, _counts = classify_frame(df, spec)
            funnel = {
                "raw": len(df),
                "hygiene": int(np.count_nonzero(hyg_mask)),
                "linear_sa": sa_lin_cnt,
                "binned": len(classed),
            }
            return classed, funnel

        if self._raw_a is not None:
            self._class_a, funnel_a = prepare(self._raw_a, f"Tire A ({self.var_name_a.get()})")
            self._res_a = compute_linear_stiffness(
                self._class_a,
                sa_max_deg=sa_max,
                min_points=min_pts,
                filter_stationary=use_hygiene,
                filter_long_slip=use_hygiene,
            )
            summary_a = (
                f"Tire A ({self.var_name_a.get()}): {len(self._res_a)} CFa bins "
                f"(|SA|<={sa_max}°, linear pts={funnel_a['linear_sa']:,}, binned={funnel_a['binned']:,})."
            )
            notes.insert(0, summary_a)
            if self._res_a.empty and not self._raw_a.empty:
                fz_fin = pd.to_numeric(self._raw_a["FZ"], errors="coerce").dropna()
                fz_rng = f"{float(np.abs(fz_fin).min()):.0f}..{float(np.abs(fz_fin).max()):.0f} N" if not fz_fin.empty else "N/A"
                notes.append(
                    f"  ! No CFa bins passed regression for Tire A (file Fz={fz_rng}, linear pts={funnel_a['linear_sa']}). "
                    "Click 'Auto-Detect Bins', lower Min pts, or widen |SA| max."
                )
        else:
            self._class_a = None
            self._res_a = pd.DataFrame()

        if self._raw_b is not None:
            self._class_b, funnel_b = prepare(self._raw_b, f"Tire B ({self.var_name_b.get()})")
            self._res_b = compute_linear_stiffness(
                self._class_b,
                sa_max_deg=sa_max,
                min_points=min_pts,
                filter_stationary=use_hygiene,
                filter_long_slip=use_hygiene,
            )
            notes.append(
                f"Tire B ({self.var_name_b.get()}): {len(self._res_b)} CFa bins (linear pts={funnel_b['linear_sa']:,})."
            )
        else:
            self._class_b = None
            self._res_b = pd.DataFrame()

        # Gain B vs A merge
        if not self._res_a.empty and not self._res_b.empty:
            merged = pd.merge(
                self._res_a[[*CELL_COLUMNS, "cfa_N_deg", "Fz_mean_N", "IA_mean_deg", "P_mean_kPa"]],
                self._res_b[[*CELL_COLUMNS, "cfa_N_deg"]],
                on=CELL_COLUMNS,
                suffixes=("_a", "_b"),
            )
            merged["delta_N_deg"] = merged["cfa_N_deg_b"] - merged["cfa_N_deg_a"]
            merged["gain_pct"] = 100.0 * merged["delta_N_deg"] / merged["cfa_N_deg_a"].replace(0.0, np.nan)
            self._gain_df = merged
        else:
            self._gain_df = pd.DataFrame()

        # Refresh filter combobox options numerically
        all_dfs = [d for d in (self._res_a, self._res_b) if not d.empty]
        if all_dfs:
            cat = pd.concat(all_dfs, ignore_index=True)
            fz_vals = ["All"] + _sort_bin_labels(cat, "Fz_bin_N", "Fz_mean_N")
            ia_vals = ["All"] + _sort_bin_labels(cat, "IA_bin_deg", "IA_mean_deg")
            p_vals = ["All"] + _sort_bin_labels(cat, "P_bin_kPa", "P_mean_kPa")
        else:
            fz_vals = ia_vals = p_vals = ["All"]

        self.cb_fz["values"] = fz_vals
        self.cb_ia["values"] = ia_vals
        self.cb_p["values"] = p_vals
        if self.var_sel_fz.get() not in fz_vals:
            self.var_sel_fz.set("All")
        if self.var_sel_ia.get() not in ia_vals:
            self.var_sel_ia.set("All")
        if self.var_sel_p.get() not in p_vals:
            self.var_sel_p.set("All")

        self._selected_cell = None
        self._populate_tables()
        self._set_notes("\n".join(notes))
        self.draw()
        self._status(notes[0] if notes else "CFa calculation updated.")

    def _set_notes(self, text: str):
        self.txt_notes.config(state="normal")
        self.txt_notes.delete("1.0", "end")
        self.txt_notes.insert("1.0", text)
        self.txt_notes.config(state="disabled")

    def _populate_tables(self):
        for tree in (self.tree_res, self.tree_gain):
            for iid in tree.get_children():
                tree.delete(iid)
        self._row_cell_map.clear()

        row_i = 0
        for label, df in ((self.var_name_a.get(), self._res_a), (self.var_name_b.get(), self._res_b)):
            if df is None or df.empty:
                continue
            for _, r in df.iterrows():
                tag = "even" if row_i % 2 == 0 else "odd"
                norm_s = f"{r['cfa_norm_deg']:.3f}" if np.isfinite(r.get("cfa_norm_deg", np.nan)) else "--"
                iid = self.tree_res.insert(
                    "",
                    "end",
                    values=(
                        label,
                        r["Fz_bin_N"],
                        r["IA_bin_deg"],
                        r["P_bin_kPa"],
                        f"{r['cfa_N_deg']:.1f}",
                        f"{r['cfa_N_rad'] / 1000.0:.2f}",
                        norm_s,
                        f"{r['r2']:.4f}",
                        f"{r['rmse_N']:.1f}",
                        int(r["n_points"]),
                    ),
                    tags=(tag,),
                )
                self._row_cell_map[iid] = (str(r["Fz_bin_N"]), str(r["IA_bin_deg"]), str(r["P_bin_kPa"]))
                row_i += 1

        if not self._gain_df.empty:
            for g_i, (_, r) in enumerate(self._gain_df.iterrows()):
                tag = "even" if g_i % 2 == 0 else "odd"
                self.tree_gain.insert(
                    "",
                    "end",
                    values=(
                        r["Fz_bin_N"],
                        r["IA_bin_deg"],
                        r["P_bin_kPa"],
                        f"{r['cfa_N_deg_a']:.1f}",
                        f"{r['cfa_N_deg_b']:.1f}",
                        f"{r['delta_N_deg']:+.1f}",
                        f"{r['gain_pct']:+.2f}%" if np.isfinite(r["gain_pct"]) else "--",
                    ),
                    tags=(tag,),
                )

    # ------------------------------------------------------------------
    # PLOTTING
    # ------------------------------------------------------------------
    def _filter_df(self, df: pd.DataFrame, *, ignore: str = "") -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        if ignore != "fz" and self.var_sel_fz.get() != "All":
            out = out[out["Fz_bin_N"] == self.var_sel_fz.get()]
        if ignore != "ia" and self.var_sel_ia.get() != "All":
            out = out[out["IA_bin_deg"] == self.var_sel_ia.get()]
        if ignore != "p" and self.var_sel_p.get() != "All":
            out = out[out["P_bin_kPa"] == self.var_sel_p.get()]
        return out

    def draw(self):
        c = self.c
        self.fig.clear()
        self.fig.patch.set_facecolor(c["panel"])
        ax = self.fig.add_subplot(111)

        if self._res_a.empty and self._res_b.empty:
            msg = (
                "No valid CFa bins matched current filters — check 'Diagnostics & Notes'\n"
                "or click 'Auto-Detect Bins' / widen |SA| max."
                if (self._raw_a is not None or self._raw_b is not None)
                else "Load .MAT / .CSV / .TXT telemetry for Tire A (or pull from Step 1)"
            )
            ax.text(0.5, 0.5, msg, ha="center", va="center", color=c["text_muted"])
            style_axes_2d(ax, zero_lines=False, c=c)
            self.canvas.draw_idle()
            return

        mode = self.var_plot_mode.get()
        if mode.startswith("CFa(Fz)"):
            self._plot_vs_axis(ax, "Fz_mean_N", "fz", "Vertical Load Fz [N]", "CFa(Fz) — Cornering Stiffness vs Load")
        elif mode.startswith("CFa(IA)"):
            self._plot_vs_axis(ax, "IA_mean_deg", "ia", "Camber IA [deg]", "CFa(IA) — Camber Sensitivity")
        elif mode.startswith("CFa(P)"):
            self._plot_vs_axis(ax, "P_mean_kPa", "p", "Pressure P [kPa]", "CFa(P) — Inflation Pressure Sensitivity")
        elif mode.startswith("Heatmap"):
            self._plot_heatmap(ax)
        else:
            self._plot_regression(ax)

        self.fig.tight_layout(pad=1.5)
        self.canvas.draw_idle()

    def _plot_vs_axis(self, ax, x_col: str, ignore_key: str, xlabel: str, title: str):
        c = self.c
        bin_col = BIN_BY_XCOL[x_col]
        plotted = False
        for label, df, color, marker in (
            (self.var_name_a.get(), self._filter_df(self._res_a, ignore=ignore_key), CHART_PALETTE[0], "o"),
            (self.var_name_b.get(), self._filter_df(self._res_b, ignore=ignore_key), CHART_PALETTE[1], "s"),
        ):
            if df.empty or not df[x_col].notna().any():
                continue
            valid = df[df[x_col].notna()].copy()
            grp = (
                valid.groupby(bin_col, as_index=False)
                .agg({x_col: "mean", "cfa_N_deg": "mean"})
                .sort_values(x_col)
            )
            if len(valid) > len(grp):
                ax.scatter(valid[x_col], valid["cfa_N_deg"], color=color, marker=marker, s=22, alpha=0.30)
            ax.plot(grp[x_col], grp["cfa_N_deg"], marker=marker, markersize=6.5, color=color, linewidth=2.3, label=label)
            plotted = True

        style_axes_2d(ax, title=title, xlabel=xlabel, ylabel="CFa [N/deg]", zero_lines=False, c=c)
        if plotted:
            style_legend(ax, c=c)

    def _plot_heatmap(self, ax):
        c = self.c
        target = self.var_heatmap_target.get()
        df = self._filter_df(self._res_b if target == "B" and not self._res_b.empty else self._res_a, ignore="fz")
        df = df if not df.empty else self._res_a
        if df.empty:
            style_axes_2d(ax, title="No data for CFa heatmap", c=c)
            return

        piv = df.pivot_table(index="IA_bin_deg", columns="Fz_bin_N", values="cfa_N_deg", aggfunc="mean")
        if piv.empty:
            style_axes_2d(ax, title="No data for CFa heatmap", c=c)
            return

        fz_cols = _sort_bin_labels(df, "Fz_bin_N", "Fz_mean_N")
        ia_rows = _sort_bin_labels(df, "IA_bin_deg", "IA_mean_deg")
        piv = piv.reindex(index=[r for r in ia_rows if r in piv.index], columns=[col for col in fz_cols if col in piv.columns])
        arr = piv.to_numpy(dtype=float)
        if not np.isfinite(arr).any():
            style_axes_2d(ax, title="No finite CFa cells for heatmap", c=c)
            return

        im = ax.imshow(arr, cmap="plasma", aspect="auto")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns.tolist(), color=c["text_secondary"], fontsize=8.5)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index.tolist(), color=c["text_secondary"], fontsize=8.5)

        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if np.isfinite(arr[i, j]):
                    ax.text(j, i, f"{arr[i, j]:.0f}", ha="center", va="center", color="#ffffff", fontweight="bold", fontsize=8.5)

        name = self.var_name_b.get() if target == "B" and not self._res_b.empty else self.var_name_a.get()
        style_axes_2d(ax, title=f"CFa Heatmap [N/deg] — {name}", xlabel="Fz Bin [N]", ylabel="Camber IA Bin [deg]", zero_lines=False, c=c)
        ax.grid(False)
        cbar = self.fig.colorbar(im, ax=ax, shrink=0.85)
        style_colorbar(cbar, "CFa [N/deg]", c)

    def _plot_regression(self, ax):
        c = self.c
        try:
            sa_max = parse_float(self.var_sa_max.get(), "SA max") if self.var_sa_max.get() else 1.5
        except ValueError:
            sa_max = 1.5
        use_hygiene = bool(self.var_filter_hygiene.get())
        sa_in_rad = bool(self.var_sa_in_rad.get())
        plotted = False

        ref_cell: Optional[Tuple[str, str, str]] = self._selected_cell
        row_a = None
        if not self._res_a.empty:
            if ref_cell is not None:
                hit_a = self._res_a[
                    (self._res_a["Fz_bin_N"] == ref_cell[0])
                    & (self._res_a["IA_bin_deg"] == ref_cell[1])
                    & (self._res_a["P_bin_kPa"] == ref_cell[2])
                ]
                if not hit_a.empty:
                    row_a = hit_a.iloc[0]
            if row_a is None:
                f_a = self._filter_df(self._res_a)
                row_a = f_a.iloc[0] if not f_a.empty else self._res_a.iloc[0]
                ref_cell = (row_a["Fz_bin_N"], row_a["IA_bin_deg"], row_a["P_bin_kPa"])

        row_b = None
        if not self._res_b.empty:
            if ref_cell is not None:
                same_b = self._res_b[
                    (self._res_b["Fz_bin_N"] == ref_cell[0])
                    & (self._res_b["IA_bin_deg"] == ref_cell[1])
                    & (self._res_b["P_bin_kPa"] == ref_cell[2])
                ]
                if not same_b.empty:
                    row_b = same_b.iloc[0]
            if row_b is None:
                f_b = self._filter_df(self._res_b)
                row_b = f_b.iloc[0] if not f_b.empty else self._res_b.iloc[0]
                if ref_cell is None:
                    ref_cell = (row_b["Fz_bin_N"], row_b["IA_bin_deg"], row_b["P_bin_kPa"])

        for label, cls_df, row, color in (
            (self.var_name_a.get(), self._class_a, row_a, CHART_PALETTE[0]),
            (self.var_name_b.get(), self._class_b, row_b, CHART_PALETTE[1]),
        ):
            if cls_df is None or cls_df.empty or row is None:
                continue
            hyg_mask, _ = _condition_hygiene_mask(cls_df, filter_stationary=use_hygiene, filter_long_slip=use_hygiene)
            sa_vals = pd.to_numeric(cls_df["SA"], errors="coerce")
            if sa_in_rad and not self._raw_is_converted(cls_df):
                sa_vals = np.degrees(sa_vals)
            mask = (
                hyg_mask
                & (cls_df["Fz_bin_N"] == row["Fz_bin_N"])
                & (cls_df["IA_bin_deg"] == row["IA_bin_deg"])
                & (cls_df["P_bin_kPa"] == row["P_bin_kPa"])
                & (np.abs(sa_vals) <= sa_max)
            )
            pts = cls_df[mask]
            if pts.empty:
                continue
            ax.scatter(pts["SA"], pts["FY"], color=color, s=20, alpha=0.55)
            xs = np.linspace(-sa_max, sa_max, 50)
            ys = row["slope_N_deg"] * xs + row["intercept_N"]
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=2.3,
                label=f"{label} ({row['Fz_bin_N']}N, IA={row['IA_bin_deg']}°, P={row['P_bin_kPa']}): CFa={row['cfa_N_deg']:.1f} N/°, R²={row['r2']:.3f}",
            )
            plotted = True

        title = (
            f"Fy(SA) Linear Fit | Fz={ref_cell[0]} N, IA={ref_cell[1]} deg, P={ref_cell[2]} kPa"
            if ref_cell is not None
            else "Fy(SA) Linear Zone Regression"
        )
        style_axes_2d(ax, title=title, xlabel="SA [deg]", ylabel="Fy [N]", c=c)
        if plotted:
            style_legend(ax, c=c)

    @staticmethod
    def _raw_is_converted(df: pd.DataFrame) -> bool:
        # _class_a / _class_b already have SA converted during prepare()
        return True

    # ------------------------------------------------------------------
    # EXPORT HELPERS (Called by Unified Export Hub or scripts)
    # ------------------------------------------------------------------
    def get_export_dataframe(self, include_gain: bool = True) -> pd.DataFrame:
        chunks = []
        if not self._res_a.empty:
            a = self._res_a.copy()
            a.insert(0, "Tire", self.var_name_a.get())
            chunks.append(a)
        if not self._res_b.empty:
            b = self._res_b.copy()
            b.insert(0, "Tire", self.var_name_b.get())
            chunks.append(b)
        if not chunks:
            return pd.DataFrame()
        out = pd.concat(chunks, ignore_index=True)
        if include_gain and not self._gain_df.empty:
            gain_cols = self._gain_df[[*CELL_COLUMNS, "delta_N_deg", "gain_pct"]]
            out = out.merge(gain_cols, on=CELL_COLUMNS, how="left")
        return out

    def export_csv(self, path: Optional[str] = None, *, sep: str = ";"):
        df_exp = self.get_export_dataframe()
        if df_exp.empty:
            messagebox.showwarning("No Results", "Calculate CFa results before exporting.")
            return
        if not path:
            path = filedialog.asksaveasfilename(
                title="Export CFa Stiffness Results (CSV)",
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv"), ("All Files", "*.*")],
                initialfile="PUTM_CFa_Analysis.csv",
            )
        if not path:
            return
        df_exp.to_csv(path, sep=sep, index=False, encoding="utf-8")
        self._status(f"Exported CFa stiffness results to {path}")

    def save_plot(self, path: Optional[str] = None, dpi: int = 220):
        if not path:
            path = filedialog.asksaveasfilename(
                title="Save CFa Stiffness Plot",
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
                initialfile="PUTM_CFa_Plot.png",
            )
        if not path:
            return
        self.fig.savefig(path, dpi=dpi, facecolor=self.fig.get_facecolor())
        self._status(f"Saved CFa plot: {path}")
