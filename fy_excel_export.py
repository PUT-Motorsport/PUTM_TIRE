"""fy_excel_export.py — Excel (.xlsx) report generator for Fy(SA) tire characteristics.
=======================================================================================
PUT MOTORSPORT — Formula Student Tire Laboratory.
"""

from __future__ import annotations

import datetime
import os
import re
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import openpyxl
from openpyxl.chart import Reference, ScatterChart, Series
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from condition_export import (
    DEFAULT_FZ_CENTERS,
    DEFAULT_FZ_TOL,
    DEFAULT_IA_CENTERS,
    DEFAULT_IA_TOL,
    DEFAULT_P_RANGES,
    NO_DATA,
    PRESSURE_UNITS,
    Bin,
    ConditionSpec,
    center_bins,
    collect_condition_data,
    parse_float,
    parse_number_list,
    parse_range_list,
    range_bins,
)
from fitting import fit_pure_magic
from tire_model import RunConfig, pacejka_fy

# -----------------------------------------------------------------------------
# Default Slip Angle (SA) grid settings
# -----------------------------------------------------------------------------
DEFAULT_SA_MIN = -12.0
DEFAULT_SA_MAX = 12.0
DEFAULT_SA_STEP = 0.5
DEFAULT_SA_LIN_RANGE = 1.2  # [deg] for linear cornering stiffness CFa


@dataclass(frozen=True)
class FyExcelSpec:
    """Configuration parameters for Fy(SA) Excel workbook generation."""
    fz_bins: Tuple[Bin, ...]
    ia_bins: Tuple[Bin, ...]
    p_bins: Tuple[Bin, ...]
    p_to_kpa: float = 1.0
    p_unit_label: str = "kPa"
    nearest_only: bool = True
    sa_min: float = DEFAULT_SA_MIN
    sa_max: float = DEFAULT_SA_MAX
    sa_step: float = DEFAULT_SA_STEP
    sa_lin_range_deg: float = DEFAULT_SA_LIN_RANGE
    include_fits: bool = True
    include_binned: bool = True
    include_charts: bool = True
    include_kpi: bool = True
    min_points_for_fit: int = 15

    @classmethod
    def default(cls) -> "FyExcelSpec":
        return cls(
            fz_bins=center_bins((1000.0,), DEFAULT_FZ_TOL),
            ia_bins=center_bins((0.0,), DEFAULT_IA_TOL),
            p_bins=range_bins(DEFAULT_P_RANGES),
            p_to_kpa=1.0,
            p_unit_label="kPa",
            nearest_only=True,
            sa_min=DEFAULT_SA_MIN,
            sa_max=DEFAULT_SA_MAX,
            sa_step=DEFAULT_SA_STEP,
        )

    def to_condition_spec(self) -> ConditionSpec:
        """Convert to ConditionSpec compatible with condition_export.py."""
        return ConditionSpec(
            fz_bins=self.fz_bins,
            ia_bins=self.ia_bins,
            p_bins=self.p_bins,
            p_to_kpa=self.p_to_kpa,
            nearest_only=self.nearest_only,
        )


@dataclass
class ConditionFyResult:
    """Wynik obliczeń dla pojedynczej komórki (Fz, Camber IA, Ciśnienie P)."""
    fz_bin: str
    ia_bin: str
    p_bin: str
    fz_mean: float
    ia_mean: float
    p_mean_kpa: float
    p_mean_psi: float
    n_samples: int
    fit_success: bool = False
    fit_message: str = ""
    pacejka_params: Dict[str, float] = field(default_factory=dict)
    r2: float = float("nan")
    rmse: float = float("nan")
    fy_peak: float = float("nan")
    sa_peak_deg: float = float("nan")
    cfa_deg: float = float("nan")     # [N/deg]
    cfa_rad: float = float("nan")     # [N/rad]
    cfa_lin_deg: float = float("nan") # [N/deg] z regresji liniowej
    mu_peak: float = float("nan")
    sa_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    fy_fitted: np.ndarray = field(default_factory=lambda: np.array([]))
    # Punkty uśrednione z pomiarów
    sa_binned: np.ndarray = field(default_factory=lambda: np.array([]))
    fy_binned_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    fy_binned_std: np.ndarray = field(default_factory=lambda: np.array([]))
    fy_binned_count: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def label(self) -> str:
        return f"Fz={self.fz_bin}N | P={self.p_bin}kPa"


@dataclass
class FyExcelExportResult:
    """Zbiorczy rezultat przygotowany do wygenerowania skoroszytu Excel."""
    spec: FyExcelSpec
    conditions: List[ConditionFyResult]
    sa_grid: np.ndarray
    summary_df: pd.DataFrame
    notes: List[str] = field(default_factory=list)
    files_used: int = 0
    rows_read: int = 0
    rows_used: int = 0
    cells_fitted: int = 0
    cells_empty: int = 0


# -----------------------------------------------------------------------------
# Logika obliczeniowa (bez zależności od Tkinter)
# -----------------------------------------------------------------------------
def make_sa_grid(sa_min: float, sa_max: float, sa_step: float) -> np.ndarray:
    """Generate a uniform Slip Angle grid in degrees."""
    if sa_step <= 0:
        raise ValueError("SA step must be greater than 0.")
    if sa_min >= sa_max:
        raise ValueError(f"Inverted SA range: {sa_min} >= {sa_max}.")
    n_points = int(round((sa_max - sa_min) / sa_step)) + 1
    return np.round(np.linspace(sa_min, sa_max, n_points), decimals=3)


def _compute_binned_data(sa: np.ndarray, fy: np.ndarray, sa_grid: np.ndarray, step: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average measured points inside each SA grid bucket."""
    binned_sa, means, stds, counts = [], [], [], []
    half = step / 2.0
    for target in sa_grid:
        mask = (sa >= target - half) & (sa < target + half)
        n = int(np.count_nonzero(mask))
        if n > 0:
            sub_fy = fy[mask]
            binned_sa.append(target)
            means.append(float(np.mean(sub_fy)))
            stds.append(float(np.std(sub_fy)) if n > 1 else 0.0)
            counts.append(n)
        else:
            binned_sa.append(target)
            means.append(float("nan"))
            stds.append(float("nan"))
            counts.append(0)
    return np.array(binned_sa), np.array(means), np.array(stds), np.array(counts, dtype=int)


def _fit_cell_pacejka(
    df_cell: pd.DataFrame,
    spec: FyExcelSpec,
    fz_bin: str,
    ia_bin: str,
    p_bin: str,
    sa_grid: np.ndarray,
) -> ConditionFyResult:
    """Fit Pacejka model and compute KPIs for a single (Fz, IA, P) cell."""
    n_samples = len(df_cell)
    if n_samples == 0:
        return ConditionFyResult(
            fz_bin=fz_bin, ia_bin=ia_bin, p_bin=p_bin,
            fz_mean=float("nan"), ia_mean=float("nan"),
            p_mean_kpa=float("nan"), p_mean_psi=float("nan"),
            n_samples=0, fit_success=False, fit_message="No samples in cell",
            sa_grid=sa_grid, fy_fitted=np.full(len(sa_grid), np.nan),
        )

    sa_deg = df_cell["SA"].to_numpy(dtype=float)
    fy_measured = -df_cell["FY"].to_numpy(dtype=float)
    fz_measured = np.abs(df_cell["FZ"].to_numpy(dtype=float))
    ia_measured = df_cell["IA"].to_numpy(dtype=float) if "IA" in df_cell else np.zeros(n_samples)
    p_kpa = df_cell["P_kPa"].to_numpy(dtype=float) if "P_kPa" in df_cell else np.full(n_samples, np.nan)

    fz_mean = float(np.nanmean(fz_measured))
    ia_mean = float(np.nanmean(ia_measured))
    p_mean_kpa = float(np.nanmean(p_kpa)) if np.isfinite(p_kpa).any() else float("nan")
    p_mean_psi = p_mean_kpa / 6.894757 if np.isfinite(p_mean_kpa) else float("nan")

    b_sa, b_mean, b_std, b_count = _compute_binned_data(sa_deg, fy_measured, sa_grid, spec.sa_step)

    lin_mask = np.abs(sa_deg) <= spec.sa_lin_range_deg
    lin_cfa_deg = float("nan")
    if np.count_nonzero(lin_mask) >= 8 and np.ptp(sa_deg[lin_mask]) >= 0.2:
        try:
            slope, _ = np.polyfit(sa_deg[lin_mask], fy_measured[lin_mask], 1)
            lin_cfa_deg = abs(float(slope))
        except Exception:
            lin_cfa_deg = float("nan")

    if n_samples < spec.min_points_for_fit or not spec.include_fits:
        peak_est = float(np.percentile(np.abs(fy_measured), 98)) if n_samples > 0 else float("nan")
        return ConditionFyResult(
            fz_bin=fz_bin, ia_bin=ia_bin, p_bin=p_bin,
            fz_mean=fz_mean, ia_mean=ia_mean,
            p_mean_kpa=p_mean_kpa, p_mean_psi=p_mean_psi,
            n_samples=n_samples, fit_success=False,
            fit_message=f"Fit skipped ({n_samples} < {spec.min_points_for_fit})",
            fy_peak=peak_est,
            mu_peak=peak_est / fz_mean if fz_mean > 0 else float("nan"),
            cfa_lin_deg=lin_cfa_deg,
            sa_grid=sa_grid, fy_fitted=b_mean,
            sa_binned=b_sa, fy_binned_mean=b_mean, fy_binned_std=b_std, fy_binned_count=b_count,
        )

    x_rad = np.deg2rad(sa_deg)
    fit_success = False
    fit_message = ""
    params: Dict[str, float] = {}
    r2 = float("nan")
    rmse = float("nan")
    fy_fitted = np.full(len(sa_grid), np.nan)
    fy_peak = float("nan")
    sa_peak_deg = float("nan")
    cfa_rad = float("nan")
    cfa_deg = float("nan")
    mu_peak = float("nan")

    try:
        fit_out = fit_pure_magic(x_rad, fy_measured, model_name="Fy0", slip_unit="rad")
        params = fit_out.parameters
        diag = fit_out.diagnostics
        r2 = float(diag.get("r2", float("nan")))
        rmse = float(diag.get("rmse", float("nan")))

        sa_rad_grid = np.deg2rad(sa_grid)
        b = params["B"]
        c = params["C"]
        d = params["D"]
        e = params["E"]
        sh = params["Sh"]
        sv = params["Sv"]
        fy_fitted = pacejka_fy(sa_rad_grid, b, c, d, e, sh, sv)

        dense_sa = np.linspace(spec.sa_min, spec.sa_max, 601)
        dense_fy = pacejka_fy(np.deg2rad(dense_sa), b, c, d, e, sh, sv)
        idx_peak = int(np.argmax(np.abs(dense_fy)))
        fy_peak = float(dense_fy[idx_peak])
        sa_peak_deg = float(dense_sa[idx_peak])

        cfa_rad = float(abs(b * c * d))
        cfa_deg = float(cfa_rad * (np.pi / 180.0))
        mu_peak = abs(fy_peak) / fz_mean if fz_mean > 0 else float("nan")
        fit_success = True
        fit_message = "OK"
    except Exception as exc:
        fit_success = False
        fit_message = str(exc)
        fy_fitted = b_mean

    return ConditionFyResult(
        fz_bin=fz_bin, ia_bin=ia_bin, p_bin=p_bin,
        fz_mean=fz_mean, ia_mean=ia_mean,
        p_mean_kpa=p_mean_kpa, p_mean_psi=p_mean_psi,
        n_samples=n_samples, fit_success=fit_success, fit_message=fit_message,
        pacejka_params=params, r2=r2, rmse=rmse,
        fy_peak=fy_peak, sa_peak_deg=sa_peak_deg,
        cfa_deg=cfa_deg, cfa_rad=cfa_rad, cfa_lin_deg=lin_cfa_deg,
        mu_peak=mu_peak,
        sa_grid=sa_grid, fy_fitted=fy_fitted,
        sa_binned=b_sa, fy_binned_mean=b_mean, fy_binned_std=b_std, fy_binned_count=b_count,
    )


def build_fy_excel_from_df(
    df: pd.DataFrame,
    spec: FyExcelSpec,
    files_used: int = 1,
    rows_read: int = 0,
    notes: Optional[List[str]] = None,
) -> FyExcelExportResult:
    """Process a binned dataframe into a complete Excel workbook result."""
    sa_grid = make_sa_grid(spec.sa_min, spec.sa_max, spec.sa_step)
    condition_results: List[ConditionFyResult] = []
    collected_notes = list(notes or [])

    req = ["Fz_bin_N", "IA_bin_deg", "P_bin_kPa", "SA", "FY", "FZ"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Dataframe is missing required columns: {missing}")

    # Filtr tylko dla testów Cornering, jeśli kolumna __type__ istnieje
    if "__type__" in df.columns:
        valid_df = df[df["__type__"] == "Cornering"].copy()
    else:
        valid_df = df.copy()

    valid_df = valid_df.dropna(subset=["SA", "FY", "FZ"])
    rows_used = len(valid_df)

    fz_labels = [b.label for b in spec.fz_bins]
    ia_labels = [b.label for b in spec.ia_bins]
    p_labels = [b.label for b in spec.p_bins]

    cells_fitted = 0
    cells_empty = 0

    for fz_l, ia_l, p_l in product(fz_labels, ia_labels, p_labels):
        cell_mask = (
            (valid_df["Fz_bin_N"] == fz_l) &
            (valid_df["IA_bin_deg"] == ia_l) &
            (valid_df["P_bin_kPa"] == p_l)
        )
        cell_df = valid_df[cell_mask]
        res = _fit_cell_pacejka(cell_df, spec, fz_l, ia_l, p_l, sa_grid)
        condition_results.append(res)
        if res.fit_success:
            cells_fitted += 1
        if res.n_samples == 0:
            cells_empty += 1

    # Budowa tabeli podsumowującej KPI
    rows_summary = []
    for c in condition_results:
        p = c.pacejka_params
        rows_summary.append({
            "Fz_bin_N": c.fz_bin,
            "Fz_mean_N": c.fz_mean,
            "IA_bin_deg": c.ia_bin,
            "IA_mean_deg": c.ia_mean,
            "P_bin_kPa": c.p_bin,
            "P_mean_kPa": c.p_mean_kpa,
            "P_mean_psi": c.p_mean_psi,
            "n_samples": c.n_samples,
            "fit_status": "OK" if c.fit_success else c.fit_message,
            "Fy_peak_N": c.fy_peak,
            "SA_peak_deg": c.sa_peak_deg,
            "CFa_N_per_deg": c.cfa_deg,
            "CFa_N_per_rad": c.cfa_rad,
            "CFa_lin_N_per_deg": c.cfa_lin_deg,
            "mu_peak": c.mu_peak,
            "B": p.get("B", float("nan")),
            "C": p.get("C", float("nan")),
            "D": p.get("D", float("nan")),
            "E": p.get("E", float("nan")),
            "Sh_deg": np.rad2deg(p.get("Sh", 0.0)) if "Sh" in p else float("nan"),
            "Sv_N": p.get("Sv", float("nan")),
            "R2": c.r2,
            "RMSE_N": c.rmse,
        })
    summary_df = pd.DataFrame(rows_summary)

    return FyExcelExportResult(
        spec=spec,
        conditions=condition_results,
        sa_grid=sa_grid,
        summary_df=summary_df,
        notes=collected_notes,
        files_used=files_used,
        rows_read=rows_read,
        rows_used=rows_used,
        cells_fitted=cells_fitted,
        cells_empty=cells_empty,
    )


def build_fy_excel_data(
    runs: Iterable[RunConfig],
    spec: FyExcelSpec,
    cache: Optional[dict] = None,
) -> FyExcelExportResult:
    """Read unfiltered measurement files, bin by condition, and fit Fy(SA) curves for Excel."""
    runs_list = list(runs)
    if not runs_list:
        raise ValueError("No measurement files available to process.")

    cond_spec = spec.to_condition_spec()
    cond_result = collect_condition_data(runs_list, cond_spec, cache=cache)

    if cond_result.data.empty:
        raise ValueError(
            "No samples fall inside the configured Fz, Camber, and Pressure bins. "
            "Check bin tolerances and pressure units."
        )

    return build_fy_excel_from_df(
        df=cond_result.data,
        spec=spec,
        files_used=cond_result.files_used,
        rows_read=cond_result.rows_read,
        notes=cond_result.notes,
    )


# -----------------------------------------------------------------------------
# OpenPyXL Workbook Generation
# -----------------------------------------------------------------------------
def _sanitize_sheet_title(name: str) -> str:
    """Sanitize Excel worksheet tab name (<=31 chars, no forbidden symbols)."""
    clean = re.sub(r"[\\/*?:\[\]]", "_", name).strip()
    return clean[:31] if clean else "Sheet"


COLOR_HEADER_BG = "161B22"
COLOR_HEADER_FG = "FFFFFF"
COLOR_SUBHEAD_BG = "21262D"
COLOR_SUBHEAD_FG = "00E5FF"
COLOR_ACCENT_ORANGE = "FF5500"
COLOR_ROW_ALT = "F8FAFC"
COLOR_BORDER = "E2E8F0"
FONT_NAME = "Segoe UI"


def _apply_title_block(ws: openpyxl.worksheet.worksheet.Worksheet, title: str, subtitle: str) -> None:
    ws["A1"] = f"PUT MOTORSPORT — {title.upper()}"
    ws["A1"].font = Font(name=FONT_NAME, size=13, bold=True, color=COLOR_ACCENT_ORANGE)
    ws["A2"] = subtitle
    ws["A2"].font = Font(name=FONT_NAME, size=9, italic=True, color="64748B")
    ws.row_dimensions[1].height = 24
    ws.row_dimensions[2].height = 18


def _apply_table_header(
    ws: openpyxl.worksheet.worksheet.Worksheet,
    row_idx: int,
    headers: Sequence[str],
    bg_color: str = COLOR_HEADER_BG,
    fg_color: str = COLOR_HEADER_FG,
) -> None:
    ws.row_dimensions[row_idx].height = 26
    fill = PatternFill(start_color=bg_color, end_color=bg_color, fill_type="solid")
    font = Font(name=FONT_NAME, size=9.5, bold=True, color=fg_color)
    thin = Side(style="thin", color=COLOR_BORDER)
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    for col_idx, text in enumerate(headers, start=1):
        c = ws.cell(row=row_idx, column=col_idx, value=text)
        c.fill = fill
        c.font = font
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = border


def _autofit_columns(ws: openpyxl.worksheet.worksheet.Worksheet, min_width: int = 12, max_width: int = 40) -> None:
    for col in ws.columns:
        col_letter = get_column_letter(col[0].column)
        max_len = 0
        for cell in col:
            val = cell.value
            if val is not None:
                s = str(val)
                lines = s.split("\n")
                line_max = max(len(l) for l in lines)
                if line_max > max_len:
                    max_len = line_max
        ws.column_dimensions[col_letter].width = min(max(max_len + 3, min_width), max_width)


def generate_excel_workbook(result: FyExcelExportResult) -> openpyxl.Workbook:
    """Create the formatted OpenPyXL workbook from FyExcelExportResult."""
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    thin = Side(style="thin", color=COLOR_BORDER)
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    font_data = Font(name=FONT_NAME, size=9.5)
    fill_alt = PatternFill(start_color=COLOR_ROW_ALT, end_color=COLOR_ROW_ALT, fill_type="solid")

    spec = result.spec
    sa_grid = result.sa_grid
    num_sa = len(sa_grid)
    unique_fzs = [b.label for b in spec.fz_bins]
    unique_ps = [b.label for b in spec.p_bins]

    cond_map: Dict[Tuple[str, str, str], ConditionFyResult] = {
        (c.fz_bin, c.ia_bin, c.p_bin): c for c in result.conditions
    }

    # 1. Per-load Fy(SA) sheets
    for fz_label in unique_fzs:
        sheet_title = _sanitize_sheet_title(f"Fy(SA) @ Fz={fz_label}N")
        ws = wb.create_sheet(title=sheet_title)

        _apply_title_block(
            ws,
            title=f"Lateral Force Fy(SA) — Vertical Load Fz = {fz_label} N",
            subtitle=f"Lateral force vs slip angle across operating pressures | IA = {spec.ia_bins[0].label} deg",
        )

        col_headers = ["Slip Angle SA [deg]"]
        active_conditions: List[ConditionFyResult] = []

        for p_label in unique_ps:
            key = (fz_label, spec.ia_bins[0].label, p_label)
            if key in cond_map:
                c_res = cond_map[key]
                col_headers.append(f"Fy [N] (P = {p_label} {spec.p_unit_label})")
                active_conditions.append(c_res)

        start_row = 4
        _apply_table_header(ws, start_row, col_headers)

        for r_offset, sa_val in enumerate(sa_grid):
            curr_row = start_row + 1 + r_offset
            row_vals = [float(sa_val)]

            for c_res in active_conditions:
                if len(c_res.fy_fitted) > r_offset and np.isfinite(c_res.fy_fitted[r_offset]):
                    row_vals.append(round(float(c_res.fy_fitted[r_offset]), 2))
                else:
                    row_vals.append(None)

            for col_idx, val in enumerate(row_vals, start=1):
                cell = ws.cell(row=curr_row, column=col_idx, value=val)
                cell.font = font_data
                cell.border = border
                if col_idx == 1:
                    cell.alignment = Alignment(horizontal="center")
                    cell.number_format = "0.0"
                else:
                    cell.alignment = Alignment(horizontal="right")
                    cell.number_format = "#,##0.0"
                if r_offset % 2 == 1:
                    cell.fill = fill_alt

        if spec.include_charts and active_conditions and num_sa > 1:
            chart = ScatterChart()
            chart.title = f"Fy(SA) — Pressure Sensitivity (Fz = {fz_label} N)"
            chart.style = 13
            chart.x_axis.title = "Slip Angle SA [deg]"
            chart.y_axis.title = "Lateral Force Fy [N]"
            chart.width = 19
            chart.height = 14

            x_ref = Reference(ws, min_col=1, min_row=start_row + 1, max_row=start_row + num_sa)
            for c_idx in range(2, len(col_headers) + 1):
                y_ref = Reference(ws, min_col=c_idx, min_row=start_row, max_row=start_row + num_sa)
                s = Series(y_ref, x_ref, title_from_data=True)
                s.smooth = True
                chart.series.append(s)

            chart_col = get_column_letter(len(col_headers) + 2)
            ws.add_chart(chart, f"{chart_col}4")

        _autofit_columns(ws)

    # 2. Combined condition matrix sheet
    if len(unique_fzs) > 1:
        ws_mat = wb.create_sheet(title=_sanitize_sheet_title("Matrix_Fy_SA_Fz_P"))
        _apply_title_block(
            ws_mat,
            title="Full Operating Condition Lookup Matrix Fy(SA, Fz, P)",
            subtitle="Fitted lateral force table across all selected Fz and inflation pressure cells",
        )

        mat_headers = ["SA [deg]"]
        mat_cols_conds: List[ConditionFyResult] = []

        for fz_l, p_l in product(unique_fzs, unique_ps):
            key = (fz_l, spec.ia_bins[0].label, p_l)
            if key in cond_map:
                c_res = cond_map[key]
                mat_headers.append(f"Fz={fz_l}N | P={p_l}{spec.p_unit_label}")
                mat_cols_conds.append(c_res)

        start_row = 4
        _apply_table_header(ws_mat, start_row, mat_headers, bg_color="0F172A", fg_color="38BDF8")

        for r_offset, sa_val in enumerate(sa_grid):
            curr_row = start_row + 1 + r_offset
            row_vals = [float(sa_val)]

            for c_res in mat_cols_conds:
                if len(c_res.fy_fitted) > r_offset and np.isfinite(c_res.fy_fitted[r_offset]):
                    row_vals.append(round(float(c_res.fy_fitted[r_offset]), 2))
                else:
                    row_vals.append(None)

            for col_idx, val in enumerate(row_vals, start=1):
                cell = ws_mat.cell(row=curr_row, column=col_idx, value=val)
                cell.font = font_data
                cell.border = border
                if col_idx == 1:
                    cell.alignment = Alignment(horizontal="center")
                    cell.number_format = "0.0"
                else:
                    cell.alignment = Alignment(horizontal="right")
                    cell.number_format = "#,##0.0"
                if r_offset % 2 == 1:
                    cell.fill = fill_alt

        _autofit_columns(ws_mat)

    # 3. KPI Summary Sheet
    if spec.include_kpi:
        ws_kpi = wb.create_sheet(title=_sanitize_sheet_title("Summary_KPI"))
        _apply_title_block(
            ws_kpi,
            title="Engineering KPI Summary & Pacejka Parameters",
            subtitle="Peak lateral grip, cornering stiffness CFa, and Magic Formula fit coefficients",
        )

        kpi_headers = [
            "Fz [N]", f"Pressure [{spec.p_unit_label}]", "Pressure [psi]", "Camber [deg]",
            "Samples N", "Fit Status", "Peak Fy [N]", "SA @ Peak [deg]",
            "CFa [N/deg]", "CFa [N/rad]", "CFa lin [N/deg]", "mu peak (Fy/Fz)",
            "B", "C", "D [N]", "E", "Sh [deg]", "Sv [N]", "R2", "RMSE [N]",
        ]
        start_row = 4
        _apply_table_header(ws_kpi, start_row, kpi_headers, bg_color=COLOR_HEADER_BG, fg_color="F1F5F9")

        for idx, row in result.summary_df.iterrows():
            curr_row = start_row + 1 + idx
            p_psi = row["P_mean_psi"]
            vals = [
                row["Fz_bin_N"],
                row["P_bin_kPa"],
                round(p_psi, 1) if np.isfinite(p_psi) else None,
                row["IA_bin_deg"],
                int(row["n_samples"]),
                str(row["fit_status"]),
                round(float(row["Fy_peak_N"]), 1) if np.isfinite(row["Fy_peak_N"]) else None,
                round(float(row["SA_peak_deg"]), 2) if np.isfinite(row["SA_peak_deg"]) else None,
                round(float(row["CFa_N_per_deg"]), 1) if np.isfinite(row["CFa_N_per_deg"]) else None,
                round(float(row["CFa_N_per_rad"]), 1) if np.isfinite(row["CFa_N_per_rad"]) else None,
                round(float(row["CFa_lin_N_per_deg"]), 1) if np.isfinite(row["CFa_lin_N_per_deg"]) else None,
                round(float(row["mu_peak"]), 3) if np.isfinite(row["mu_peak"]) else None,
                round(float(row["B"]), 2) if np.isfinite(row["B"]) else None,
                round(float(row["C"]), 2) if np.isfinite(row["C"]) else None,
                round(float(row["D"]), 1) if np.isfinite(row["D"]) else None,
                round(float(row["E"]), 2) if np.isfinite(row["E"]) else None,
                round(float(row["Sh_deg"]), 3) if np.isfinite(row["Sh_deg"]) else None,
                round(float(row["Sv_N"]), 1) if np.isfinite(row["Sv_N"]) else None,
                round(float(row["R2"]), 4) if np.isfinite(row["R2"]) else None,
                round(float(row["RMSE_N"]), 1) if np.isfinite(row["RMSE_N"]) else None,
            ]

            for col_idx, val in enumerate(vals, start=1):
                cell = ws_kpi.cell(row=curr_row, column=col_idx, value=val)
                cell.font = font_data
                cell.border = border
                if col_idx in (1, 2, 3, 4, 5, 6):
                    cell.alignment = Alignment(horizontal="center")
                else:
                    cell.alignment = Alignment(horizontal="right")
                if idx % 2 == 1:
                    cell.fill = fill_alt

        _autofit_columns(ws_kpi)

    # 4. Binned experimental data sheet
    if spec.include_binned:
        ws_bin = wb.create_sheet(title=_sanitize_sheet_title("Binned_Measurements"))
        _apply_title_block(
            ws_bin,
            title="Binned Experimental TTC Measurements vs SA",
            subtitle="Mean lateral force, standard deviation, and sample counts per Slip Angle bin",
        )

        binned_headers = [
            "Fz [N]", f"Pressure [{spec.p_unit_label}]", "Camber [deg]",
            "Slip Angle SA [deg]", "Mean Fy [N]", "Std Dev Fy [N]", "Samples N in Bin",
        ]
        start_row = 4
        _apply_table_header(ws_bin, start_row, binned_headers, bg_color=COLOR_HEADER_BG, fg_color="CBD5E1")

        curr_row = start_row + 1
        for c in result.conditions:
            if c.n_samples == 0:
                continue
            for sa_v, mean_v, std_v, cnt_v in zip(c.sa_binned, c.fy_binned_mean, c.fy_binned_std, c.fy_binned_count):
                if cnt_v == 0:
                    continue
                vals = [
                    c.fz_bin,
                    c.p_bin,
                    c.ia_bin,
                    round(float(sa_v), 2),
                    round(float(mean_v), 2) if np.isfinite(mean_v) else None,
                    round(float(std_v), 2) if np.isfinite(std_v) else None,
                    int(cnt_v),
                ]
                for col_idx, val in enumerate(vals, start=1):
                    cell = ws_bin.cell(row=curr_row, column=col_idx, value=val)
                    cell.font = font_data
                    cell.border = border
                    if col_idx in (1, 2, 3, 4):
                        cell.alignment = Alignment(horizontal="center")
                    elif col_idx == 7:
                        cell.alignment = Alignment(horizontal="center")
                    else:
                        cell.alignment = Alignment(horizontal="right")
                curr_row += 1

        _autofit_columns(ws_bin)

    # 5. Metadata sheet
    ws_info = wb.create_sheet(title=_sanitize_sheet_title("Metadata_Info"))
    _apply_title_block(
        ws_info,
        title="Dataset Metadata & Conventions",
        subtitle="Technical parameters, sign conventions, and Magic Formula equations",
    )

    info_data = [
        ("Application", "PUTM Tire Lab — Formula Student Tire Dynamics Suite"),
        ("Team", "PUT Motorsport — Formula Student Team"),
        ("Generated At", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("Input Files Loaded", result.files_used),
        ("Measurement Rows Used", result.rows_used),
        ("Models Fitted", result.cells_fitted),
        ("Empty Cells", result.cells_empty),
        ("", ""),
        ("SIGN CONVENTIONS & UNITS", ""),
        ("Lateral Force Fy", "Fy = -FY_raw (positive Fy corresponds to positive slip angle SA > 0)"),
        ("Slip Angle SA", "Input/display in degrees [deg]; converted to radians internally"),
        ("Vertical Load Fz", "Positive downward compression [N] (raw Calspan TTC is negative)"),
        ("Inflation Pressure P", f"Input unit [{spec.p_unit_label}]; multiplier to kPa: {spec.p_to_kpa}"),
        ("", ""),
        ("PURE-SLIP MAGIC FORMULA (Fy0)", ""),
        ("Equation", "Fy0 = D * sin(C * arctan(B*alpha_eq - E*(B*alpha_eq - arctan(B*alpha_eq)))) + Sv"),
        ("Equivalent Slip", "alpha_eq = alpha + Sh"),
        ("Cornering Stiffness CFa", "CFa = B * C * |D| [N/rad] (slope at alpha = 0)"),
        ("Peak Friction mu", "mu = |Fy_peak| / Fz"),
    ]

    ws_info.row_dimensions[4].height = 22
    for idx, (lbl, val) in enumerate(info_data, start=4):
        c_lbl = ws_info.cell(row=idx, column=1, value=lbl)
        c_val = ws_info.cell(row=idx, column=2, value=val)

        if not val and lbl:
            c_lbl.font = Font(name=FONT_NAME, size=10, bold=True, color=COLOR_ACCENT_ORANGE)
            c_lbl.fill = PatternFill(start_color=COLOR_SUBHEAD_BG, end_color=COLOR_SUBHEAD_BG, fill_type="solid")
            c_val.fill = PatternFill(start_color=COLOR_SUBHEAD_BG, end_color=COLOR_SUBHEAD_BG, fill_type="solid")
        else:
            c_lbl.font = Font(name=FONT_NAME, size=9.5, bold=True, color="334155")
            c_val.font = Font(name=FONT_NAME, size=9.5, color="0F172A")
            c_lbl.border = border
            c_val.border = border

    if result.notes:
        note_start = len(info_data) + 6
        ws_info.cell(row=note_start, column=1, value="NOTES & WARNINGS:").font = Font(
            name=FONT_NAME, size=10, bold=True, color="E11D48"
        )
        for n_idx, note in enumerate(result.notes, start=note_start + 1):
            ws_info.cell(row=n_idx, column=1, value=f"- {note}").font = Font(name=FONT_NAME, size=9, color="475569")

    _autofit_columns(ws_info, min_width=18, max_width=80)
    return wb


def save_fy_excel(path: str, result: FyExcelExportResult) -> None:
    """Save generated workbook to an .xlsx file."""
    if not result.conditions:
        raise ValueError("No results available to save.")
    wb = generate_excel_workbook(result)
    wb.save(path)


def format_export_summary(result: FyExcelExportResult) -> str:
    """Plain-text summary of Excel generation output."""
    lines = [
        "=================================================================",
        "  PUT MOTORSPORT — EXCEL WORKBOOK REPORT Fy(SA) [Fz, P]",
        "=================================================================",
        f"Source files: {result.files_used} | Samples used: {result.rows_used}",
        f"Cells (Fz x IA x P): {len(result.conditions)} | Fitted: {result.cells_fitted} | Empty: {result.cells_empty}",
        f"Slip Angle SA grid: {result.spec.sa_min:g} to {result.spec.sa_max:g} deg (step {result.spec.sa_step:g} deg, {len(result.sa_grid)} pts)",
        "",
        "  Fz [N]    P [kPa]    Samples    Peak Fy [N]   CFa [N/deg]   mu peak   Status",
        "-------------------------------------------------------------------------",
    ]

    for c in result.conditions:
        peak_str = f"{c.fy_peak:>10.1f}" if np.isfinite(c.fy_peak) else "       N/A"
        cfa_str = f"{c.cfa_deg:>11.1f}" if np.isfinite(c.cfa_deg) else "        N/A"
        mu_str = f"{c.mu_peak:>7.2f}" if np.isfinite(c.mu_peak) else "    N/A"
        status_str = "OK" if c.fit_success else (c.fit_message[:12] if c.fit_message else "-")
        lines.append(
            f"{c.fz_bin:>8} {c.p_bin:>10} {c.n_samples:>10} {peak_str} {cfa_str} {mu_str}   {status_str}"
        )

    if result.notes:
        lines.append("")
        lines.append("Notes:")
        for note in result.notes:
            lines.append(f" ! {note}")

    return "\n".join(lines)
