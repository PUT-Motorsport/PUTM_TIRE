"""Export tire test measurements binned into Fz x camber (IA) x pressure cells (CSV).

Tk-independent module. Reads unfiltered raw data from loaded ``.mat`` / ``.csv``
files so all operating conditions appear in a single structured export.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from tire_model import RunConfig, TireDataManager

NO_DATA = "N/A"  # marker used when a file lacks a channel altogether (e.g. missing P)
PRESSURE_UNITS = {"kPa": 1.0, "psi": 6.894757, "bar": 100.0}

DEFAULT_FZ_CENTERS = (600.0, 800.0, 1000.0, 1200.0)
DEFAULT_FZ_TOL = 200.0
DEFAULT_IA_CENTERS = (0.0, -1.0, -2.0, -3.0)
DEFAULT_IA_TOL = 0.5
DEFAULT_P_RANGES = ((50.0, 60.0), (65.0, 75.0), (80.0, 85.0), (94.0, 100.0))

CELL_COLUMNS = ["Fz_bin_N", "IA_bin_deg", "P_bin_kPa"]
_CORE_CHANNELS = {"Cornering": ["SA", "FY", "FZ", "IA"], "Drive/Brake": ["FX", "SL", "FZ"]}


# ------------------------------------------------------------------ intervals
@dataclass(frozen=True)
class Bin:
    label: str
    low: float
    high: float

    @property
    def center(self) -> float:
        return (self.low + self.high) / 2.0


def _fmt(value: float) -> str:
    return f"{value:g}"


def center_bins(centers: Sequence[float], tol: float) -> Tuple[Bin, ...]:
    """Closed intervals [c - tol, c + tol]."""
    if tol < 0:
        raise ValueError("Tolerance cannot be negative.")
    if not centers:
        raise ValueError("Provide at least one bin center.")
    return tuple(Bin(_fmt(c), c - tol, c + tol) for c in centers)


def range_bins(ranges: Sequence[Tuple[float, float]]) -> Tuple[Bin, ...]:
    if not ranges:
        raise ValueError("Provide at least one pressure range.")
    bins = []
    for low, high in ranges:
        if low >= high:
            raise ValueError(f"Range {_fmt(low)}-{_fmt(high)} is inverted or empty.")
        bins.append(Bin(f"{_fmt(low)}-{_fmt(high)}", low, high))
    return tuple(bins)


@dataclass(frozen=True)
class ConditionSpec:
    fz_bins: Tuple[Bin, ...]
    ia_bins: Tuple[Bin, ...]
    p_bins: Tuple[Bin, ...]
    p_to_kpa: float = 1.0       # multiplier from input unit to kPa
    nearest_only: bool = True   # Fz/IA: assign to nearest center only (no duplicates)

    @classmethod
    def default(cls) -> "ConditionSpec":
        return cls(
            fz_bins=center_bins(DEFAULT_FZ_CENTERS, DEFAULT_FZ_TOL),
            ia_bins=center_bins(DEFAULT_IA_CENTERS, DEFAULT_IA_TOL),
            p_bins=range_bins(DEFAULT_P_RANGES),
        )


# --------------------------------------------------------------- UI parsing
def parse_float(text: str, name: str = "value") -> float:
    try:
        return float(str(text).strip().replace(",", "."))
    except ValueError:
        raise ValueError(f"Field '{name}' must be numeric (got: '{text}').")


def parse_number_list(text: str, name: str = "list") -> List[float]:
    """Comma, semicolon, or space-separated numbers (dot decimals, e.g. 0.5)."""
    parts = [p for p in re.split(r"[,\s;]+", str(text).strip()) if p]
    if not parts:
        raise ValueError(f"Field '{name}' is empty.")
    try:
        return [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"Field '{name}': enter comma-separated numbers with dot decimals (e.g. 0, -1, -2.5).")


_RANGE = re.compile(r"(\d+(?:[.,]\d+)?)\s*[-–—]\s*(\d+(?:[.,]\d+)?)")


def parse_range_list(text: str, name: str = "ranges") -> List[Tuple[float, float]]:
    """Ranges formatted as ``50-60, 65-75``."""
    cleaned = str(text).strip()
    found = _RANGE.findall(cleaned)
    if not found:
        raise ValueError(f"Field '{name}': enter ranges such as 50-60, 65-75.")
    leftover = _RANGE.sub("", cleaned)
    if re.search(r"[^\s,;]", leftover):
        raise ValueError(f"Field '{name}': unrecognized token '{leftover.strip(' ,;')}'. Use 50-60, 65-75.")
    return [(float(a.replace(",", ".")), float(b.replace(",", "."))) for a, b in found]


# --------------------------------------------------------------- classification
def _membership(values: np.ndarray, bins: Sequence[Bin], nearest_only: bool) -> List[np.ndarray]:
    """Membership masks for each bin (NaN belongs to no bin)."""
    member = np.column_stack([(values >= b.low) & (values <= b.high) for b in bins])
    if nearest_only and len(bins) > 1:
        centers = np.array([b.center for b in bins])
        dist = np.abs(values[:, None] - centers[None, :])
        dist = np.where(member & np.isfinite(dist), dist, np.inf)
        winner = np.argmin(dist, axis=1)
        member = member & (winner[:, None] == np.arange(len(bins))[None, :])
    return [member[:, i] for i in range(len(bins))]


def _dimension(values: np.ndarray, bins: Sequence[Bin], nearest_only: bool) -> List[Tuple[str, np.ndarray]]:
    """When channel is missing/all-NaN, assign all rows to NO_DATA."""
    if not np.isfinite(values).any():
        return [(NO_DATA, np.ones(len(values), dtype=bool))]
    return [(b.label, m) for b, m in zip(bins, _membership(values, bins, nearest_only))]


def classify_frame(df: pd.DataFrame, spec: ConditionSpec) -> Tuple[pd.DataFrame, Dict[Tuple[str, str, str], int]]:
    """Classify samples into (Fz, IA, P) bins and return (long_df, counts)."""
    fz = df["FZ"].to_numpy(dtype=float)
    ia = df["IA"].to_numpy(dtype=float) if "IA" in df else np.full(len(df), np.nan)
    pressure = df["P_kPa"].to_numpy(dtype=float)

    chunks, counts = [], {}
    for (fz_label, fz_mask), (ia_label, ia_mask), (p_label, p_mask) in product(
        _dimension(fz, spec.fz_bins, spec.nearest_only),
        _dimension(ia, spec.ia_bins, spec.nearest_only),
        _dimension(pressure, spec.p_bins, False),
    ):
        mask = fz_mask & ia_mask & p_mask
        n = int(mask.sum())
        counts[(fz_label, ia_label, p_label)] = n
        if n:
            chunk = df[mask].copy()
            chunk.insert(0, "P_bin_kPa", p_label)
            chunk.insert(0, "IA_bin_deg", ia_label)
            chunk.insert(0, "Fz_bin_N", fz_label)
            chunks.append(chunk)
    data = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=CELL_COLUMNS)
    return data, counts


# ----------------------------------------------------------------- collection
@dataclass
class ConditionResult:
    data: pd.DataFrame
    summary: pd.DataFrame
    notes: List[str]
    rows_read: int = 0
    rows_exported: int = 0       # rows written (may include overlap duplicates if nearest_only=False)
    rows_unique: int = 0         # unique source samples
    rows_outside: int = 0        # finite samples falling outside all bins
    files_used: int = 0


def _load_raw(run: RunConfig, spec: ConditionSpec, cache: Optional[dict]) -> pd.DataFrame:
    key = (os.path.abspath(run.filepath), run.test_type)
    if cache is not None and key in cache:
        raw = cache[key]
    else:
        raw = TireDataManager.load_unfiltered(run)
        raw["__row__"] = raw.index.astype(int)
        raw["__source__"] = os.path.basename(run.filepath)
        raw["__type__"] = run.test_type
        if cache is not None:
            cache[key] = raw
    df = raw.copy()
    core = [c for c in _CORE_CHANNELS[run.test_type] if c in df]
    # Keep rows where required channels exist (allow IA missing/NaN when no IA values at all)
    if run.test_type == "Cornering" and ("IA" not in df or not np.isfinite(df["IA"]).any()):
        core = ["SA", "FY", "FZ"]
    finite = np.isfinite(df[core]).all(axis=1)
    df = df[finite].copy()
    p_values = df["P"] if "P" in df else pd.Series(np.nan, index=df.index)
    df.insert(df.columns.get_loc("P") + 1 if "P" in df else len(df.columns), "P_kPa", p_values * spec.p_to_kpa)
    return df


def collect_condition_data(runs: Iterable[RunConfig], spec: ConditionSpec,
                           cache: Optional[dict] = None) -> ConditionResult:
    """Load unfiltered unique files and bin samples into Fz x IA x P cells."""
    notes: List[str] = []
    chunks: List[pd.DataFrame] = []
    totals: Dict[Tuple[str, str, str], int] = {}
    seen, rows_read, files_used, labels_used = set(), 0, 0, {"fz": set(), "ia": set(), "p": set()}

    for run in runs:
        key = (os.path.abspath(run.filepath), run.test_type)
        if key in seen:
            continue
        seen.add(key)
        name = os.path.basename(run.filepath)
        if run.test_type not in _CORE_CHANNELS:
            notes.append(f"{name}: skipped (unsupported test type '{run.test_type}').")
            continue
        try:
            df = _load_raw(run, spec, cache)
        except Exception as exc:
            notes.append(f"{name}: failed to load ({exc}).")
            continue

        files_used += 1
        rows_read += len(df)
        if not np.isfinite(df["P_kPa"]).any():
            notes.append(f"{name}: missing pressure channel P - assigned '{NO_DATA}' in P_bin_kPa.")
        if "IA" not in df or not np.isfinite(df["IA"]).any():
            notes.append(f"{name} ({run.test_type}): missing camber channel IA - assigned '{NO_DATA}' in IA_bin_deg.")

        data, counts = classify_frame(df, spec)
        for cell, n in counts.items():
            totals[cell] = totals.get(cell, 0) + n
        for fz_label, ia_label, p_label in counts:
            labels_used["fz"].add(fz_label); labels_used["ia"].add(ia_label); labels_used["p"].add(p_label)
        if len(data):
            chunks.append(data)

    data = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=CELL_COLUMNS)

    def order(bins: Sequence[Bin], used: set) -> List[str]:
        return [b.label for b in bins if b.label in used] + ([NO_DATA] if NO_DATA in used else [])

    fz_order = order(spec.fz_bins, labels_used["fz"]) or [b.label for b in spec.fz_bins]
    ia_order = order(spec.ia_bins, labels_used["ia"]) or [b.label for b in spec.ia_bins]
    p_order = order(spec.p_bins, labels_used["p"]) or [b.label for b in spec.p_bins]

    if len(data):
        for column, categories in zip(CELL_COLUMNS, (fz_order, ia_order, p_order)):
            data[column] = pd.Categorical(data[column], categories=categories, ordered=True)
        data = data.sort_values(CELL_COLUMNS, kind="stable").reset_index(drop=True)
        for column in CELL_COLUMNS:
            data[column] = data[column].astype(str)
        lead = [*CELL_COLUMNS, "__source__", "__type__", "__row__"]
        data = data[lead + [c for c in data.columns if c not in lead]]

    summary = pd.DataFrame(
        [(fz, ia, p, totals.get((fz, ia, p), 0)) for fz, ia, p in product(fz_order, ia_order, p_order)],
        columns=[*CELL_COLUMNS, "n_samples"],
    )

    unique = (data[["__source__", "__type__", "__row__"]].drop_duplicates().shape[0]) if len(data) else 0
    result = ConditionResult(
        data=data, summary=summary, notes=notes, rows_read=rows_read, rows_exported=len(data),
        rows_unique=unique, rows_outside=max(rows_read - unique, 0), files_used=files_used,
    )
    if result.rows_exported > result.rows_unique:
        extra = result.rows_exported - result.rows_unique
        notes.append(
            f"{extra} rows are duplicates from overlapping Fz/IA windows "
            f"(enable 'Nearest center only' to avoid duplicates)."
        )
    return result


def format_summary(result: ConditionResult) -> str:
    """Human-readable summary of non-empty cells and notes."""
    lines = [
        f"Files: {result.files_used} | samples read: {result.rows_read} | "
        f"exported rows: {result.rows_exported} (unique: {result.rows_unique}) | outside bins: {result.rows_outside}",
        "",
        "  Fz [N]   IA [deg]   P [kPa]    samples",
    ]
    filled = result.summary[result.summary["n_samples"] > 0]
    for row in filled.itertuples(index=False):
        lines.append(f"{row[0]:>8} {row[1]:>10} {row[2]:>10} {row[3]:>9}")
    if filled.empty:
        lines.append("  (no cell contains samples - check bin ranges and pressure unit)")
    empty = len(result.summary) - len(filled)
    if empty:
        lines.append(f"\nEmpty cells: {empty} of {len(result.summary)}")
    if result.notes:
        lines += ["", *[f"! {note}" for note in result.notes]]
    return "\n".join(lines)


def write_csv(path: str, result: ConditionResult, *, sep: str = ",") -> None:
    """Write binned matrix CSV."""
    if result.data.empty:
        raise ValueError("No samples inside selected condition bins - nothing to save.")
    result.data.to_csv(path, sep=sep, index=False)
