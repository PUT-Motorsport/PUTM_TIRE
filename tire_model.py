"""
tire_model.py — Tire physics & measurement data I/O layer.
=========================================================
Contains Magic Formula evaluation (Fy0, Fx0, Mz0), Load Sensitivity (Fy_load),
Combined Slip reduction factors (Gxa, Gyk), and resilient loading/filtering
for MATLAB (.mat) and ASCII/CSV (.csv, .txt, .dat, .tsv) tire test files.
"""

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import scipy.io

CHANNEL_ALIASES = {
    "SA": ["SA", "SLIP_ANGLE", "SLIPANGLE", "ALPHA", "ALPHA_DEG", "ALPHA_RAD", "SLIP_ANG", "SLIP_ANGLE_DEG"],
    "FY": ["FY", "LATERAL_FORCE", "LAT_FORCE", "LATFORCE", "FY_N", "FY_KN", "FY_SAE"],
    "FX": ["FX", "LONGITUDINAL_FORCE", "LONG_FORCE", "LONGFORCE", "FX_N", "FX_KN", "FX_SAE"],
    "FZ": [
        "FZ", "NORMAL_LOAD", "NORMAL_FORCE", "NORM_FORCE", "FZW",
        "VERTICAL_FORCE", "VERTICAL_LOAD", "VERT_FORCE", "VERT_LOAD",
        "WHEEL_LOAD", "LOAD", "FZ_N", "FZ_KN", "FZ_SAE",
    ],
    "IA": ["IA", "INCLINATION_ANGLE", "CAMBER", "CAMBER_ANGLE", "CAMBER_DEG", "IA_DEG", "GAMMA", "GAMMA_DEG"],
    "SL": ["SL", "LONG_SLIP", "LONGSIP", "SLIP_RATIO", "KAPPA", "SR"],
    "MZ": ["MZ", "ALIGNING_TORQUE", "ALIGNING_MOMENT", "SELF_ALIGNING_TORQUE", "MZ_NM"],
    "ET": ["ET", "TIME", "ELAPSED_TIME", "T"],
    "V": ["V", "SPEED", "VELOCITY", "VEL", "VX", "BELT_SPEED", "VEHICLE_SPEED", "CAR_SPEED"],
    "P": ["P", "PRESSURE", "INFLATION_PRESSURE", "PRESS", "TIRE_PRESSURE"],
    "TSTI": ["TSTI", "TIRE_TEMP_INNER", "TIRE_TEMP_I"],
    "TSTC": ["TSTC", "TIRE_TEMP_CENTER", "TIRE_TEMP_C"],
    "TSTO": ["TSTO", "TIRE_TEMP_OUTER", "TIRE_TEMP_O"],
}

_UNIT_STRIP_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)")
_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]+")
_UNIT_TOKEN_RE = re.compile(
    r"^(?:s|sec|ms|m/s|km/h|mph|kph|deg|rad|n|kn|lbf|nm|kpa|pa|psi|bar|c|f|k|-|%)$",
    re.IGNORECASE,
)


def _normalize_channel_key(name: str) -> str:
    """Normalize channel headers such as 'Slip Angle [deg]' -> 'SLIPANGLE'."""
    stripped = _UNIT_STRIP_RE.sub("", str(name)).upper()
    return _NON_ALNUM_RE.sub("", stripped)


def _extract_unit_hint(name: str) -> str:
    """Extract optional unit text from brackets like 'P [psi]' -> 'psi'."""
    m = re.search(r"[\[\(]([^\]\)]+)[\]\)]", str(name))
    return m.group(1).strip() if m else ""

# ==============================================================================
# 1. MODELE CZYSTEGO POŚLIZGU (PURE SLIP)
# ==============================================================================

def pacejka_fy(alpha, B, C, D, E, Sh, Sv):
    """Pure lateral Magic Formula; ``alpha`` and ``Sh`` are in radians."""
    alpha_eq = alpha + Sh
    return D * np.sin(
        C * np.arctan(B * alpha_eq - E * (B * alpha_eq - np.arctan(B * alpha_eq)))
    ) + Sv

def pacejka_fx(kappa, Bx, Cx, Dx, Ex, Shx, Svx):
    kappa_eq = kappa + Shx
    return Dx * np.sin(
        Cx * np.arctan(Bx * kappa_eq - Ex * (Bx * kappa_eq - np.arctan(Bx * kappa_eq)))
    ) + Svx

def pacejka_mz(alpha, Bz, Cz, Dz, Ez, Shz, Svz):
    """Simple aligning-moment curve; ``alpha`` and ``Shz`` are in radians."""
    alpha_eq = alpha + Shz
    return Dz * np.sin(
        Cz * np.arctan(Bz * alpha_eq - Ez * (Bz * alpha_eq - np.arctan(Bz * alpha_eq)))
    ) + Svz


# ==============================================================================
# 2. CZUŁOŚĆ NA OBCIĄŻENIE (LOAD SENSITIVITY - FIT 3D)
# ==============================================================================

def pacejka_fy_load(X, pCy1, pDy1, pDy2, pEy1, pKy1, pKy2, pUy1, pVy1, fz0=1000.0):
    """Load-sensitive lateral model; alpha is in radians and Fz is positive."""
    alpha, fz = X
    dfz = (fz - fz0) / fz0

    mu_y = pDy1 + pDy2 * dfz
    D = mu_y * fz
    C = pCy1
    E = pEy1

    Ky = pKy1 * fz0 * np.sin(2.0 * np.arctan(fz / (pKy2 * fz0)))
    B = Ky / (C * D + 1e-6) 

    Sh = pUy1
    Sv = pVy1 * fz

    alpha_eq = alpha + Sh
    return D * np.sin(C * np.arctan(B * alpha_eq - E * (B * alpha_eq - np.arctan(B * alpha_eq)))) + Sv


# ==============================================================================
# 3. COMBINED SLIP
# ==============================================================================

def pacejka_comb_Gxa(alpha, rBx1, rCx1):
    """Longitudinal force reduction factor Gxa under slip angle alpha."""
    return np.cos(rCx1 * np.arctan(rBx1 * alpha))

def pacejka_comb_Gyk(kappa, rBy1, rCy1):
    """Lateral force reduction factor Gyk under longitudinal slip kappa (SL)."""
    return np.cos(rCy1 * np.arctan(rBy1 * kappa))

def combined_slip_fy(alpha, kappa, B, C, D, E, Sh, Sv, rBy1, rCy1):
    """Evaluate combined lateral force Fy under longitudinal slip kappa."""
    fy0 = pacejka_fy(alpha, B, C, D, E, Sh, Sv)
    gyk = pacejka_comb_Gyk(kappa, rBy1, rCy1)
    return fy0 * gyk

def combined_slip_fx(kappa, alpha, Bx, Cx, Dx, Ex, Shx, Svx, rBx1, rCx1):
    """Evaluate combined longitudinal force Fx under slip angle alpha."""
    fx0 = pacejka_fx(kappa, Bx, Cx, Dx, Ex, Shx, Svx)
    gxa = pacejka_comb_Gxa(alpha, rBx1, rCx1)
    return fx0 * gxa


# ==============================================================================
# DATA STRUCTURES & ROBUST FILE IMPORTERS (.MAT / .CSV / .TXT / .DAT)
# ==============================================================================

@dataclass
class RunConfig:
    filepath: str
    test_type: str
    target_fz: float = 0.0
    tol_fz: float = 0.0
    target_param: float = 0.0
    tol_param: float = 0.0
    n_points: int = 0
    target_sa: Optional[float] = None
    tol_sa: Optional[float] = None
    target_ia: Optional[float] = None
    tol_ia: Optional[float] = None

    def __post_init__(self):
        if self.tol_fz < 0 or self.tol_param < 0:
            raise ValueError("Tolerances cannot be negative.")
        if (self.target_sa is None) != (self.tol_sa is None):
            raise ValueError("SA filter requires both a target value and a tolerance.")
        if (self.target_ia is None) != (self.tol_ia is None):
            raise ValueError("IA filter requires both a target value and a tolerance.")
        if self.tol_sa is not None and self.tol_sa < 0:
            raise ValueError("SA tolerance cannot be negative.")
        if self.tol_ia is not None and self.tol_ia < 0:
            raise ValueError("IA tolerance cannot be negative.")


def _unwrap_mat_value(val: Any) -> Any:
    """Unwrap nested 1-element object/cell containers produced by scipy.io.loadmat."""
    while isinstance(val, np.ndarray) and val.dtype == object and val.size == 1:
        inner = val.flat[0]
        if inner is val:
            break
        val = inner
    return val


def _coerce_numeric_1d(val: Any) -> Optional[np.ndarray]:
    """Safely convert MATLAB/CSV arrays to a 1-D float ndarray; return None if non-numeric."""
    val = _unwrap_mat_value(val)
    if val is None or isinstance(val, (str, bytes)):
        return None
    arr = np.asarray(val)
    if arr.dtype.names:
        return None
    if arr.dtype.kind in ("U", "S"):
        return None
    if arr.dtype == object:
        try:
            flat = [float(x) if x is not None else np.nan for x in arr.reshape(-1)]
            return np.asarray(flat, dtype=float)
        except Exception:
            return None
    try:
        out = np.asarray(arr, dtype=float).reshape(-1)
        return out
    except Exception:
        try:
            series = pd.to_numeric(pd.Series(arr.reshape(-1)), errors="coerce")
            return series.to_numpy(dtype=float)
        except Exception:
            return None


class TireDataManager:
    @staticmethod
    def _expand_mat_mapping(mat_data, _depth: int = 0) -> dict:
        flat: Dict[str, Any] = {}
        if _depth > 5 or not isinstance(mat_data, dict):
            return flat
        for key, val in mat_data.items():
            if str(key).startswith("__") and str(key) != "__unit_hints__":
                continue
            if str(key) == "__unit_hints__":
                flat["__unit_hints__"] = val
                continue
            val_u = _unwrap_mat_value(val)
            flat[key] = val_u
            if isinstance(val_u, np.ndarray) and val_u.dtype.names and val_u.size >= 1:
                rec = val_u.flat[0]
                nested = {field_name: rec[field_name] for field_name in val_u.dtype.names}
                expanded_nested = TireDataManager._expand_mat_mapping(nested, _depth + 1)
                for n_k, n_v in expanded_nested.items():
                    flat.setdefault(n_k, n_v)
            elif isinstance(val_u, dict):
                expanded_nested = TireDataManager._expand_mat_mapping(val_u, _depth + 1)
                for n_k, n_v in expanded_nested.items():
                    flat.setdefault(n_k, n_v)
        return flat

    @staticmethod
    def _read_delimited_lines(filepath: str) -> List[str]:
        last_exc: Optional[Exception] = None
        for enc in ("utf-8-sig", "utf-8", "cp1250", "latin-1"):
            try:
                with open(filepath, "r", encoding=enc) as f:
                    return f.read().splitlines()
            except UnicodeDecodeError as exc:
                last_exc = exc
                continue
        if last_exc is not None:
            with open(filepath, "r", encoding="latin-1", errors="replace") as f:
                return f.read().splitlines()
        return []

    @staticmethod
    def _load_delimited_dict(filepath: str) -> dict:
        lines = TireDataManager._read_delimited_lines(filepath)
        non_empty = [(idx, ln) for idx, ln in enumerate(lines) if ln.strip() and not ln.strip().startswith(("#", "!", "//"))]
        if not non_empty:
            raise ValueError(f"File is empty or contains only comments: {os.path.basename(filepath)}")

        # Find best header row (handle Calspan TTC / MoTeC metadata preamble lines)
        header_pos = 0
        sep_choice: Optional[str] = None
        alias_norm_set = {
            _normalize_channel_key(a)
            for alist in CHANNEL_ALIASES.values()
            for a in alist
        }
        best_score = -1
        for pos, (_line_idx, ln) in enumerate(non_empty[:45]):
            for cand_sep in (";", "\t", ",", None):
                parts = ln.split() if cand_sep is None else [p.strip().strip('"').strip("'") for p in ln.split(cand_sep)]
                if len(parts) < 3:
                    continue
                alpha_cnt = sum(1 for p in parts if any(ch.isalpha() for ch in p))
                alias_hits = sum(1 for p in parts if _normalize_channel_key(p) in alias_norm_set)
                score = alias_hits * 10 + (alpha_cnt if alias_hits > 0 else 0)
                if score > best_score:
                    best_score = score
                    header_pos = pos
                    sep_choice = cand_sep

        header_line_idx = non_empty[header_pos][0]
        raw_text = "\n".join(lines[header_line_idx:])
        from io import StringIO

        if sep_choice is None:
            df_csv = pd.read_csv(StringIO(raw_text), sep=r"\s+", engine="python", comment="#")
        else:
            df_csv = pd.read_csv(StringIO(raw_text), sep=sep_choice, engine="python")

        if df_csv.empty or df_csv.shape[1] < 2:
            df_csv = pd.read_csv(filepath, sep=None, engine="python")

        unit_hints: Dict[str, str] = {}
        # Drop units sub-header row if first data row consists mainly of units like deg/N/kPa/psi
        if not df_csv.empty:
            first_row_vals = [str(v).strip() for v in df_csv.iloc[0].tolist()]
            unit_hits = sum(1 for v in first_row_vals if _UNIT_TOKEN_RE.match(v))
            non_numeric_hits = 0
            for v in first_row_vals:
                try:
                    float(v.replace(",", "."))
                except ValueError:
                    non_numeric_hits += 1
            if unit_hits >= 2 or (non_numeric_hits >= max(2, len(first_row_vals) // 2)):
                for col_name, unit_val in zip(df_csv.columns, first_row_vals):
                    if unit_val and _UNIT_TOKEN_RE.match(unit_val):
                        unit_hints[str(col_name).strip()] = unit_val
                df_csv = df_csv.iloc[1:].reset_index(drop=True)

        out: Dict[str, Any] = {}
        for col in df_csv.columns:
            col_str = str(col).strip()
            inline_unit = _extract_unit_hint(col_str)
            if inline_unit and col_str not in unit_hints:
                unit_hints[col_str] = inline_unit
            series = df_csv[col]
            if not pd.api.types.is_numeric_dtype(series):
                # Support European decimal commas (e.g. "-1,52" inside semicolon/tab CSVs)
                cleaned = series.astype(str).str.strip().str.replace(",", ".", regex=False)
                out[col_str] = pd.to_numeric(cleaned, errors="coerce").to_numpy(dtype=float)
            else:
                out[col_str] = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        out["__unit_hints__"] = unit_hints
        return out

    @staticmethod
    def _load_h5_group(group, out: Dict[str, Any], prefix: str = "") -> None:
        for key in group.keys():
            if str(key).startswith("#"):
                continue
            item = group[key]
            full_key = f"{prefix}_{key}" if prefix else key
            if hasattr(item, "keys"):
                TireDataManager._load_h5_group(item, out, full_key)
            else:
                arr = np.asarray(item)
                out[key] = arr
                if prefix:
                    out.setdefault(full_key, arr)

    @staticmethod
    def _load_source_dict(filepath: str) -> dict:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        ext = os.path.splitext(filepath)[1].lower()
        if ext in (".csv", ".txt", ".tsv", ".dat", ".asc"):
            return TireDataManager._load_delimited_dict(filepath)
        try:
            raw_mat = scipy.io.loadmat(filepath)
        except NotImplementedError as exc:
            try:
                import h5py  # optional support for MATLAB v7.3 HDF5 files
            except ImportError as h5_exc:
                raise ValueError(
                    "This .mat file uses MATLAB v7.3 (HDF5) format, which requires h5py. "
                    "Either save it in MATLAB with '-v7' or export to CSV/TXT."
                ) from h5_exc
            h5_out: Dict[str, Any] = {}
            with h5py.File(filepath, "r") as hf:
                TireDataManager._load_h5_group(hf, h5_out)
            return h5_out
        return TireDataManager._expand_mat_mapping(raw_mat)

    @staticmethod
    def find_channel(mat_data, canonical_name) -> Optional[np.ndarray]:
        expanded = TireDataManager._expand_mat_mapping(mat_data)
        upper_keys = {str(k).upper(): k for k in expanded.keys() if not str(k).startswith("__")}
        norm_keys: Dict[str, str] = {}
        for k in expanded.keys():
            if str(k).startswith("__"):
                continue
            norm_keys.setdefault(_normalize_channel_key(str(k)), k)

        for alias in CHANNEL_ALIASES[canonical_name]:
            raw_key = None
            if alias.upper() in upper_keys:
                raw_key = upper_keys[alias.upper()]
            elif _normalize_channel_key(alias) in norm_keys:
                raw_key = norm_keys[_normalize_channel_key(alias)]
            if raw_key is not None:
                arr = _coerce_numeric_1d(expanded[raw_key])
                if arr is not None and arr.size > 0:
                    return arr
        return None

    @classmethod
    def _build_frame(cls, mat_data, required, optional):
        channels = {name: cls.find_channel(mat_data, name) for name in required + optional}
        missing = [name for name in required if channels[name] is None]
        if missing:
            available = sorted(str(key) for key in mat_data if not str(key).startswith("__"))
            raise KeyError(
                f"Missing required channels {missing}. Available channels: {available}"
            )

        expected_length = max(len(channels[name]) for name in required)
        warnings_list: List[str] = []

        # Validate required lengths (allow len-1 broadcast if another required vector is longer)
        mismatched_req: Dict[str, int] = {}
        for name in required:
            vec = channels[name]
            if len(vec) == expected_length:
                continue
            if len(vec) == 1 and expected_length > 1:
                channels[name] = np.full(expected_length, float(vec[0]), dtype=float)
                warnings_list.append(f"Broadcast scalar channel {name}={float(vec[0]):g} to {expected_length} rows.")
            else:
                mismatched_req[name] = len(vec)
        if mismatched_req:
            raise ValueError(
                f"Required channel length mismatch: expected {expected_length}, got {mismatched_req}"
            )

        # Normalize optional channels: broadcast length-1 scalars, gracefully drop invalid lengths
        for name in optional:
            vec = channels[name]
            if vec is None or len(vec) == expected_length:
                continue
            if len(vec) == 1:
                channels[name] = np.full(expected_length, float(vec[0]), dtype=float)
                warnings_list.append(f"Broadcast constant {name}={float(vec[0]):g} across {expected_length} samples.")
            else:
                warnings_list.append(
                    f"Ignored optional channel '{name}' (length {len(vec)} != {expected_length})."
                )
                channels[name] = None

        values = {
            name: np.asarray(channel, dtype=float) if channel is not None else np.full(expected_length, np.nan)
            for name, channel in channels.items()
        }
        df = pd.DataFrame(values)
        # Fail clearly if a required channel became 100% NaN (e.g., bad delimiter / corrupt export)
        empty_req = [name for name in required if not np.isfinite(df[name]).any()]
        if empty_req:
            raise ValueError(
                f"Required channel(s) {empty_req} contain no finite numeric values. "
                "Check delimiter, decimal format, or file structure."
            )

        df.attrs["load_warnings"] = warnings_list
        if "__unit_hints__" in mat_data and isinstance(mat_data["__unit_hints__"], dict):
            df.attrs["unit_hints"] = dict(mat_data["__unit_hints__"])
        return df

    @classmethod
    def load_unfiltered(cls, run: RunConfig) -> pd.DataFrame:
        """Read available measurement channels without applying UI filters."""
        mat_data = cls._load_source_dict(run.filepath)
        if run.test_type == "Cornering":
            df = cls._build_frame(
                mat_data,
                required=["SA", "FY", "FZ"],
                optional=["IA", "MZ", "FX", "SL", "ET", "V", "P", "TSTI", "TSTC", "TSTO"],
            )
        elif run.test_type == "Drive/Brake":
            required = ["FX", "SL", "FZ"]
            if run.target_sa is not None:
                required.append("SA")
            if run.target_ia is not None:
                required.append("IA")
            optional = [name for name in ["FY", "MZ", "SA", "IA", "ET", "V", "P", "TSTI", "TSTC", "TSTO"] if name not in required]
            df = cls._build_frame(mat_data, required=required, optional=optional)
        else:
            raise ValueError(f"Unsupported test type: {run.test_type}")
        load_warnings = list(df.attrs.get("load_warnings", []))
        unit_hints = dict(df.attrs.get("unit_hints", {}))
        df["FZ_RAW"] = df["FZ"]
        df["FZ"] = np.abs(df["FZ"])
        df.attrs["load_warnings"] = load_warnings
        df.attrs["unit_hints"] = unit_hints
        return df

    @classmethod
    def load_and_filter(cls, run: RunConfig) -> pd.DataFrame:
        df = cls.load_unfiltered(run)

        if run.test_type == "Cornering":
            has_ia = "IA" in df and np.isfinite(df["IA"]).any()
            if has_ia:
                finite = np.isfinite(df[["SA", "FY", "FZ", "IA"]]).all(axis=1)
                filtered = df[
                    finite
                    & (np.abs(df["IA"] - run.target_param) <= run.tol_param)
                    & (np.abs(df["FZ"] - abs(run.target_fz)) <= run.tol_fz)
                ].copy()
            else:
                finite = np.isfinite(df[["SA", "FY", "FZ"]]).all(axis=1)
                filtered = df[
                    finite & (np.abs(df["FZ"] - abs(run.target_fz)) <= run.tol_fz)
                ].copy()

        elif run.test_type == "Drive/Brake":
            finite = np.isfinite(df[["FX", "SL", "FZ"]]).all(axis=1)
            selection = (
                finite
                & (np.abs(df["SL"] - run.target_param) <= run.tol_param)
                & (np.abs(df["FZ"] - abs(run.target_fz)) <= run.tol_fz)
            )
            if run.target_sa is not None and "SA" in df and np.isfinite(df["SA"]).any():
                selection &= np.abs(df["SA"] - run.target_sa) <= float(run.tol_sa)
            if run.target_ia is not None and "IA" in df and np.isfinite(df["IA"]).any():
                selection &= np.abs(df["IA"] - run.target_ia) <= float(run.tol_ia)
            filtered = df[selection].copy()

        filtered["__row__"] = filtered.index.astype(int)
        filtered["__source__"] = os.path.basename(run.filepath)
        filtered["__source_path__"] = os.path.abspath(run.filepath)
        filtered["__type__"] = run.test_type
        filtered["__target_fz__"] = abs(run.target_fz)
        filtered["__target_ia__"] = run.target_param if run.test_type == "Cornering" else run.target_ia
        filtered["__target_sa__"] = np.nan if run.test_type == "Cornering" else run.target_sa
        return filtered
