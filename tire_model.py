"""
tire_model.py — Warstwa logiki i obliczeń modelu opony
=====================================================
Zawiera algorytmy Pacejki (Fy0, Fx0, Mz0), logikę czułości na obciążenie (Load Sensitivity),
współczynniki poślizgu łączonego (Combined Slip), oraz rozszerzoną obsługę odczytu i filtorwania .mat.
"""

import os
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
import scipy.io

CHANNEL_ALIASES = {
    "SA": ["SA", "SLIP_ANGLE", "SLIPANGLE"],
    "FY": ["FY", "LATERAL_FORCE"],
    "FX": ["FX", "LONGITUDINAL_FORCE", "LONGFORCE"],
    "FZ": ["FZ", "NORMAL_LOAD", "FZW"],
    "IA": ["IA", "INCLINATION_ANGLE", "CAMBER"],
    "SL": ["SL", "LONG_SLIP", "LONGSIP", "KAPPA"],
    "MZ": ["MZ", "ALIGNING_TORQUE", "ALIGNING_MOMENT"], 
    "ET": ["ET", "TIME", "ELAPSED_TIME"],
    "V": ["V", "SPEED", "VELOCITY"],
    "P": ["P", "PRESSURE", "INFLATION_PRESSURE"],
    "TSTI": ["TSTI", "TIRE_TEMP_INNER"],
    "TSTC": ["TSTC", "TIRE_TEMP_CENTER"],
    "TSTO": ["TSTO", "TIRE_TEMP_OUTER"],
}

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
# 3. POŚLIZG ŁĄCZONY (COMBINED SLIP)
# ==============================================================================

def pacejka_comb_Gxa(alpha, rBx1, rCx1):
    """Współczynnik redukcji siły wzdłużnej Fx przy występowaniu kąta znoszenia alpha."""
    return np.cos(rCx1 * np.arctan(rBx1 * alpha))

def pacejka_comb_Gyk(kappa, rBy1, rCy1):
    """Współczynnik redukcji siły bocznej Fy przy występowaniu poślizgu wzdłużnego kappa (SL)."""
    return np.cos(rCy1 * np.arctan(rBy1 * kappa))

def combined_slip_fy(alpha, kappa, B, C, D, E, Sh, Sv, rBy1, rCy1):
    """Oblicza wypadkową siłę boczną Fy z uwzględnieniem poślizgu wzdłużnego kappa."""
    fy0 = pacejka_fy(alpha, B, C, D, E, Sh, Sv)
    gyk = pacejka_comb_Gyk(kappa, rBy1, rCy1)
    return fy0 * gyk

def combined_slip_fx(kappa, alpha, Bx, Cx, Dx, Ex, Shx, Svx, rBx1, rCx1):
    """Oblicza wypadkową siłę wzdłużną Fx z uwzględnieniem kąta znoszenia alpha."""
    fx0 = pacejka_fx(kappa, Bx, Cx, Dx, Ex, Shx, Svx)
    gxa = pacejka_comb_Gxa(alpha, rBx1, rCx1)
    return fx0 * gxa


# ==============================================================================
# STRUKTURY DANYCH I OBSŁUGA PLIKÓW MAT
# ==============================================================================

@dataclass
class RunConfig:
    filepath: str
    test_type: str       
    target_fz: float     
    tol_fz: float        
    target_param: float  
    tol_param: float     
    n_points: int = 0
    target_sa: Optional[float] = None
    tol_sa: Optional[float] = None
    target_ia: Optional[float] = None
    tol_ia: Optional[float] = None

    def __post_init__(self):
        if self.tol_fz < 0 or self.tol_param < 0:
            raise ValueError("Tolerancje nie mogą być ujemne.")
        if (self.target_sa is None) != (self.tol_sa is None):
            raise ValueError("Filtr SA wymaga jednocześnie wartości docelowej i tolerancji.")
        if (self.target_ia is None) != (self.tol_ia is None):
            raise ValueError("Filtr IA wymaga jednocześnie wartości docelowej i tolerancji.")
        if self.tol_sa is not None and self.tol_sa < 0:
            raise ValueError("Tolerancja SA nie może być ujemna.")
        if self.tol_ia is not None and self.tol_ia < 0:
            raise ValueError("Tolerancja IA nie może być ujemna.")


class TireDataManager:
    @staticmethod
    def find_channel(mat_data, canonical_name):
        keys = {k.upper(): k for k in mat_data.keys() if not k.startswith("__")}
        for alias in CHANNEL_ALIASES[canonical_name]:
            if alias.upper() in keys:
                return mat_data[keys[alias.upper()]].flatten()
        return None

    @classmethod
    def _build_frame(cls, mat_data, required, optional):
        channels = {name: cls.find_channel(mat_data, name) for name in required + optional}
        missing = [name for name in required if channels[name] is None]
        if missing:
            available = sorted(key for key in mat_data if not key.startswith("__"))
            raise KeyError(f"Brak wymaganych kanałów {missing}. Dostępne kanały: {available}")

        expected_length = len(channels[required[0]])
        mismatched = {
            name: len(values)
            for name, values in channels.items()
            if values is not None and len(values) != expected_length
        }
        if mismatched:
            raise ValueError(f"Kanały mają różne długości: oczekiwano {expected_length}, otrzymano {mismatched}")

        values = {
            name: channel if channel is not None else np.full(expected_length, np.nan)
            for name, channel in channels.items()
        }
        return pd.DataFrame(values)

    @classmethod
    def load_and_filter(cls, run: RunConfig) -> pd.DataFrame:
        mat_data = scipy.io.loadmat(run.filepath)

        if run.test_type == "Cornering":
            df = cls._build_frame(
                mat_data,
                required=["SA", "FY", "FZ", "IA"],
                optional=["MZ", "FX", "SL", "ET", "V", "P", "TSTI", "TSTC", "TSTO"],
            )
            df["FZ_RAW"] = df["FZ"]
            df["FZ"] = np.abs(df["FZ"])
            finite = np.isfinite(df[["SA", "FY", "FZ", "IA"]]).all(axis=1)
            filtered = df[
                finite
                & (np.abs(df["IA"] - run.target_param) <= run.tol_param)
                & (np.abs(df["FZ"] - abs(run.target_fz)) <= run.tol_fz)
            ].copy()

        elif run.test_type == "Drive/Brake":
            required = ["FX", "SL", "FZ"]
            if run.target_sa is not None:
                required.append("SA")
            if run.target_ia is not None:
                required.append("IA")
            optional = [name for name in ["FY", "MZ", "SA", "IA", "ET", "V", "P", "TSTI", "TSTC", "TSTO"] if name not in required]
            df = cls._build_frame(mat_data, required=required, optional=optional)
            df["FZ_RAW"] = df["FZ"]
            df["FZ"] = np.abs(df["FZ"])
            finite = np.isfinite(df[["FX", "SL", "FZ"]]).all(axis=1)
            selection = (
                finite
                & (np.abs(df["SL"] - run.target_param) <= run.tol_param)
                & (np.abs(df["FZ"] - abs(run.target_fz)) <= run.tol_fz)
            )
            if run.target_sa is not None:
                selection &= np.abs(df["SA"] - run.target_sa) <= float(run.tol_sa)
            if run.target_ia is not None:
                selection &= np.abs(df["IA"] - run.target_ia) <= float(run.tol_ia)
            filtered = df[
                selection
            ].copy()
        else:
            raise ValueError(f"Nieobsługiwany typ testu: {run.test_type}")

        filtered["__row__"] = filtered.index.astype(int)
        filtered["__source__"] = os.path.basename(run.filepath)
        filtered["__source_path__"] = os.path.abspath(run.filepath)
        filtered["__type__"] = run.test_type
        filtered["__target_fz__"] = abs(run.target_fz)
        filtered["__target_ia__"] = run.target_param if run.test_type == "Cornering" else run.target_ia
        filtered["__target_sa__"] = np.nan if run.test_type == "Cornering" else run.target_sa
        return filtered
