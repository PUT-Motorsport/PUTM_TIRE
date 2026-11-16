"""Pure-slip and simplified Magic Formula curve generator (no Tk dependencies)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from tire_model import pacejka_fx, pacejka_fy, pacejka_fy_load, pacejka_mz

CHANNELS: Dict[str, dict] = {
    "Fy": {"func": pacejka_fy, "slip_unit": "rad", "force_unit": "N", "axis_label": "Slip Angle SA [deg]", "default_range": 15.0},
    "Fx": {"func": pacejka_fx, "slip_unit": "ratio", "force_unit": "N", "axis_label": "Longitudinal Slip SL [-]", "default_range": 0.30},
    "Mz": {"func": pacejka_mz, "slip_unit": "rad", "force_unit": "Nm", "axis_label": "Slip Angle SA [deg]", "default_range": 15.0},
}


@dataclass(frozen=True)
class SimpleMagicFormula:
    """y = D*sin(C*atan(B*x' - E*(B*x' - atan(B*x')))) + Sv,  x' = x + Sh."""

    channel: str
    B: float
    C: float
    D: float
    E: float
    Sh: float = 0.0  # rad for Fy/Mz, ratio for Fx
    Sv: float = 0.0

    def __post_init__(self):
        if self.channel not in CHANNELS:
            raise ValueError(f"Unknown channel '{self.channel}'. Available: {sorted(CHANNELS)}")

    @property
    def label(self) -> str:
        return f"{self.channel}0"

    def __call__(self, slip):
        func = CHANNELS[self.channel]["func"]
        return func(np.asarray(slip, dtype=float), self.B, self.C, self.D, self.E, self.Sh, self.Sv)

    def as_dict(self) -> Dict[str, object]:
        info = CHANNELS[self.channel]
        names = {"Fy": "Fy0", "Fx": "Fx0", "Mz": "Mz0"}
        return {
            "model": f"{names[self.channel]}_manual",
            "B": float(self.B), "C": float(self.C), "D": float(self.D),
            "E": float(self.E), "Sh": float(self.Sh), "Sv": float(self.Sv),
            "slip_unit": info["slip_unit"], "force_unit": info["force_unit"],
            "source": "simple_pacejka",
        }


def from_physical(channel: str, *, mu: float, fz: float, stiffness: float,
                  C: float = 1.3, E: float = -0.5, Sh: float = 0.0, Sv: float = 0.0) -> SimpleMagicFormula:
    """Build a simplified Magic Formula model from physical tire properties."""
    if channel == "Mz":
        raise ValueError("For Mz, specify B, C, D, E directly (D is in Nm).")
    if mu <= 0 or fz <= 0 or stiffness <= 0 or C <= 0:
        raise ValueError("mu, Fz, stiffness, and C must all be positive.")
    d = mu * fz
    stiffness_model = stiffness * (180.0 / np.pi) if CHANNELS[channel]["slip_unit"] == "rad" else stiffness
    return SimpleMagicFormula(channel, B=stiffness_model / (C * d), C=C, D=d, E=E, Sh=Sh, Sv=Sv)


def slip_axis(channel: str, half_range: float | None = None, n: int = 801) -> Tuple[np.ndarray, np.ndarray]:
    """Return (display_axis, model_axis). Fy/Mz: degrees -> radians."""
    info = CHANNELS[channel]
    half = float(half_range if half_range else info["default_range"])
    if half <= 0:
        raise ValueError("Slip axis half-range must be positive.")
    shown = np.linspace(-half, half, int(n))
    model_axis = np.deg2rad(shown) if info["slip_unit"] == "rad" else shown
    return shown, model_axis


def evaluate(model: SimpleMagicFormula, half_range: float | None = None, n: int = 801):
    shown, model_axis = slip_axis(model.channel, half_range, n)
    return shown, model(model_axis)


def summarize(model: SimpleMagicFormula, half_range: float | None = None) -> Dict[str, object]:
    """Peak force, slip at peak, initial stiffness, and parameter warnings."""
    shown, force = evaluate(model, half_range)
    eps = 1e-4
    slope_model = float((model(eps) - model(-eps)) / (2 * eps))

    idx = int(np.argmax(force)) if slope_model >= 0 else int(np.argmin(force))
    peak_at_edge = idx in (0, len(force) - 1)
    is_angle = CHANNELS[model.channel]["slip_unit"] == "rad"
    slope_shown = slope_model * (np.pi / 180.0) if is_angle else slope_model

    warnings: List[str] = []
    if model.B <= 0:
        warnings.append("B <= 0: curve has inverted initial stiffness.")
    if model.D == 0:
        warnings.append("D = 0: curve is completely flat.")
    elif model.D < 0:
        warnings.append("D < 0: inverted curve (valid only if matching input sign convention).")
    if not 1.0 <= model.C <= 2.5:
        warnings.append("C outside typical 1.0–2.5 range (changes curve shape).")
    if model.E > 1.0:
        warnings.append("E > 1: outside standard physical E <= 1 limit.")
    if peak_at_edge:
        warnings.append("Peak lies on axis boundary: increase slip axis half-range.")

    return {
        "peak_force": float(force[idx]),
        "slip_at_peak": float(shown[idx]),
        "initial_stiffness": slope_shown,
        "stiffness_unit": ("N/deg" if model.channel == "Fy" else "Nm/deg") if is_angle else "N/(1.0 slip)",
        "force_unit": CHANNELS[model.channel]["force_unit"],
        "warnings": warnings,
    }


LOAD_PARAMS = ("pCy1", "pDy1", "pDy2", "pEy1", "pKy1", "pKy2", "pUy1", "pVy1")


@dataclass(frozen=True)
class FrictionVsFz:
    """Load-sensitive peak friction model: mu = pD*1 + pD*2 * dfz."""
    pDy1: float = 1.50
    pDy2: float = -0.15
    pDx1: float = 1.65
    pDx2: float = -0.18
    fz0: float = 1000.0

    def __post_init__(self):
        if self.fz0 <= 0:
            raise ValueError("Fz0 must be positive.")

    @property
    def label(self) -> str:
        return "mu_vs_fz"

    def evaluate(self, fz_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        fz = np.asarray(fz_array, dtype=float)
        dfz = (fz - self.fz0) / self.fz0
        mu_y = np.maximum(self.pDy1 + self.pDy2 * dfz, 0.0)
        mu_x = np.maximum(self.pDx1 + self.pDx2 * dfz, 0.0)
        Dy = mu_y * fz
        Dx = mu_x * fz
        return mu_y, mu_x, Dy, Dx

    def as_dict(self) -> Dict[str, object]:
        return {
            "model": "mu_vs_fz",
            "pDy1": float(self.pDy1),
            "pDy2": float(self.pDy2),
            "pDx1": float(self.pDx1),
            "pDx2": float(self.pDx2),
            "fz0": float(self.fz0),
            "source": "simple_pacejka",
        }


def summarize_friction(model: FrictionVsFz, fz_max: float = 2200.0) -> Dict[str, object]:
    eval_points = [400.0, 800.0, 1000.0, 1400.0, 1800.0, float(fz_max)]
    eval_points = sorted(set(eval_points))
    mu_y, mu_x, Dy, Dx = model.evaluate(np.array(eval_points))
    rows = []
    for fz, my, mx, dy, dx in zip(eval_points, mu_y, mu_x, Dy, Dx):
        rows.append({
            "fz": fz,
            "mu_y": float(my),
            "mu_x": float(mx),
            "Dy": float(dy),
            "Dx": float(dx),
        })
    warnings = []
    if model.pDy2 >= 0:
        warnings.append("pDy2 >= 0: lateral mu increases with load Fz.")
    if model.pDx2 >= 0:
        warnings.append("pDx2 >= 0: longitudinal mu increases with load Fz.")
    return {"rows": rows, "warnings": warnings}


@dataclass(frozen=True)
class LoadSensitiveFy:
    """Load-sensitive Fy(alpha, Fz) model (alpha in radians, positive Fz)."""

    pCy1: float
    pDy1: float
    pDy2: float
    pEy1: float
    pKy1: float
    pKy2: float
    pUy1: float = 0.0
    pVy1: float = 0.0
    fz0: float = 1000.0

    channel = "Fy"

    def __post_init__(self):
        if self.fz0 <= 0:
            raise ValueError("Fz0 must be positive.")
        if self.pKy2 <= 0:
            raise ValueError("pKy2 must be positive.")

    @property
    def label(self) -> str:
        return "Fy_load"

    def __call__(self, alpha, fz):
        alpha = np.asarray(alpha, dtype=float)
        fz = np.abs(np.asarray(fz, dtype=float))
        values = [getattr(self, name) for name in LOAD_PARAMS]
        return pacejka_fy_load((alpha, fz), *values, fz0=self.fz0)

    def as_dict(self) -> Dict[str, object]:
        return {
            "model": "Fy_load_manual",
            **{name: float(getattr(self, name)) for name in LOAD_PARAMS},
            "fz0": float(self.fz0),
            "slip_unit": "rad", "force_unit": "N", "source": "simple_pacejka",
        }


def load_family(model: LoadSensitiveFy, fz_levels, half_range: float | None = None, n: int = 801):
    """Fy(SA) curve family across vertical load levels."""
    levels = [float(v) for v in fz_levels]
    if not levels or any(v <= 0 for v in levels):
        raise ValueError("Provide at least one positive Fz load level.")
    shown, model_axis = slip_axis("Fy", half_range, n)
    return shown, {fz: model(model_axis, np.full_like(model_axis, fz)) for fz in levels}


def summarize_family(model: LoadSensitiveFy, fz_levels, half_range: float | None = None) -> Dict[str, object]:
    shown, family = load_family(model, fz_levels, half_range)
    rows, edge_hit = [], False
    for fz, force in family.items():
        idx = int(np.argmax(force))
        edge_hit = edge_hit or idx in (0, len(force) - 1)
        eps = 1e-4
        ky = float((model(eps, fz) - model(-eps, fz)) / (2 * eps)) * (np.pi / 180.0)
        peak = float(force[idx])
        rows.append({"fz": fz, "peak_force": peak, "slip_at_peak": float(shown[idx]),
                     "mu_peak": peak / fz, "stiffness_n_per_deg": ky})
    warnings: List[str] = []
    if model.pDy2 >= 0:
        warnings.append("pDy2 >= 0: mu increases with Fz (tires normally lose mu at higher load).")
    if not 1.0 <= model.pCy1 <= 2.5:
        warnings.append("pCy1 outside typical 1.0–2.5 range.")
    if model.pEy1 > 1.0:
        warnings.append("pEy1 > 1: outside standard E <= 1 limit.")
    if edge_hit:
        warnings.append("Peak lies on axis boundary for one or more loads: increase slip axis range.")
    return {"rows": rows, "warnings": warnings}


def parse_fz_levels(text: str, max_levels: int = 8) -> List[float]:
    """Parse Fz level list separated by commas, semicolons, or spaces."""
    parts = [p for p in text.replace(";", ",").replace(" ", ",").split(",") if p.strip()]
    try:
        levels = [float(p.replace(",", ".")) for p in parts]
    except ValueError:
        raise ValueError("Fz levels must be numeric, e.g. 600, 800, 1000, 1200.")
    if not levels or any(v <= 0 for v in levels):
        raise ValueError("Provide at least one positive Fz load level.")
    if len(levels) > max_levels:
        raise ValueError(f"Maximum {max_levels} Fz load levels allowed.")
    return levels


_PURE_FIT_KEYS = {
    "Fy0": ("Fy", ("B", "C", "D", "E", "Sh", "Sv")),
    "Fx0": ("Fx", ("Bx", "Cx", "Dx", "Ex", "Shx", "Svx")),
    "Mz0": ("Mz", ("Bz", "Cz", "Dz", "Ez", "Shz", "Svz")),
}


def from_fit_result(result: Dict[str, object]):
    """Build sketcher model from Step 2 fit output dict."""
    name = result.get("model")
    if name in _PURE_FIT_KEYS:
        channel, keys = _PURE_FIT_KEYS[name]
        b, c, d, e, sh, sv = (float(result[key]) for key in keys)
        return SimpleMagicFormula(channel, B=b, C=c, D=d, E=e, Sh=sh, Sv=sv)
    if name == "Fy_load":
        values = {key: float(result[key]) for key in LOAD_PARAMS}
        return LoadSensitiveFy(**values, fz0=float(result["fz0"]))
    if name == "mu_vs_fz":
        return FrictionVsFz(
            pDy1=float(result.get("pDy1", 1.5)),
            pDy2=float(result.get("pDy2", -0.15)),
            pDx1=float(result.get("pDx1", 1.65)),
            pDx2=float(result.get("pDx2", -0.18)),
            fz0=float(result.get("fz0", 1000.0)),
        )
    raise ValueError(f"Unsupported model type for import: {name!r}")


def project_dict(models: Dict[str, object]) -> Dict[str, object]:
    """Export sketcher models in Schema v2 JSON format."""
    return {
        "schema_version": 2,
        "conventions": {
            "FZ": "positive compression [N]",
            "SA_model": "radians",
            "SA_UI": "degrees",
            "SL": "dimensionless ratio",
        },
        "runs": [],
        "fits": {name: model.as_dict() for name, model in models.items()},
        "note": "Parameters defined manually in 2D Curve Sketcher (not fitted from TTC data).",
    }
