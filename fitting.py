"""Numerically robust fitting helpers for the PUTM tire models.

The public functions in this module do not depend on Tkinter.  This keeps the
fit reproducible from tests, scripts, and future user interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from tire_model import pacejka_comb_Gyk, pacejka_fy, pacejka_fy_load


@dataclass(frozen=True)
class FitOutput:
    parameters: Dict[str, float]
    diagnostics: Dict[str, object]
    x: np.ndarray
    y: np.ndarray
    prediction: np.ndarray

    def as_dict(self) -> Dict[str, object]:
        return {**self.parameters, **self.diagnostics}


def _finite_xy(x, y) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError("Osie danych mają różną liczbę próbek.")
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 12:
        raise ValueError("Do dopasowania potrzeba co najmniej 12 poprawnych punktów.")
    if np.ptp(x) <= np.finfo(float).eps:
        raise ValueError("Zmienna poślizgu nie ma wystarczającego zakresu.")
    return x, y


def _balanced_weights(x: np.ndarray, n_bins: int = 60) -> np.ndarray:
    """Give each slip interval similar influence despite unequal dwell time."""
    edges = np.linspace(float(x.min()), float(x.max()), min(n_bins, x.size) + 1)
    bins = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    counts = np.bincount(bins, minlength=len(edges) - 1)
    weights = 1.0 / np.sqrt(np.maximum(counts[bins], 1))
    return weights / np.mean(weights)


def _metrics(y: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    residual = prediction - y
    scale = max(float(np.ptp(y)), float(np.max(np.abs(y))), 1.0)
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "nrmse": float(np.sqrt(np.mean(residual**2)) / scale),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "max_abs_error": float(np.max(np.abs(residual))),
        "r2": float(1.0 - np.sum(residual**2) / denominator) if denominator > 0 else float("nan"),
    }


def _covariance_diagnostics(result, residual_scale: float, parameter_scales=None) -> Tuple[list, float]:
    jacobian = np.asarray(result.jac, dtype=float) / residual_scale
    try:
        information = jacobian.T @ jacobian
        scales = np.ones(jacobian.shape[1]) if parameter_scales is None else np.asarray(parameter_scales, dtype=float)
        scaled_information = (jacobian * scales).T @ (jacobian * scales)
        condition = float(np.linalg.cond(scaled_information))
        dof = max(result.fun.size - result.x.size, 1)
        variance = float(np.sum((result.fun * residual_scale) ** 2) / dof)
        covariance = np.linalg.pinv(information) * variance
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0)).tolist()
    except np.linalg.LinAlgError:
        condition = float("inf")
        standard_errors = [float("nan")] * result.x.size
    return [float(value) for value in standard_errors], condition


def fit_pure_magic(
    slip,
    force,
    *,
    model_name: str,
    slip_unit: str,
    shape_factor: float = 1.3,
) -> FitOutput:
    """Fit B, D, E, Sh and Sv with C explicitly fixed.

    Slip angles must be supplied in radians; longitudinal slip is a ratio.
    A robust loss, balanced slip-bin weights and several starts make this much
    less sensitive to dwell points and initial guesses than ``curve_fit``.
    """
    x, y = _finite_xy(slip, force)
    if slip_unit not in {"rad", "ratio"}:
        raise ValueError("slip_unit musi mieć wartość 'rad' albo 'ratio'.")

    near_zero = np.abs(x) <= max(0.15 * np.max(np.abs(x)), np.deg2rad(1.5) if slip_unit == "rad" else 0.02)
    if np.count_nonzero(near_zero) >= 4:
        slope = float(np.polyfit(x[near_zero], y[near_zero], 1)[0])
    else:
        slope = float((y[np.argmax(x)] - y[np.argmin(x)]) / np.ptp(x))
    force_sign = 1.0 if slope >= 0 else -1.0
    y_fit = force_sign * y

    peak = max(float(np.percentile(np.abs(y_fit), 98)), 1.0)
    vertical_limit = max(0.75 * peak, 50.0)
    shift_limit = np.deg2rad(5.0) if slip_unit == "rad" else 0.12
    lower = np.array([0.05, 0.20 * peak, -5.0, -shift_limit, -vertical_limit])
    upper = np.array([150.0, 2.50 * peak, 1.0, shift_limit, vertical_limit])

    stiffness_guess = max(abs(slope), peak / max(np.max(np.abs(x)), 1e-3))
    b_guess = float(np.clip(stiffness_guess / (shape_factor * peak), 0.1, 100.0))
    weights = _balanced_weights(x)
    residual_scale = max(0.05 * peak, 10.0)

    def predict(parameters):
        b, d, e, sh, sv = parameters
        return pacejka_fy(x, b, shape_factor, d, e, sh, sv)

    def residual(parameters):
        return weights * (predict(parameters) - y_fit) / residual_scale

    starts = []
    for b_multiplier in (0.45, 1.0, 2.0):
        for curvature in (-1.0, 0.0, 0.7):
            starts.append(
                np.clip(
                    [b_guess * b_multiplier, peak, curvature, 0.0, float(np.median(y_fit[near_zero])) if np.any(near_zero) else 0.0],
                    lower + 1e-9,
                    upper - 1e-9,
                )
            )

    candidates = [
        least_squares(
            residual,
            start,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=1.0,
            x_scale="jac",
            max_nfev=12000,
        )
        for start in starts
    ]
    result = min(candidates, key=lambda item: float(np.sum(item.fun**2)))
    if not result.success:
        raise RuntimeError(f"Optymalizacja nie osiągnęła zbieżności: {result.message}")

    b, d, e, sh, sv = [float(value) for value in result.x]
    prediction_fit_sign = predict(result.x)
    prediction = force_sign * prediction_fit_sign
    d *= force_sign
    sv *= force_sign

    errors, condition = _covariance_diagnostics(result, residual_scale, upper - lower)
    bound_distance = np.minimum((result.x - lower) / (upper - lower), (upper - result.x) / (upper - lower))
    bound_hits = [name for name, distance in zip(("B", "D", "E", "Sh", "Sv"), bound_distance) if distance < 1e-4]
    warnings = []
    if bound_hits:
        warnings.append("Parametry na granicy: " + ", ".join(bound_hits))
    if not np.isfinite(condition) or condition > 1e10:
        warnings.append("Słaba identyfikowalność parametrów")

    parameters = {"model": model_name, "B": b, "C": float(shape_factor), "D": d, "E": e, "Sh": sh, "Sv": sv}
    diagnostics = {
        **_metrics(y, prediction),
        "n_points": int(x.size),
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "condition_number": condition,
        "standard_errors_free_parameters": errors,
        "free_parameter_order": ["B", "D", "E", "Sh", "Sv"],
        "bound_hits": bound_hits,
        "warnings": warnings,
        "slip_unit": slip_unit,
        "force_unit": "N" if model_name != "Mz0" else "Nm",
        "shape_factor_fixed": True,
    }
    return FitOutput(parameters, diagnostics, x, y, prediction)


def fit_load_sensitive(alpha, fz, force) -> FitOutput:
    """Robust joint Fy(alpha, Fz) fit using radians and positive Fz."""
    alpha = np.asarray(alpha, dtype=float).reshape(-1)
    fz = np.abs(np.asarray(fz, dtype=float).reshape(-1))
    force = np.asarray(force, dtype=float).reshape(-1)
    if not (alpha.size == fz.size == force.size):
        raise ValueError("Kanały alpha, Fz i Fy mają różne długości.")
    valid = np.isfinite(alpha) & np.isfinite(fz) & np.isfinite(force) & (fz > 0)
    alpha, fz, force = alpha[valid], fz[valid], force[valid]
    if alpha.size < 30:
        raise ValueError("Fit load-sensitive wymaga co najmniej 30 poprawnych punktów.")

    fz0 = float(np.median(fz))
    load_levels = np.unique(np.round(fz / 100.0) * 100.0)
    if load_levels.size < 3 or np.ptp(fz) < 0.20 * fz0:
        raise ValueError("Fit load-sensitive wymaga co najmniej trzech wyraźnych poziomów Fz.")

    alpha_edges = np.linspace(float(alpha.min()), float(alpha.max()), 51)
    alpha_bins = np.clip(np.digitize(alpha, alpha_edges[1:-1]), 0, len(alpha_edges) - 2)
    load_bins = np.digitize(fz, np.quantile(fz, [0.25, 0.5, 0.75]))
    joint_bins = alpha_bins * 4 + load_bins
    counts = np.bincount(joint_bins, minlength=200)
    weights = 1.0 / np.sqrt(np.maximum(counts[joint_bins], 1))
    weights /= np.mean(weights)

    mu_guess = float(np.percentile(np.abs(force) / fz, 90))
    initial = np.array([1.3, np.clip(mu_guess, 0.5, 3.0), -0.1, -0.5, 15.0, 2.0, 0.0, 0.0])
    lower = np.array([1.0, 0.3, -2.0, -5.0, 2.0, 0.25, -0.10, -0.5])
    upper = np.array([2.0, 4.0, 1.0, 1.0, 60.0, 10.0, 0.10, 0.5])
    residual_scale = max(0.05 * float(np.percentile(np.abs(force), 98)), 10.0)

    def predict(parameters):
        return pacejka_fy_load((alpha, fz), *parameters, fz0=fz0)

    def residual(parameters):
        return weights * (predict(parameters) - force) / residual_scale

    starts = []
    for pdy2 in (-0.5, -0.1, 0.2):
        for pey1 in (-1.5, -0.5, 0.5):
            start = initial.copy()
            start[2], start[3] = pdy2, pey1
            starts.append(np.clip(start, lower + 1e-9, upper - 1e-9))
    candidates = [
        least_squares(
            residual, start, bounds=(lower, upper), loss="soft_l1", f_scale=1.0,
            x_scale="jac", max_nfev=20000,
        )
        for start in starts
    ]
    result = min(candidates, key=lambda item: float(np.sum(item.fun**2)))
    if not result.success:
        raise RuntimeError(f"Optymalizacja nie osiągnęła zbieżności: {result.message}")

    prediction = predict(result.x)
    names = ("pCy1", "pDy1", "pDy2", "pEy1", "pKy1", "pKy2", "pUy1", "pVy1")
    parameters = {"model": "Fy_load", **{name: float(value) for name, value in zip(names, result.x)}, "fz0": fz0}
    errors, condition = _covariance_diagnostics(result, residual_scale, upper - lower)
    distances = np.minimum((result.x - lower) / (upper - lower), (upper - result.x) / (upper - lower))
    bound_hits = [name for name, distance in zip(names, distances) if distance < 1e-4]
    warnings = ["Parametry na granicy: " + ", ".join(bound_hits)] if bound_hits else []
    if not np.isfinite(condition) or condition > 1e10:
        warnings.append("Słaba identyfikowalność parametrów")
    diagnostics = {
        **_metrics(force, prediction),
        "n_points": int(alpha.size), "n_load_levels_100N": int(load_levels.size),
        "success": bool(result.success), "message": str(result.message), "nfev": int(result.nfev),
        "condition_number": condition, "standard_errors_free_parameters": errors,
        "free_parameter_order": list(names), "bound_hits": bound_hits, "warnings": warnings,
        "slip_unit": "rad", "force_unit": "N",
    }
    return FitOutput(parameters, diagnostics, alpha, force, prediction)


def build_combined_slip_targets(
    data: pd.DataFrame,
    *,
    zero_slip_tolerance: float = 0.012,
    min_abs_alpha_deg: float = 0.5,
) -> pd.DataFrame:
    """Build measured Gyk targets from matched near-zero-slip baselines."""
    required = {"SA", "IA", "FZ", "SL", "FY"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Brak kanałów do Combined Slip: {missing}")

    source_columns = [name for name in ("__source_path__", "__source__") if name in data.columns]
    # Prefer the absolute path when available; basename is only a fallback.
    source_columns = source_columns[:1]
    frame = data[list(required) + source_columns].replace([np.inf, -np.inf], np.nan).dropna(subset=list(required)).copy()
    frame = frame[np.abs(frame["SA"]) >= min_abs_alpha_deg]
    if frame.empty:
        raise ValueError("Brak punktów Combined Slip przy niezerowym kącie SA.")

    frame["FY_CANON"] = -frame["FY"]
    frame["SA_GROUP"] = (frame["SA"] / 0.25).round() * 0.25
    frame["IA_GROUP"] = (frame["IA"] / 0.25).round() * 0.25
    frame["FZ_GROUP"] = (np.abs(frame["FZ"]) / 100.0).round() * 100.0
    group_columns = source_columns + ["SA_GROUP", "IA_GROUP", "FZ_GROUP"]

    baseline = (
        frame[np.abs(frame["SL"]) <= zero_slip_tolerance]
        .groupby(group_columns)["FY_CANON"]
        .agg([("FY0", "median"), ("BASELINE_POINTS", "size")])
        .reset_index()
    )
    baseline = baseline[(baseline["BASELINE_POINTS"] >= 3) & (np.abs(baseline["FY0"]) >= 100.0)]
    targets = frame.merge(baseline, on=group_columns, how="inner")
    targets["CONDITION_ID"] = targets[group_columns].astype(str).agg("|".join, axis=1)
    targets["GYK"] = np.abs(targets["FY_CANON"] / targets["FY0"])
    # Keep moderate values above one visible; the robust loss handles them.
    # Only ratios indicating a near-zero/invalid baseline are rejected here.
    targets = targets[(targets["GYK"] >= 0.0) & (targets["GYK"] <= 3.0)]
    if len(targets) < 30:
        raise ValueError("Za mało dopasowanych punktów bazowych do Combined Slip.")
    return targets


def fit_combined_slip(data: pd.DataFrame) -> FitOutput:
    targets = build_combined_slip_targets(data)
    kappa, gyk = _finite_xy(targets["SL"], targets["GYK"])
    weights = _balanced_weights(kappa)
    residual_scale = 0.05
    lower = np.array([0.1, 0.25])
    upper = np.array([100.0, 3.0])

    def residual(parameters):
        return weights * (pacejka_comb_Gyk(kappa, *parameters) - gyk) / residual_scale

    candidates = [
        least_squares(
            residual,
            [b, c],
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=1.0,
            x_scale="jac",
            max_nfev=10000,
        )
        for b in (2.0, 7.0, 20.0)
        for c in (0.7, 1.2, 1.8)
    ]
    result = min(candidates, key=lambda item: float(np.sum(item.fun**2)))
    if not result.success:
        raise RuntimeError(f"Optymalizacja nie osiągnęła zbieżności: {result.message}")
    prediction = pacejka_comb_Gyk(kappa, *result.x)
    errors, condition = _covariance_diagnostics(result, residual_scale, upper - lower)
    distances = np.minimum((result.x - lower) / (upper - lower), (upper - result.x) / (upper - lower))
    bound_hits = [name for name, distance in zip(("rBy1", "rCy1"), distances) if distance < 1e-4]
    warnings = ["Parametry na granicy: " + ", ".join(bound_hits)] if bound_hits else []
    parameters = {"model": "CombinedSlip", "rBy1": float(result.x[0]), "rCy1": float(result.x[1])}
    diagnostics = {
        **_metrics(gyk, prediction),
        "n_points": int(kappa.size),
        "n_condition_groups": int(targets["CONDITION_ID"].nunique()),
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "condition_number": condition,
        "standard_errors_free_parameters": errors,
        "free_parameter_order": ["rBy1", "rCy1"],
        "bound_hits": bound_hits,
        "warnings": warnings,
        "slip_unit": "ratio",
        "force_unit": "dimensionless",
        "baseline": "median measured Fy at matched SA/IA/FZ and |SL| <= 0.012",
    }
    return FitOutput(parameters, diagnostics, kappa, gyk, prediction)
