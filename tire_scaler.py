"""tire_scaler.py — Dimensional and compound tire scaling (e.g. 18x6-10 -> 16x7.5-10 R20).
=========================================================================================
PUT MOTORSPORT — Formula Student Tire Dynamics Suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np

from mf51_model import MF51Model


@dataclass
class TireScaleReport:
    success: bool
    source_tire: str
    target_tire: str
    compound: str
    k_radius: float
    k_width: float
    pdx1_before: float
    pdx1_after: float
    pdx2_before: float
    pdx2_after: float
    pkx1_before: float
    pkx1_after: float
    pkx2_before: float
    pkx2_after: float
    log: List[str]


def scale_18x6_to_16x75_r20(model: MF51Model) -> TireScaleReport:
    """Scale longitudinal Fx parameters from Hoosier 18.0x6.0-10 R20 to 16.0x7.5-10 R20."""
    log: List[str] = []

    r_source = 0.2286
    w_source = 0.1524
    r_target = 0.2032
    w_target = 0.1905

    k_radius = r_target / r_source
    k_width = w_target / w_source

    log.append(f"Geometry: 18x6.0-10 -> 16x7.5-10 (Radius: {k_radius:.3f}x, Width: +{(k_width - 1.0)*100:.1f}%)")

    pdx1_old = model.get("PDX1", 2.22)
    pdx2_old = model.get("PDX2", -0.08)
    pkx1_old = model.get("PKX1", 48.0)
    pkx2_old = model.get("PKX2", -11.7)

    pdy1 = model.get("PDY1", 1.65)
    pdy2 = model.get("PDY2", -0.15)

    if pdy1 > 1.0:
        pdx1_new = float(np.clip(pdy1 * 1.06, 1.45, 2.35))
    else:
        pdx1_new = float(np.clip(pdx1_old * 0.98, 1.45, 2.35))
    log.append(f"Peak friction: PDX1 {pdx1_old:.3f} -> {pdx1_new:.3f} (ratio mu_x/mu_y = {pdx1_new/max(pdy1, 0.1):.2f})")

    if pdy2 < 0.0:
        pdx2_new = float(np.clip(pdy2 * 0.80, -0.45, -0.10))
    else:
        pdx2_new = float(np.clip(pdx2_old * (w_source / w_target), -0.35, -0.10))
    log.append(f"Load sensitivity: PDX2 {pdx2_old:.4f} -> {pdx2_new:.4f} (matched to PDY2={pdy2:.4f})")

    pkx1_new = 28.0
    log.append(f"Longitudinal stiffness: PKX1 {pkx1_old:.1f} -> {pkx1_new:.1f} (places Fx peak at kappa ~ 0.11)")

    pkx2_new = -2.5
    model.set("PKX3", 0.0)
    log.append(f"Extrapolation stabilization: PKX2 {pkx2_old:.2f} -> {pkx2_new:.2f} (keeps Kx > 0 up to 3500 N)")

    model.set("PDX1", pdx1_new)
    model.set("PDX2", pdx2_new)
    model.set("PKX1", pkx1_new)
    model.set("PKX2", pkx2_new)

    model.set("UNLOADED_RADIUS", 0.205)
    model.set("WIDTH", 0.180)
    model.set("RIM_RADIUS", 0.127)
    model.set("RIM_WIDTH", 0.178)

    return TireScaleReport(
        success=True,
        source_tire="Hoosier 18.0x6.0-10 (R20)",
        target_tire="Hoosier 16.0x7.5-10 (R20)",
        compound="R20",
        k_radius=k_radius,
        k_width=k_width,
        pdx1_before=pdx1_old,
        pdx1_after=pdx1_new,
        pdx2_before=pdx2_old,
        pdx2_after=pdx2_new,
        pkx1_before=pkx1_old,
        pkx1_after=pkx1_new,
        pkx2_before=pkx2_old,
        pkx2_after=pkx2_new,
        log=log,
    )
