"""auto_mf51_extractor.py — Intelligent multi-channel extraction & MF 5.1 fitting
from loaded measurement files and operating conditions (Cornering, Drive/Brake, Fz, IA, P).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from tire_model import RunConfig, TireDataManager
from fitting import fit_load_sensitive, fit_pure_magic, fit_combined_slip
from mf51_model import MF51Model


@dataclass
class ExtractionResult:
    success: bool
    message: str
    files_analyzed: List[str] = field(default_factory=list)
    pressure_kpa: Optional[float] = None
    all_pressures_kpa: List[float] = field(default_factory=list)
    fz_levels_corn: List[float] = field(default_factory=list)
    fz_levels_drive: List[float] = field(default_factory=list)
    camber_levels: List[float] = field(default_factory=list)
    parameters: Dict[str, float] = field(default_factory=dict)
    diagnostics_log: List[str] = field(default_factory=list)


def _normalize_pressure_series(p_series: pd.Series) -> pd.Series:
    """Convert pressure channel to kPa (handles bar / psi / kPa auto-detection)."""
    valid = p_series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=p_series.index)
    med = float(np.median(np.abs(valid)))
    if med < 2.0:
        return p_series * 100.0
    if med < 30.0:
        return p_series * 6.894757
    return p_series


def _find_clusters(values: np.ndarray, threshold: float = 80.0) -> List[float]:
    """Find distinct numeric clusters (e.g. Fz loads or IA angles) in O(N log N)."""
    val = np.sort(values[np.isfinite(values)])
    if len(val) == 0:
        return []
    clusters: List[float] = []
    run_sum = float(val[0])
    run_cnt = 1
    for x in val[1:]:
        mean = run_sum / run_cnt
        if abs(float(x) - mean) <= threshold:
            run_sum += float(x)
            run_cnt += 1
        else:
            clusters.append(float(np.round(mean)))
            run_sum = float(x)
            run_cnt = 1
    if run_cnt > 0:
        clusters.append(float(np.round(run_sum / run_cnt)))
    return sorted(list(dict.fromkeys(clusters)))


class AutoMF51Extractor:
    """Automated multi-condition Pacejka 5.1 fitting engine for TTC data."""

    @classmethod
    def extract_from_app_data(
        cls,
        model: MF51Model,
        runs: List[RunConfig],
        df_cornering: Optional[pd.DataFrame] = None,
        df_drive: Optional[pd.DataFrame] = None,
        target_pressure: Optional[float] = None,
    ) -> ExtractionResult:
        """Extract and fit a full MF 5.1 parameter set across all loaded runs/data."""
        log: List[str] = []
        files_used: List[str] = []

        corn_frames: List[pd.DataFrame] = []
        drive_frames: List[pd.DataFrame] = []

        if df_cornering is not None and not df_cornering.empty:
            corn_frames.append(df_cornering.copy())
        if df_drive is not None and not df_drive.empty:
            drive_frames.append(df_drive.copy())

        for r in runs:
            if not os.path.exists(r.filepath):
                continue
            fname = os.path.basename(r.filepath)
            if fname not in files_used:
                files_used.append(fname)
            try:
                raw = TireDataManager.load_unfiltered(r)
                raw["__source__"] = fname
                raw["__test_type__"] = r.test_type
                if r.test_type == "Cornering":
                    corn_frames.append(raw)
                elif r.test_type == "Drive/Brake":
                    drive_frames.append(raw)
            except Exception as exc:
                log.append(f"Warning: Failed to read raw channels from {fname}: {exc}")

        if not corn_frames and not drive_frames:
            return ExtractionResult(
                success=False,
                message="No measurement data found. Import .mat/.csv files or load the FSAE Demo first.",
                diagnostics_log=log,
            )

        all_corn = pd.concat(corn_frames, ignore_index=True) if corn_frames else pd.DataFrame()
        all_drive = pd.concat(drive_frames, ignore_index=True) if drive_frames else pd.DataFrame()

        if files_used:
            names_lower = [f.lower() for f in files_used]
            has_16 = any("16x" in f or "16." in f or "160" in f for f in names_lower)
            has_18 = any("18x" in f or "18." in f or "180" in f for f in names_lower)
            has_r20 = any("r20" in f for f in names_lower)
            has_r25 = any("r25" in f for f in names_lower)
            if (has_16 and has_18) or (has_r20 and has_r25):
                log.append("WARNING — MIXED TIRE SPECS: Loaded files appear to mix different sizes or compounds (e.g., 16x7.5 vs 18x6 or R20 vs R25B).")
                log.append("   Consider running dimensional tire scaling (Step 3) after fitting.")

        if not all_corn.empty and "FZ" in all_corn:
            all_corn["FZ"] = np.abs(all_corn["FZ"])
        if not all_drive.empty and "FZ" in all_drive:
            all_drive["FZ"] = np.abs(all_drive["FZ"])

        detected_pressures: List[float] = []
        for df_tmp in [all_corn, all_drive]:
            if not df_tmp.empty and "P" in df_tmp and np.isfinite(df_tmp["P"]).any():
                df_tmp["P_kPa"] = _normalize_pressure_series(df_tmp["P"])
                valid_p = df_tmp["P_kPa"].dropna()
                if not valid_p.empty:
                    detected_pressures.extend(_find_clusters(valid_p.to_numpy(), threshold=5.0))

        detected_pressures = sorted(list(dict.fromkeys(detected_pressures)))

        selected_p = target_pressure
        if selected_p is None and detected_pressures:
            selected_p = detected_pressures[0]
            log.append(f"Detected pressures P: {', '.join(f'{p:.1f} kPa' for p in detected_pressures)}")
            log.append(f"Selected primary calibration pressure: {selected_p:.1f} kPa")

        if selected_p is not None:
            if not all_corn.empty and "P_kPa" in all_corn and np.isfinite(all_corn["P_kPa"]).any():
                mask_p = np.abs(all_corn["P_kPa"] - selected_p) <= 7.0
                if mask_p.sum() >= 30:
                    all_corn = all_corn[mask_p].copy()
            if not all_drive.empty and "P_kPa" in all_drive and np.isfinite(all_drive["P_kPa"]).any():
                mask_p = np.abs(all_drive["P_kPa"] - selected_p) <= 7.0
                if mask_p.sum() >= 30:
                    all_drive = all_drive[mask_p].copy()

        fitted_params: Dict[str, float] = {}

        # 3. CORNERING (Fy)
        fz_corn_levels: List[float] = []
        camber_levels: List[float] = []

        if not all_corn.empty and all(c in all_corn for c in ["SA", "FY", "FZ"]):
            valid_c = all_corn.dropna(subset=["SA", "FY", "FZ"])
            valid_c = valid_c[(valid_c["FZ"] > 80.0) & (np.abs(valid_c["SA"]) <= 20.0)].copy()

            if len(valid_c) >= 30:
                fz_corn_levels = _find_clusters(valid_c["FZ"].to_numpy(), threshold=100.0)
                log.append(f"Cornering: {len(valid_c)} valid samples across Fz levels: {fz_corn_levels} N")

                fz0_calc = float(np.median(valid_c["FZ"]))
                fz0 = float(np.round(fz0_calc / 50.0) * 50.0)
                model.set("FNOMIN", fz0)
                fitted_params["FNOMIN"] = fz0

                ia_series = valid_c["IA"] if "IA" in valid_c else pd.Series(0.0, index=valid_c.index)
                pure_camber_mask = np.abs(ia_series) <= 0.75
                sub_c = valid_c[pure_camber_mask] if pure_camber_mask.sum() >= 30 else valid_c

                alpha_rad = np.deg2rad(sub_c["SA"].to_numpy(dtype=float))
                fz_arr = sub_c["FZ"].to_numpy(dtype=float)
                fy_arr = -sub_c["FY"].to_numpy(dtype=float)

                if len(fz_corn_levels) >= 3 and (np.ptp(fz_arr) >= 0.20 * fz0):
                    try:
                        fit_out = fit_load_sensitive(alpha_rad, fz_arr, fy_arr)
                        p = fit_out.parameters
                        model.set("PCY1", float(p["pCy1"]))
                        model.set("PDY1", float(p["pDy1"]))
                        model.set("PDY2", float(p["pDy2"]))
                        model.set("PEY1", float(p["pEy1"]))
                        model.set("PKY1", float(p["pKy1"]))
                        model.set("PKY2", float(p["pKy2"]))
                        model.set("PHY1", float(p["pUy1"]))
                        model.set("PVY1", float(p["pVy1"]))

                        fitted_params.update({
                            "PCY1": p["pCy1"], "PDY1": p["pDy1"], "PDY2": p["pDy2"],
                            "PEY1": p["pEy1"], "PKY1": p["pKy1"], "PKY2": p["pKy2"],
                            "PHY1": p["pUy1"], "PVY1": p["pVy1"],
                        })
                        log.append(f"OK Fitted 3D Fy(SA, Fz): mu_y0={p['pDy1']:.3f}, PDY2={p['pDy2']:.4f}, PKY1={p['pKy1']:.1f}")
                    except Exception as e:
                        log.append(f"Joint 3D Fy_load fit skipped ({e}); fitting individual Fz levels...")
                        cls._fit_cornering_by_levels(model, sub_c, fz_corn_levels, fz0, fitted_params, log)
                else:
                    cls._fit_cornering_by_levels(model, sub_c, fz_corn_levels, fz0, fitted_params, log)

                if "IA" in valid_c and np.isfinite(valid_c["IA"]).any():
                    camber_levels = _find_clusters(valid_c["IA"].to_numpy(), threshold=0.5)
                    log.append(f"Detected Camber angles (IA): {camber_levels} deg")
                    cls._fit_camber_effects(model, valid_c, fz0, fitted_params, log)

                if "MZ" in valid_c and np.isfinite(valid_c["MZ"]).any() and (np.abs(valid_c["MZ"]) > 0.1).any():
                    cls._fit_aligning_torque(model, valid_c, fz0, fitted_params, log)

        # 4. DRIVE / BRAKE (Fx)
        fz_drive_levels: List[float] = []
        if not all_drive.empty and all(c in all_drive for c in ["SL", "FX", "FZ"]):
            valid_d = all_drive.dropna(subset=["SL", "FX", "FZ"])
            valid_d = valid_d[(valid_d["FZ"] > 80.0) & (np.abs(valid_d["SL"]) <= 0.40)].copy()

            if len(valid_d) >= 25:
                if "SA" in valid_d and np.isfinite(valid_d["SA"]).any():
                    pure_sa = np.abs(valid_d["SA"]) <= 0.6
                    valid_d_pure = valid_d[pure_sa] if pure_sa.sum() >= 25 else valid_d
                else:
                    valid_d_pure = valid_d

                fz_drive_levels = _find_clusters(valid_d_pure["FZ"].to_numpy(), threshold=100.0)
                log.append(f"Drive/Brake: {len(valid_d_pure)} valid samples across Fz levels: {fz_drive_levels} N")

                fz0_d = model.get("FNOMIN", 1000.0)
                cls._fit_drive_multi_fz(model, valid_d_pure, fz_drive_levels, fz0_d, fitted_params, log)

                if "SA" in valid_d and "FY" in valid_d and (np.abs(valid_d["SA"]) > 1.0).any():
                    cls._fit_combined_slip(model, valid_d, fitted_params, log)

        if not fitted_params:
            return ExtractionResult(
                success=False,
                message="Loaded files did not contain enough valid samples to extract MF 5.1 parameters.",
                files_analyzed=files_used,
                diagnostics_log=log,
            )

        summary_msg = cls._build_summary_message(files_used, selected_p, fz_corn_levels, fz_drive_levels, camber_levels, fitted_params)
        return ExtractionResult(
            success=True,
            message=summary_msg,
            files_analyzed=files_used,
            pressure_kpa=selected_p,
            all_pressures_kpa=detected_pressures,
            fz_levels_corn=fz_corn_levels,
            fz_levels_drive=fz_drive_levels,
            camber_levels=camber_levels,
            parameters=fitted_params,
            diagnostics_log=log,
        )

    @classmethod
    def _fit_cornering_by_levels(
        cls, model: MF51Model, df: pd.DataFrame, fz_levels: List[float], fz0: float,
        fitted_params: Dict[str, float], log: List[str]
    ):
        fits_by_fz = []
        for fz_val in fz_levels:
            mask = np.abs(df["FZ"] - fz_val) <= 120.0
            sub = df[mask]
            if len(sub) < 15:
                continue
            x = np.deg2rad(sub["SA"].to_numpy(dtype=float))
            y = -sub["FY"].to_numpy(dtype=float)
            try:
                out = fit_pure_magic(x, y, model_name="Fy0", slip_unit="rad")
                p = out.parameters
                mu_est = p["D"] / fz_val
                stiff_est = (p["B"] * p["C"] * p["D"]) / fz_val
                fits_by_fz.append((fz_val, mu_est, stiff_est, p))
                log.append(f"  > Fz={fz_val:.0f} N: mu_y={mu_est:.3f}, stiffness={stiff_est:.1f} [1/rad], Cy={p['C']:.2f}")
            except Exception as e:
                log.append(f"  > Fit error at Fz={fz_val:.0f} N: {e}")

        if not fits_by_fz:
            return

        avg_C = float(np.mean([item[3]["C"] for item in fits_by_fz]))
        avg_E = float(np.mean([item[3]["E"] for item in fits_by_fz]))
        model.set("PCY1", avg_C)
        model.set("PEY1", avg_E)
        fitted_params["PCY1"] = avg_C
        fitted_params["PEY1"] = avg_E

        if len(fits_by_fz) >= 2:
            fz_arr = np.array([item[0] for item in fits_by_fz])
            mu_arr = np.array([item[1] for item in fits_by_fz])
            stiff_arr = np.array([item[2] for item in fits_by_fz])
            dfz_arr = (fz_arr - fz0) / fz0

            pdy2, pdy1 = np.polyfit(dfz_arr, mu_arr, 1)
            _, pky1 = np.polyfit(dfz_arr, stiff_arr, 1)

            pdy1 = float(np.clip(pdy1, 0.8, 3.0))
            pdy2 = float(np.clip(pdy2, -0.6, 0.1))
            pky1 = float(np.clip(pky1, 15.0, 70.0))

            model.set("PDY1", pdy1)
            model.set("PDY2", pdy2)
            model.set("PKY1", pky1)
            fitted_params.update({"PDY1": pdy1, "PDY2": pdy2, "PKY1": pky1})
            log.append(f"OK Lateral load sensitivity: PDY1={pdy1:.3f}, PDY2={pdy2:.4f}, PKY1={pky1:.1f}")
        else:
            item = fits_by_fz[0]
            pdy1 = float(item[1])
            pky1 = float(item[2])
            model.set("PDY1", pdy1)
            model.set("PKY1", pky1)
            fitted_params.update({"PDY1": pdy1, "PKY1": pky1})
            log.append(f"OK Lateral params mapped from single Fz: PDY1={pdy1:.3f}, PKY1={pky1:.1f}")

    @classmethod
    def _fit_camber_effects(
        cls, model: MF51Model, df: pd.DataFrame, fz0: float,
        fitted_params: Dict[str, float], log: List[str]
    ):
        small_sa = df[np.abs(df["SA"]) <= 0.6]
        if len(small_sa) >= 20 and np.ptp(small_sa["IA"]) >= 1.0:
            ia_rad = np.deg2rad(small_sa["IA"].to_numpy(dtype=float))
            fz = small_sa["FZ"].to_numpy(dtype=float)
            fy = -small_sa["FY"].to_numpy(dtype=float)

            try:
                slope, _ = np.polyfit(ia_rad, fy / fz, 1)
                pvy3 = float(np.clip(slope, 0.01, 0.25))
                model.set("PVY3", pvy3)
                fitted_params["PVY3"] = pvy3
                log.append(f"OK Camber Thrust: PVY3 = {pvy3:.4f} [1/rad]")
            except Exception as e:
                log.append(f"Failed to fit Camber Thrust ({e})")

            try:
                slope_k, _ = np.polyfit(ia_rad, fy, 1)
                pky4_mag = float(np.clip(slope_k / max(fz0, 1.0), 0.8, 6.0))
                model.set("PKY4", pky4_mag)
                fitted_params["PKY4"] = pky4_mag
                log.append(f"OK PAC2002 Camber stiffness: PKY4 = {pky4_mag:.3f} 1/rad (~{slope_k / 57.3:.1f} N/deg @ {fz0:.0f} N)")
            except Exception as e:
                log.append(f"Failed to fit camber stiffness PKY4 ({e})")

        ia_clusters = _find_clusters(df["IA"].to_numpy(), threshold=0.5)
        if len(ia_clusters) >= 2:
            peaks_by_gamma = []
            for ia_val in ia_clusters:
                sub = df[np.abs(df["IA"] - ia_val) <= 0.4]
                if len(sub) >= 20:
                    peak_fy = float(np.percentile(np.abs(sub["FY"]), 95))
                    avg_fz = float(np.median(sub["FZ"]))
                    peaks_by_gamma.append((np.deg2rad(ia_val), peak_fy / avg_fz))

            if len(peaks_by_gamma) >= 2:
                g_arr = np.array([item[0] for item in peaks_by_gamma])
                mu_arr = np.array([item[1] for item in peaks_by_gamma])
                try:
                    p = np.polyfit(g_arr**2, mu_arr, 1)
                    if p[0] < 0 and p[1] > 0:
                        pdy3 = float(np.clip(-p[0] / p[1], 0.05, 3.5))
                        model.set("PDY3", pdy3)
                        fitted_params["PDY3"] = pdy3
                        log.append(f"OK Camber grip drop: PDY3 = {pdy3:.3f}")
                except Exception:
                    pass

    @classmethod
    def _fit_drive_multi_fz(
        cls, model: MF51Model, df: pd.DataFrame, fz_levels: List[float], fz0: float,
        fitted_params: Dict[str, float], log: List[str]
    ):
        fits_by_fz = []
        for fz_val in fz_levels:
            mask = np.abs(df["FZ"] - fz_val) <= 120.0
            sub = df[mask]
            if len(sub) < 15:
                continue
            x = sub["SL"].to_numpy(dtype=float)
            y = sub["FX"].to_numpy(dtype=float)
            try:
                out = fit_pure_magic(x, y, model_name="Fx0", slip_unit="ratio")
                p = out.parameters
                mu_x_est = p["D"] / fz_val
                stiff_x_est = (p["B"] * p["C"] * p["D"]) / fz_val
                fits_by_fz.append((fz_val, mu_x_est, stiff_x_est, p))
                log.append(f"  > Drive Fz={fz_val:.0f} N: mu_x={mu_x_est:.3f}, Kx={stiff_x_est:.1f} [/ratio], Cx={p['C']:.2f}")
            except Exception as e:
                log.append(f"  > Fx fit error at Fz={fz_val:.0f} N: {e}")

        if not fits_by_fz:
            return

        avg_Cx = float(np.mean([item[3]["C"] for item in fits_by_fz]))
        avg_Ex = float(np.mean([item[3]["E"] for item in fits_by_fz]))
        model.set("PCX1", avg_Cx)
        model.set("PEX1", avg_Ex)
        fitted_params["PCX1"] = avg_Cx
        fitted_params["PEX1"] = avg_Ex

        if len(fits_by_fz) >= 2:
            fz_arr = np.array([item[0] for item in fits_by_fz])
            mu_arr = np.array([item[1] for item in fits_by_fz])
            stiff_arr = np.array([item[2] for item in fits_by_fz])
            dfz_arr = (fz_arr - fz0) / fz0

            pdx2, pdx1 = np.polyfit(dfz_arr, mu_arr, 1)
            pkx2, pkx1 = np.polyfit(dfz_arr, stiff_arr, 1)

            pdx1 = float(np.clip(pdx1, 0.9, 3.0))
            pdx2 = float(np.clip(pdx2, -0.6, 0.1))
            pkx1 = float(np.clip(pkx1, 10.0, 60.0))

            min_pkx2 = -0.18 * pkx1
            pkx2_clamped = float(np.clip(pkx2, min_pkx2, 10.0))

            model.set("PDX1", pdx1)
            model.set("PDX2", pdx2)
            model.set("PKX1", pkx1)
            model.set("PKX2", pkx2_clamped)
            model.set("PKX3", 0.0)

            fitted_params.update({"PDX1": pdx1, "PDX2": pdx2, "PKX1": pkx1, "PKX2": pkx2_clamped, "PKX3": 0.0})
            log.append(f"OK Longitudinal load sensitivity: PDX1={pdx1:.3f}, PDX2={pdx2:.4f}, PKX1={pkx1:.1f}, PKX2={pkx2_clamped:.2f}")
        else:
            item = fits_by_fz[0]
            pdx1 = float(item[1])
            pkx1 = float(item[2])
            model.set("PDX1", pdx1)
            model.set("PKX1", pkx1)
            fitted_params.update({"PDX1": pdx1, "PKX1": pkx1})
            log.append(f"OK Fx params mapped from single Fz: PDX1={pdx1:.3f}, PKX1={pkx1:.1f}")

    @classmethod
    def _fit_aligning_torque(
        cls, model: MF51Model, df: pd.DataFrame, fz0: float,
        fitted_params: Dict[str, float], log: List[str]
    ):
        valid_mz = df.dropna(subset=["SA", "MZ", "FZ"])
        if len(valid_mz) < 30:
            return
        x = np.deg2rad(valid_mz["SA"].to_numpy(dtype=float))
        y = valid_mz["MZ"].to_numpy(dtype=float)
        try:
            out = fit_pure_magic(x, y, model_name="Mz0", slip_unit="rad")
            p = out.parameters
            r0 = model.get("UNLOADED_RADIUS", 0.203)
            dz = p["D"]
            bz = p["B"]
            cz = p["C"]
            ez = p["E"]

            ky = model.get("PKY1", 38.0) * fz0
            trail_0 = (bz * cz * dz) / max(ky, 1e-3)
            qdz1 = float(np.clip(trail_0 / r0, 0.05, 0.40))

            model.set("QCZ1", float(np.clip(cz, 1.0, 1.4)))
            model.set("QDZ1", qdz1)
            model.set("QBZ1", float(np.clip(bz, 5.0, 25.0)))
            model.set("QEZ1", float(np.clip(ez, -3.0, 1.0)))

            fitted_params.update({"QCZ1": cz, "QDZ1": qdz1, "QBZ1": bz, "QEZ1": ez})
            log.append(f"OK Aligning torque Mz: trail t0 = {trail_0*1000.0:.1f} mm, QDZ1 = {qdz1:.3f}")
        except Exception as e:
            log.append(f"Failed to fit aligning torque Mz: {e}")

    @classmethod
    def _fit_combined_slip(
        cls, model: MF51Model, df: pd.DataFrame,
        fitted_params: Dict[str, float], log: List[str]
    ):
        try:
            out = fit_combined_slip(df)
            p = out.parameters
            rby1 = float(p.get("rBy1", 10.0))
            rcy1 = float(p.get("rCy1", 1.05))
            model.set("RBY1", rby1)
            model.set("RCY1", rcy1)
            fitted_params.update({"RBY1": rby1, "RCY1": rcy1})
            log.append(f"OK Combined Slip reduction: RBY1 = {rby1:.2f}, RCY1 = {rcy1:.2f}")
        except Exception as e:
            log.append(f"Combined Slip fit skipped ({e})")

    @classmethod
    def _build_summary_message(
        cls, files: List[str], pressure: Optional[float],
        fz_corn: List[float], fz_drive: List[float], camber: List[float],
        params: Dict[str, float]
    ) -> str:
        lines = [
            "MF 5.1 MULTI-CHANNEL EXTRACTION COMPLETED:",
            "",
            f"Files analyzed ({len(files)}):",
        ]
        for f in files[:4]:
            lines.append(f"   - {f}")
        if len(files) > 4:
            lines.append(f"   - ... and {len(files) - 4} additional file(s)")

        lines.append("")
        if pressure is not None:
            lines.append(f"Operating Pressure (P): {pressure:.1f} kPa")
        if fz_corn:
            lines.append(f"Cornering Loads (Fz): {', '.join(f'{x:.0f} N' for x in fz_corn)}")
        if fz_drive:
            lines.append(f"Drive/Brake Loads (Fz): {', '.join(f'{x:.0f} N' for x in fz_drive)}")
        if camber:
            lines.append(f"Camber Angles (IA): {', '.join(f'{x:.1f} deg' for x in camber)}")

        lines.append("")
        lines.append("KEY EXTRACTED MF 5.1 PARAMETERS:")

        if "PDY1" in params and "PDY2" in params:
            lines.append(f"   - Lateral Grip mu_y(Fz):   PDY1 = {params['PDY1']:.3f} | PDY2 = {params['PDY2']:.4f}")
        if "PKY1" in params:
            lines.append(f"   - Cornering Stiffness:     PKY1 = {params['PKY1']:.1f} 1/rad" + (f" | PKY2 = {params['PKY2']:.2f}" if "PKY2" in params else ""))
        if "PVY3" in params:
            lines.append(f"   - Camber Thrust:           PVY3 = {params['PVY3']:.4f} 1/rad" + (f" | PDY3 = {params['PDY3']:.3f}" if "PDY3" in params else ""))
        if "PDX1" in params and "PDX2" in params:
            lines.append(f"   - Long. Grip mu_x(Fz):     PDX1 = {params['PDX1']:.3f} | PDX2 = {params['PDX2']:.4f}")
        if "PKX1" in params:
            lines.append(f"   - Long. Stiffness:         PKX1 = {params['PKX1']:.1f}")
        if "QDZ1" in params:
            lines.append(f"   - Pneumatic Trail Mz:      QDZ1 = {params['QDZ1']:.3f}")

        lines.append("")
        lines.append("Active MF 5.1 model and all simulation plots have been updated.")
        return "\n".join(lines)
