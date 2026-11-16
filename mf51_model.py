"""mf51_model.py — Silnik obliczeniowy Magic Formula 5.1 / MF-Tyre 5.2
=====================================================================
Implementuje standardowy model Pacejka MF 5.1 (PAC2002 / MF-Tyre 5.2) dla:
- Fy (siła boczna) z czułością na obciążenie Fz oraz pochylenie koła (camber/IA gamma)
- Fx (siła wzdłużna) z czułością na Fz oraz gamma
- Mz (moment stabilizujący) z podziałem fizycznym na ramię śladu pneumatycznego (trail)
  oraz moment resztkowy (residual torque)
- Combined Slip (poślizg łączony: redukcja Gxa, Gyk)
- Generator standardowego pliku parametrów .tir (kompatybilnego z Adams/Car, IPG CarMaker,
  OptimumLap itp.)
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Any
import numpy as np


# ==============================================================================
# PARAMETRY DOMYŚLNE DLA PROFILI OPON (np. bolid Formula Student - 16x7.5-10)
# ==============================================================================

DEFAULT_MF51_PARAMS: Dict[str, float] = {
    # Bazowe parametry geometryczne i obciążeniowe (FSAE 16x7.5-10)
    "FNOMIN": 1000.0,          # Obciążenie nominalne koła [N] (w bolidach FSAE kalibrowane na 700 N)
    "UNLOADED_RADIUS": 0.205,  # Promień swobodny opony [m] (~16.0 cali średnicy)
    "WIDTH": 0.180,            # Szerokość opony [m] (7.5 cala ~ 0.190 m)
    "RIM_RADIUS": 0.127,       # Promień obręczy [m] (10 cali średnicy obręczy)
    "RIM_WIDTH": 0.178,        # Szerokość obręczy [m] (7 cali szerokości)
    "ASPECT_RATIO": 0.50,      # Profil opony

    # Ciśnienia robocze (PRESSURE)
    "NOMPRES": 82737.0,        # Nominalne ciśnienie [Pa] (~12.0 psi / 0.827 bar)
    "INFLPRES": 82737.0,       # Ciśnienie pomiarowe [Pa]

    # Współczynniki skalujące (SCALING COEFFICIENTS)
    "LFZO": 1.0,
    "LCX": 1.0,
    "LMUX": 1.0,               # Skalowanie przyczepności wzdłużnej (1.0 surowe TTC, 0.65 tor asfaltowy)
    "LEX": 1.0,
    "LKX": 1.0,
    "LHX": 1.0,
    "LVX": 1.0,
    "LGAX": 1.0,
    "LCY": 1.0,
    "LMUY": 1.0,               # Skalowanie przyczepności bocznej (1.0 surowe TTC, 0.65 tor asfaltowy)
    "LEY": 1.0,
    "LKY": 1.0,
    "LHY": 1.0,
    "LVY": 1.0,
    "LGAY": 1.0,
    "LTR": 1.0,
    "LRES": 1.0,
    "LGAZ": 1.0,
    "LXAL": 1.0,
    "LYKA": 1.0,
    "LVYKA": 1.0,
    "LS": 1.0,

    # Sztywności pionowe i ugięcie (VERTICAL)
    "VERTICAL_STIFFNESS": 125000.0, # Sztywność promieniowa opony [N/m] (~125 N/mm dla opon FSAE)
    "VERTICAL_DAMPING": 500.0,      # Tłumienie pionowe [N*s/m]
    "BREFF": 8.0,
    "DREFF": 0.25,
    "FREFF": 0.07,
    "Q_RE0": 1.0,
    "Q_V1": 0.0,
    "Q_V2": 0.0,
    "Q_FZ1": 0.0,
    "Q_FZ2": 0.0,
    "Q_FCX": 0.0,
    "Q_FCY": 0.0,

    # Współczynniki ciśnieniowe (PRESSURE COEFFICIENTS)
    "PPX1": 0.0, "PPX2": 0.0, "PPX3": 0.0, "PPX4": 0.0,
    "PPY1": 0.0, "PPY2": 0.0, "PPY3": 0.0, "PPY4": 0.0, "PPY5": 0.0,
    "PPZ1": 0.0, "PPZ2": 0.0,

    # Siła boczna (LATERAL - Fy)
    "PCY1": 1.30,        # Współczynnik kształtu Cy
    "PDY1": 1.65,        # Współczynnik tarcia mu_y przy Fz0
    "PDY2": -0.15,       # Zależność mu_y od przyrostu obciążenia dfz
    "PDY3": -0.80,       # Spadek mu_y z kwadratem kąta pochylenia gamma^2
    "PEY1": -0.50,       # Współczynnik krzywizny Ey przy Fz0
    "PEY2": -0.05,       # Zależność Ey od dfz
    "PEY3": 0.0,         # Zależność Ey od gamma
    "PEY4": 0.0,         # Asymetria Ey
    "PKY1": 38.0,        # Maksymalna sztywność znoszenia / Fz0 [1/rad]
    "PKY2": 1.80,        # Położenie szczytu sztywności znoszenia (Fz / Fz0)
    "PKY3": 0.15,        # Wpływ gamma na sztywność znoszenia
    "PKY4": 2.20,        # Sztywność pochylenia koła Camber stiffness przy Fz0 [1/rad]
    "PKY5": 0.0,         # Czułość sztywności pochylenia na dfz
    "PKY6": 0.0,         # Nieliniowość sztywności pochylenia
    "PKY7": 0.0,         # Wpływ obciążenia przy dużym pochyleniu
    "PHY1": 0.0,         # Przesunięcie poziome Shy przy Fz0 [rad]
    "PHY2": 0.0,         # Zależność Shy od dfz
    "PHY3": 0.04,        # Wpływ camberu na przesunięcie poziome (camber thrust) [rad/rad]
    "PVY1": 0.0,         # Przesunięcie pionowe Svy / Fz
    "PVY2": 0.0,         # Zależność Svy od dfz
    "PVY3": 0.12,        # Siła pionowa generowana pochyleniem koła (camber thrust force)
    "PVY4": 0.0,

    # Siła wzdłużna (LONGITUDINAL - Fx)
    "PCX1": 1.65,        # Współczynnik kształtu Cx
    "PDX1": 1.75,        # Współczynnik tarcia mu_x przy Fz0
    "PDX2": -0.15,       # Zależność mu_x od dfz
    "PDX3": 0.0,         # Wpływ gamma na mu_x
    "PEX1": -0.40,       # Krzywizna Ex
    "PEX2": 0.0,
    "PEX3": 0.0,
    "PEX4": 0.0,
    "PKX1": 26.0,        # Sztywność poślizgu wzdłużnego Kxk / Fz
    "PKX2": 0.0,         # Zależność Kx od dfz
    "PKX3": 0.0,         # Wykładnik nasycenia Kx
    "PHX1": 0.0,         # Przesunięcie poziome Shx
    "PHX2": 0.0,
    "PVX1": 0.0,         # Przesunięcie pionowe Svx
    "PVX2": 0.0,

    # Moment stabilizujący (ALIGNING - Mz)
    "QBZ1": 9.5,         # Sztywność ramienia śladu pneumatycznego Bt
    "QBZ2": -2.0,        # Wpływ dfz na Bt
    "QBZ3": 0.0,
    "QBZ4": 0.0,
    "QBZ5": 0.4,         # Wpływ gamma na Bt
    "QCZ1": 1.25,        # Kształt ramienia śladu Ct
    "QDZ1": 0.10,        # Szczytowe ramię śladu pneumatycznego Dt0 = Fz*(R0/Fz0)*QDZ1 [m]
    "QDZ2": -0.02,       # Wpływ dfz na Dt
    "QDZ3": 0.15,        # Wpływ gamma na Dt
    "QEZ1": -0.80,       # Krzywizna ramienia śladu Et
    "QEZ2": 0.0,
    "QEZ3": 0.0,
    "QEZ4": 0.0,
    "QEZ5": 0.0,
    "QHZ1": 0.0,         # Przesunięcie poziome śladu Sht
    "QHZ2": 0.0,
    "QHZ3": 0.0,
    "QHZ4": 0.0,
    "QDZ6": 0.002,       # Moment resztkowy Mzr / (Fz * R0)
    "QDZ7": 0.0,
    "QDZ8": 0.04,        # Moment resztkowy od pochylenia koła (camber torque)
    "QDZ9": 0.0,
    "QBZ9": 12.0,        # Sztywność momentu resztkowego Br
    "QBZ10": 0.0,

    # Poślizg łączony (COMBINED SLIP)
    "RBX1": 11.0,        # Redukcja Fx od kąta znoszenia
    "RBX2": 8.0,
    "RCX1": 1.05,
    "REX1": 0.0,
    "RBY1": 9.5,         # Redukcja Fy od poślizgu wzdłużnego
    "RBY2": 7.0,
    "RCY1": 1.05,
    "REY1": 0.0,
}


@dataclass
class MF51Model:
    """Implementacja równań Magic Formula 5.1 / PAC2002."""
    p: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_MF51_PARAMS))

    @property
    def parameters(self) -> Dict[str, float]:
        """Słownik parametrów modelu."""
        return self.p

    def copy(self) -> MF51Model:
        return MF51Model(p=dict(self.p))

    def get(self, key: str, default: float = 0.0) -> float:
        return float(self.p.get(key, default))

    def set(self, key: str, value: float) -> None:
        self.p[key] = float(value)

    # --------------------------------------------------------------------------
    # 1. SIŁA BOCZNA: Fy0(alpha, Fz, gamma)
    # --------------------------------------------------------------------------
    def evaluate_fy(self, alpha_rad: np.ndarray, fz: float, gamma_rad: float = 0.0) -> Tuple[np.ndarray, Dict[str, float]]:
        """Oblicza siłę boczną czystego znoszenia Fy0 w N.
        
        Zwraca (fy, parametry_wewnętrzne_krzywej).
        """
        fz = max(float(fz), 1.0)
        fz0 = max(self.get("FNOMIN", 1000.0), 1.0)
        dfz = (fz - fz0) / fz0
        g = float(gamma_rad)

        # Współczynnik kształtu
        Cy = self.get("PCY1", 1.30) * self.get("LCY", 1.0)

        # Współczynnik tarcia i wartość szczytowa ze skalowaniem nawierzchni LMUY
        mu_y = (self.get("PDY1", 1.65) + self.get("PDY2", -0.15) * dfz) * (1.0 - self.get("PDY3", 0.0) * (g ** 2))
        lmuy = self.get("LMUY", 1.0)
        Dy = mu_y * fz * lmuy

        # Przesunięcia
        Shy = (self.get("PHY1", 0.0) + self.get("PHY2", 0.0) * dfz) + self.get("PHY3", 0.0) * g
        Svy = fz * ((self.get("PVY1", 0.0) + self.get("PVY2", 0.0) * dfz) + 
                    (self.get("PVY3", 0.0) + self.get("PVY4", 0.0) * dfz) * g)

        # Kąt zastępczy
        alpha_y = alpha_rad + Shy

        # Współczynnik krzywizny
        Ey = (self.get("PEY1", -0.5) + self.get("PEY2", 0.0) * dfz) * (
            1.0 - (self.get("PEY3", 0.0) + self.get("PEY4", 0.0) * g) * np.sign(alpha_y)
        )
        Ey = np.clip(Ey, -10.0, 1.0)

        # Sztywność znoszenia (Cornering Stiffness)
        pKy2 = max(self.get("PKY2", 1.8), 0.01)
        Ky_alpha = (self.get("PKY1", 38.0) * fz0 * 
                    np.sin(2.0 * np.arctan(fz / (pKy2 * fz0))) * 
                    (1.0 - self.get("PKY3", 0.0) * abs(g)))
        
        # Współczynnik sztywności By
        denom = Cy * Dy
        By = Ky_alpha / (denom + 1e-6)

        # Magic Formula
        B_ay = By * alpha_y
        atan_B_ay = np.arctan(B_ay)
        arg = B_ay - Ey * (B_ay - atan_B_ay)
        fy = Dy * np.sin(Cy * np.arctan(arg)) + Svy

        meta = {
            "mu_y": float(mu_y), "Dy": float(Dy), "Cy": float(Cy), 
            "By": float(By), "Ey_mean": float(np.mean(Ey)),
            "Ky_alpha": float(Ky_alpha), "Shy": float(Shy), "Svy": float(Svy),
        }
        return fy, meta

    # --------------------------------------------------------------------------
    # 2. SIŁA WZDŁUŻNA: Fx0(kappa, Fz, gamma)
    # --------------------------------------------------------------------------
    def evaluate_fx(self, kappa: np.ndarray, fz: float, gamma_rad: float = 0.0) -> Tuple[np.ndarray, Dict[str, float]]:
        """Oblicza siłę wzdłużną czystego poślizgu Fx0 w N."""
        fz = max(float(fz), 1.0)
        fz0 = max(self.get("FNOMIN", 1000.0), 1.0)
        dfz = (fz - fz0) / fz0
        g = float(gamma_rad)

        Cx = self.get("PCX1", 1.65) * self.get("LCX", 1.0)
        mu_x = (self.get("PDX1", 1.75) + self.get("PDX2", -0.15) * dfz) * (1.0 - self.get("PDX3", 0.0) * (g ** 2))
        lmux = self.get("LMUX", 1.0)
        Dx = mu_x * fz * lmux

        Shx = self.get("PHX1", 0.0) + self.get("PHX2", 0.0) * dfz
        Svx = fz * (self.get("PVX1", 0.0) + self.get("PVX2", 0.0) * dfz)

        kappa_x = kappa + Shx

        Ex = (self.get("PEX1", -0.4) + self.get("PEX2", 0.0) * dfz + self.get("PEX3", 0.0) * (dfz ** 2)) * (
            1.0 - self.get("PEX4", 0.0) * np.sign(kappa_x)
        )
        Ex = np.clip(Ex, -10.0, 1.0)

        # Sztywność wzdłużna Kx_kappa z fizycznym zabezpieczeniem przed spadkiem do zera
        pkx1 = self.get("PKX1", 26.0)
        raw_factor = (pkx1 + self.get("PKX2", 0.0) * dfz) * np.exp(self.get("PKX3", 0.0) * dfz)
        safe_factor = max(raw_factor, 0.20 * pkx1)
        Kx_kappa = fz * safe_factor * self.get("LKX", 1.0)
        Bx = Kx_kappa / (Cx * Dx + 1e-6)

        B_kx = Bx * kappa_x
        atan_B_kx = np.arctan(B_kx)
        arg = B_kx - Ex * (B_kx - atan_B_kx)
        fx = Dx * np.sin(Cx * np.arctan(arg)) + Svx

        meta = {
            "mu_x": float(mu_x), "Dx": float(Dx), "Cx": float(Cx),
            "Bx": float(Bx), "Ex_mean": float(np.mean(Ex)),
            "Kx_kappa": float(Kx_kappa), "Shx": float(Shx), "Svx": float(Svx),
        }
        return fx, meta

    # --------------------------------------------------------------------------
    # 3. MOMENT STABILIZUJĄCY: Mz(alpha, Fz, gamma)
    # --------------------------------------------------------------------------
    def evaluate_mz(self, alpha_rad: np.ndarray, fz: float, gamma_rad: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
        """Oblicza moment stabilizujący Mz = -t * Fy0' + Mzr.
        
        Zwraca: (mz, trail_t, residual_mzr, meta).
        trail_t jest w metrach [m], mz i mzr w [Nm].
        """
        fz = max(float(fz), 1.0)
        fz0 = max(self.get("FNOMIN", 1000.0), 1.0)
        r0 = max(self.get("UNLOADED_RADIUS", 0.205), 0.05)
        dfz = (fz - fz0) / fz0
        g = float(gamma_rad)

        # 1. Wyznaczenie siły poprzecznej bez przesunięcia pionowego (Fy0')
        fy, fy_meta = self.evaluate_fy(alpha_rad, fz, gamma_rad)
        fy_prime = fy - fy_meta["Svy"]

        # 2. Ramię śladu pneumatycznego (Pneumatic Trail: t)
        Sht = self.get("QHZ1", 0.0) + self.get("QHZ2", 0.0) * dfz + (self.get("QHZ3", 0.0) + self.get("QHZ4", 0.0) * dfz) * g
        alpha_t = alpha_rad + Sht

        Ct = self.get("QCZ1", 1.25)
        Dt0 = fz * (r0 / fz0) * (self.get("QDZ1", 0.10) + self.get("QDZ2", -0.02) * dfz)
        Dt = Dt0 * (1.0 - self.get("QDZ3", 0.0) * abs(g))

        Bt = (self.get("QBZ1", 9.5) + self.get("QBZ2", -2.0) * dfz + self.get("QBZ3", 0.0) * (dfz ** 2)) * (
            1.0 + self.get("QBZ4", 0.0) * g + self.get("QBZ5", 0.4) * abs(g)
        )

        Et = (self.get("QEZ1", -0.8) + self.get("QEZ2", 0.0) * dfz + self.get("QEZ3", 0.0) * (dfz ** 2)) * (
            1.0 + (self.get("QEZ4", 0.0) + self.get("QEZ5", 0.0) * g) * (2.0 / np.pi) * np.arctan(Bt * Ct * alpha_t)
        )
        Et = np.clip(Et, -10.0, 1.0)

        B_at = Bt * alpha_t
        arg_t = B_at - Et * (B_at - np.arctan(B_at))
        trail_t = Dt * np.cos(Ct * np.arctan(arg_t))

        # 3. Moment resztkowy (Residual Torque: Mzr)
        Shr = Sht
        alpha_r = alpha_rad + Shr
        Dr = fz * r0 * ((self.get("QDZ6", 0.002) + self.get("QDZ7", 0.0) * dfz) + 
                        (self.get("QDZ8", 0.04) + self.get("QDZ9", 0.0) * dfz) * g)
        Br = self.get("QBZ9", 12.0) + self.get("QBZ10", 0.0) * fy_meta["By"] * fy_meta["Cy"]
        mzr = Dr * np.cos(np.arctan(Br * alpha_r))

        # 4. Całkowity moment stabilizujący (zgodnie z SAE / ISO Pacejka)
        mz = -trail_t * fy_prime + mzr

        meta = {
            "Dt_peak_trail_mm": float(Dt * 1000.0),
            "Dr_peak_residual_Nm": float(Dr),
            "max_trail_mm": float(np.max(np.abs(trail_t)) * 1000.0),
            "max_mz_Nm": float(np.max(np.abs(mz))),
        }
        return mz, trail_t, mzr, meta

    # --------------------------------------------------------------------------
    # 4. POŚLIZG ŁĄCZONY (COMBINED SLIP)
    # --------------------------------------------------------------------------
    def evaluate_combined_fy(self, alpha_rad: np.ndarray, kappa: float, fz: float, gamma_rad: float = 0.0) -> np.ndarray:
        """Fy przy jednoczesnym poślizgu wzdłużnym kappa: Fy = Fy0 * Gyk."""
        fy0, _ = self.evaluate_fy(alpha_rad, fz, gamma_rad)
        rBy1 = self.get("RBY1", 9.5)
        rBy2 = self.get("RBY2", 7.0)
        rCy1 = self.get("RCY1", 1.05)
        
        # Reduktor Gyk zależny od kappa i alpha
        arg_by = rBy1 * np.cos(np.arctan(rBy2 * alpha_rad))
        gyk = np.cos(rCy1 * np.arctan(arg_by * kappa))
        return fy0 * np.clip(gyk, 0.0, 1.2)

    def evaluate_combined_fx(self, kappa: np.ndarray, alpha_rad: float, fz: float, gamma_rad: float = 0.0) -> np.ndarray:
        """Fx przy jednoczesnym kącie znoszenia alpha: Fx = Fx0 * Gxa."""
        fx0, _ = self.evaluate_fx(kappa, fz, gamma_rad)
        rBx1 = self.get("RBX1", 11.0)
        rBx2 = self.get("RBX2", 8.0)
        rCx1 = self.get("RCX1", 1.05)

        arg_bx = rBx1 * np.cos(np.arctan(rBx2 * kappa))
        gxa = np.cos(rCx1 * np.arctan(arg_bx * alpha_rad))
        return fx0 * np.clip(gxa, 0.0, 1.2)

    # --------------------------------------------------------------------------
    # 5. IMPORT Z WYNIKÓW FITU OPTIMIZERA (TTC)
    # --------------------------------------------------------------------------
    def import_from_fit_results(self, fit_results: Dict[str, Any]) -> List[str]:
        """Mapuje parametry dopasowane w optimizerze TTC na odpowiadające współczynniki MF 5.1."""
        imported = []
        if not fit_results:
            return imported

        # 1. Z dopasowania 3D Fy_load (pCy1, pDy1, pDy2, pEy1, pKy1, pKy2, pUy1, pVy1, fz0)
        if "Fy_load" in fit_results:
            r = fit_results["Fy_load"]
            p = r.get("parameters", r)
            mapping = {
                "pCy1": "PCY1", "pDy1": "PDY1", "pDy2": "PDY2",
                "pEy1": "PEY1", "pKy1": "PKY1", "pKy2": "PKY2",
                "pVy1": "PVY1", "fz0": "FNOMIN"
            }
            for src, dst in mapping.items():
                if src in p:
                    self.set(dst, float(p[src]))
                    imported.append(f"{dst} <= {src} ({float(p[src]):.4g})")
            # pUy1 w optimizerze to radiany, odpowiada PHY1
            if "pUy1" in p:
                self.set("PHY1", float(p["pUy1"]))
                imported.append(f"PHY1 <= pUy1 ({float(p['pUy1']):.4g} rad)")

        # 2. Z czystego poślizgu Fy0 (B, C, D, E, Sh, Sv)
        elif "Fy0" in fit_results:
            r = fit_results["Fy0"]
            p = r.get("parameters", r)
            if "C" in p: self.set("PCY1", float(p["C"]))
            if "E" in p: self.set("PEY1", float(p["E"]))
            if "Sh" in p: self.set("PHY1", float(p["Sh"]))
            fz0 = self.get("FNOMIN", 1000.0)
            if "D" in p:
                self.set("PDY1", float(p["D"]) / fz0)
            if "B" in p and "C" in p and "D" in p:
                # K = B * C * D
                stiffness = float(p["B"]) * float(p["C"]) * float(p["D"])
                self.set("PKY1", stiffness / fz0)
            imported.append("Imported base Fy0 parameters (C, E, Sh, mu_y, PKY1)")

        # 3. Z czystego poślizgu Fx0
        if "Fx0" in fit_results:
            r = fit_results["Fx0"]
            p = r.get("parameters", r)
            if "C" in p: self.set("PCX1", float(p["C"]))
            if "E" in p: self.set("PEX1", float(p["E"]))
            if "Sh" in p: self.set("PHX1", float(p["Sh"]))
            fz0 = self.get("FNOMIN", 1000.0)
            if "D" in p: self.set("PDX1", float(p["D"]) / fz0)
            if "B" in p and "C" in p and "D" in p:
                stiffness = float(p["B"]) * float(p["C"]) * float(p["D"])
                self.set("PKX1", stiffness / fz0)
            imported.append("Imported base Fx0 parameters (Cx, Ex, Shx, mu_x, PKX1)")

        # 4. Z poślizgu łączonego (Combined Slip Gyk / Gxa)
        if "Combined" in fit_results:
            r = fit_results["Combined"]
            p = r.get("parameters", r)
            if "rBy1" in p: self.set("RBY1", float(p["rBy1"]))
            if "rCy1" in p: self.set("RCY1", float(p["rCy1"]))
            if "rBx1" in p: self.set("RBX1", float(p["rBx1"]))
            if "rCx1" in p: self.set("RCX1", float(p["rCx1"]))
            imported.append("Imported Combined Slip parameters (RBY1, RCY1, RBX1, RCX1)")

        return imported

    # --------------------------------------------------------------------------
    # 6. GENERATOR PLIKU .TIR (Standard TYDEX / PAC2002)
    # --------------------------------------------------------------------------
    def to_tir_content(self, tyre_name: str = "PUTM_Racing_FSAE_MF51") -> str:
        """Generuje pełną zawartość standardowego pliku modelu opony .tir (PAC2002)."""
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "! ==================================================================",
            f"! TYDEX / PAC2002 TIRE PROPERTY FILE: {tyre_name}",
            f"! Generated by PUTM Tire - Magic Formula 5.1 / PAC2002 Creator",
            f"! Date: {now}",
            f"! Standard: Formula Student FSAE (16x7.5-10 Hoosier / 10-inch rim)",
            "! Note: In accordance with Adams/TYDEX/ISO, PKY1 and PKY4 are exported negative.",
            "! ==================================================================",
            "[MDI_HEADER]",
            "FILE_TYPE                ='tir'",
            "FILE_VERSION             =2.0",
            "FILE_FORMAT              ='ASCII'",
            f"! : TIRE_VERSION :       'PAC2002'",
            f"! : COMMENT :            'FSAE tire model generated in PUTM Tire'",
            "",
            "[UNITS]",
            "LENGTH                   ='meter'",
            "FORCE                    ='newton'",
            "ANGLE                    ='radians'",
            "MASS                     ='kg'",
            "TIME                     ='second'",
            "PRESSURE                 ='pascal'",
            "",
            "[MODEL]",
            "PROPERTY_FILE_FORMAT     ='PAC2002'",
            "USE_MODE                 =14",
            "VXLOW                    =1.0",
            "LONGVL                   =16.6",
            "TYRESIDE                 ='RIGHT'",
            "",
            "[DIMENSION]",
            f"UNLOADED_RADIUS          ={self.get('UNLOADED_RADIUS', 0.205):.4f}",
            f"WIDTH                    ={self.get('WIDTH', 0.180):.4f}",
            f"ASPECT_RATIO             ={self.get('ASPECT_RATIO', 0.50):.2f}",
            f"RIM_RADIUS               ={self.get('RIM_RADIUS', 0.127):.4f}",
            f"RIM_WIDTH                ={self.get('RIM_WIDTH', 0.178):.4f}",
            "",
            "[PARAMETER]",
            f"NOMPRES                  ={self.get('NOMPRES', 82737.0):.1f}",
            f"INFLPRES                 ={self.get('INFLPRES', 82737.0):.1f}",
            "",
            "[VERTICAL]",
            f"FNOMIN                   ={self.get('FNOMIN', 1000.0):.1f}",
            f"VERTICAL_STIFFNESS       ={self.get('VERTICAL_STIFFNESS', 125000.0):.1f}",
            f"VERTICAL_DAMPING         ={self.get('VERTICAL_DAMPING', 500.0):.1f}",
            f"BREFF                    ={self.get('BREFF', 8.0):.4f}",
            f"DREFF                    ={self.get('DREFF', 0.25):.4f}",
            f"FREFF                    ={self.get('FREFF', 0.07):.4f}",
            f"Q_RE0                    ={self.get('Q_RE0', 1.0):.4f}",
            f"Q_V1                     ={self.get('Q_V1', 0.0):.6f}",
            f"Q_V2                     ={self.get('Q_V2', 0.0):.6f}",
            f"Q_FZ1                    ={self.get('Q_FZ1', 0.0):.6f}",
            f"Q_FZ2                    ={self.get('Q_FZ2', 0.0):.6f}",
            f"Q_FCX                    ={self.get('Q_FCX', 0.0):.6f}",
            f"Q_FCY                    ={self.get('Q_FCY', 0.0):.6f}",
            "",
            "[STRUCTURAL]",
            "MASS                     =5.2",
            "IXX                      =0.12",
            "IYY                      =0.21",
            "BELT_MASS                =3.5",
            "",
            "[INFLATION_PRESSURE_RANGE]",
            "PRESMIN                  =50000.0",
            "PRESMAX                  =150000.0",
            "",
            "[LONG_SLIP_RANGE]",
            "KPUMIN                   =-1.0",
            "KPUMAX                   =1.0",
            "",
            "[SLIP_ANGLE_RANGE]",
            "ALPMIN                   =-0.5236",
            "ALPMAX                   =0.5236",
            "",
            "[INCLINATION_ANGLE_RANGE]",
            "CAMMIN                   =-0.1745",
            "CAMMAX                   =0.1745",
            "",
            "[VERTICAL_FORCE_RANGE]",
            "FZMIN                    =50.0",
            "FZMAX                    =3500.0",
            "",
            "[SCALING_COEFFICIENTS]",
            f"LFZO                     ={self.get('LFZO', 1.0):.4f}",
            f"LCX                      ={self.get('LCX', 1.0):.4f}",
            f"LMUX                     ={self.get('LMUX', 0.65):.4f}",
            f"LEX                      ={self.get('LEX', 1.0):.4f}",
            f"LKX                      ={self.get('LKX', 1.0):.4f}",
            f"LHX                      ={self.get('LHX', 1.0):.4f}",
            f"LVX                      ={self.get('LVX', 1.0):.4f}",
            f"LGAX                     ={self.get('LGAX', 1.0):.4f}",
            f"LCY                      ={self.get('LCY', 1.0):.4f}",
            f"LMUY                     ={self.get('LMUY', 0.65):.4f}",
            f"LEY                      ={self.get('LEY', 1.0):.4f}",
            f"LKY                      ={self.get('LKY', 1.0):.4f}",
            f"LHY                      ={self.get('LHY', 1.0):.4f}",
            f"LVY                      ={self.get('LVY', 1.0):.4f}",
            f"LGAY                     ={self.get('LGAY', 1.0):.4f}",
            f"LTR                      ={self.get('LTR', 1.0):.4f}",
            f"LRES                     ={self.get('LRES', 1.0):.4f}",
            f"LGAZ                     ={self.get('LGAZ', 1.0):.4f}",
            f"LXAL                     ={self.get('LXAL', 1.0):.4f}",
            f"LYKA                     ={self.get('LYKA', 1.0):.4f}",
            f"LVYKA                    ={self.get('LVYKA', 1.0):.4f}",
            f"LS                       ={self.get('LS', 1.0):.4f}",
            "",
            "[PRESSURE_COEFFICIENTS]",
            f"PPX1                     ={self.get('PPX1', 0.0):.6f}",
            f"PPX2                     ={self.get('PPX2', 0.0):.6f}",
            f"PPX3                     ={self.get('PPX3', 0.0):.6f}",
            f"PPX4                     ={self.get('PPX4', 0.0):.6f}",
            f"PPY1                     ={self.get('PPY1', 0.0):.6f}",
            f"PPY2                     ={self.get('PPY2', 0.0):.6f}",
            f"PPY3                     ={self.get('PPY3', 0.0):.6f}",
            f"PPY4                     ={self.get('PPY4', 0.0):.6f}",
            f"PPY5                     ={self.get('PPY5', 0.0):.6f}",
            f"PPZ1                     ={self.get('PPZ1', 0.0):.6f}",
            f"PPZ2                     ={self.get('PPZ2', 0.0):.6f}",
            "",
            "[LONGITUDINAL_COEFFICIENTS]",
            f"PCX1                     ={self.get('PCX1', 1.65):.6f}",
            f"PDX1                     ={self.get('PDX1', 1.75):.6f}",
            f"PDX2                     ={self.get('PDX2', -0.15):.6f}",
            f"PDX3                     ={self.get('PDX3', 0.0):.6f}",
            f"PEX1                     ={self.get('PEX1', -0.4):.6f}",
            f"PEX2                     ={self.get('PEX2', 0.0):.6f}",
            f"PEX3                     ={self.get('PEX3', 0.0):.6f}",
            f"PEX4                     ={self.get('PEX4', 0.0):.6f}",
            f"PKX1                     ={self.get('PKX1', 26.0):.6f}",
            f"PKX2                     ={self.get('PKX2', 0.0):.6f}",
            f"PKX3                     ={self.get('PKX3', 0.0):.6f}",
            f"PHX1                     ={self.get('PHX1', 0.0):.6f}",
            f"PHX2                     ={self.get('PHX2', 0.0):.6f}",
            f"PVX1                     ={self.get('PVX1', 0.0):.6f}",
            f"PVX2                     ={self.get('PVX2', 0.0):.6f}",
            f"RBX1                     ={self.get('RBX1', 11.0):.6f}",
            f"RBX2                     ={self.get('RBX2', 8.0):.6f}",
            f"RCX1                     ={self.get('RCX1', 1.05):.6f}",
            f"REX1                     ={self.get('REX1', 0.0):.6f}",
            "",
            "[LATERAL_COEFFICIENTS]",
            f"PCY1                     ={self.get('PCY1', 1.30):.6f}",
            f"PDY1                     ={self.get('PDY1', 1.65):.6f}",
            f"PDY2                     ={self.get('PDY2', -0.15):.6f}",
            f"PDY3                     ={self.get('PDY3', 0.0):.6f}",
            f"PEY1                     ={self.get('PEY1', -0.5):.6f}",
            f"PEY2                     ={self.get('PEY2', 0.0):.6f}",
            f"PEY3                     ={self.get('PEY3', 0.0):.6f}",
            f"PEY4                     ={self.get('PEY4', 0.0):.6f}",
            f"PKY1                     ={-abs(self.get('PKY1', 38.0)):.6f}",
            f"PKY2                     ={self.get('PKY2', 1.8):.6f}",
            f"PKY3                     ={self.get('PKY3', 0.0):.6f}",
            f"PKY4                     ={-abs(self.get('PKY4', 2.20)):.6f}",
            f"PKY5                     ={self.get('PKY5', 0.0):.6f}",
            f"PKY6                     ={self.get('PKY6', 0.0):.6f}",
            f"PKY7                     ={self.get('PKY7', 0.0):.6f}",
            f"PHY1                     ={self.get('PHY1', 0.0):.6f}",
            f"PHY2                     ={self.get('PHY2', 0.0):.6f}",
            f"PHY3                     ={self.get('PHY3', 0.04):.6f}",
            f"PVY1                     ={self.get('PVY1', 0.0):.6f}",
            f"PVY2                     ={self.get('PVY2', 0.0):.6f}",
            f"PVY3                     ={self.get('PVY3', 0.12):.6f}",
            f"PVY4                     ={self.get('PVY4', 0.0):.6f}",
            f"RBY1                     ={self.get('RBY1', 9.5):.6f}",
            f"RBY2                     ={self.get('RBY2', 7.0):.6f}",
            f"RCY1                     ={self.get('RCY1', 1.05):.6f}",
            f"REY1                     ={self.get('REY1', 0.0):.6f}",
            "",
            "[ALIGNING_COEFFICIENTS]",
            f"QBZ1                     ={self.get('QBZ1', 9.5):.6f}",
            f"QBZ2                     ={self.get('QBZ2', -2.0):.6f}",
            f"QBZ3                     ={self.get('QBZ3', 0.0):.6f}",
            f"QBZ4                     ={self.get('QBZ4', 0.0):.6f}",
            f"QBZ5                     ={self.get('QBZ5', 0.4):.6f}",
            f"QCZ1                     ={self.get('QCZ1', 1.25):.6f}",
            f"QDZ1                     ={self.get('QDZ1', 0.10):.6f}",
            f"QDZ2                     ={self.get('QDZ2', -0.02):.6f}",
            f"QDZ3                     ={self.get('QDZ3', 0.15):.6f}",
            f"QEZ1                     ={self.get('QEZ1', -0.8):.6f}",
            f"QEZ2                     ={self.get('QEZ2', 0.0):.6f}",
            f"QEZ3                     ={self.get('QEZ3', 0.0):.6f}",
            f"QEZ4                     ={self.get('QEZ4', 0.0):.6f}",
            f"QEZ5                     ={self.get('QEZ5', 0.0):.6f}",
            f"QHZ1                     ={self.get('QHZ1', 0.0):.6f}",
            f"QHZ2                     ={self.get('QHZ2', 0.0):.6f}",
            f"QHZ3                     ={self.get('QHZ3', 0.0):.6f}",
            f"QHZ4                     ={self.get('QHZ4', 0.0):.6f}",
            f"QDZ6                     ={self.get('QDZ6', 0.002):.6f}",
            f"QDZ7                     ={self.get('QDZ7', 0.0):.6f}",
            f"QDZ8                     ={self.get('QDZ8', 0.04):.6f}",
            f"QDZ9                     ={self.get('QDZ9', 0.0):.6f}",
            f"QBZ9                     ={self.get('QBZ9', 12.0):.6f}",
            f"QBZ10                    ={self.get('QBZ10', 0.0):.6f}",
            "",
            "[OVERTURNING_COEFFICIENTS]",
            "QSX1                     =0.0",
            "QSX2                     =0.0",
            "QSX3                     =0.0",
            "",
            "[ROLLING_COEFFICIENTS]",
            "QSY1                     =0.012",
            "QSY2                     =0.0",
            "QSY3                     =0.0",
            "QSY4                     =0.0",
        ]
        return "\n".join(lines) + "\n"
