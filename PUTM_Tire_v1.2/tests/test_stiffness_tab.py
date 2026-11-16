"""Testy jednostkowe dla zakladki sztywnosci skretnej i ladowania plikow TTC."""

from __future__ import annotations

import os
import tempfile
import tkinter as tk
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from scipy.io import savemat

from condition_export import ConditionSpec, classify_frame, center_bins, range_bins
from hub_tab import generate_fsae_demo_telemetry
import stiffness_tab as st_mod
from stiffness_tab import (
    PLOT_MODES,
    StiffnessTab,
    _sort_bin_labels,
    compute_linear_stiffness,
    detect_pressure_multiplier,
)
from tire_model import RunConfig, TireDataManager


class TestRunConfigAndUnfilteredLoading(unittest.TestCase):
    """Testuje RunConfig z domyslnymi polami i wczytywanie .MAT / .CSV bez filtrow."""

    def test_runconfig_defaults_and_unfiltered_mat(self) -> None:
        cfg = RunConfig(filepath="dummy.mat", test_type="Cornering")
        self.assertEqual(cfg.target_fz, 0.0)
        self.assertEqual(cfg.tol_fz, 0.0)
        self.assertEqual(cfg.target_param, 0.0)
        self.assertEqual(cfg.tol_param, 0.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            mat_path = os.path.join(tmpdir, "cornering.mat")
            sa = np.linspace(-4.0, 4.0, 25)
            savemat(
                mat_path,
                {
                    "SA": sa,
                    "FY": -500.0 * sa,
                    "FZ": np.full_like(sa, 900.0),
                    "IA": np.zeros_like(sa),
                    "P": np.full_like(sa, 82.7),
                },
            )
            mgr = TireDataManager()
            df = mgr.load_unfiltered(RunConfig(filepath=mat_path, test_type="Cornering"))
            self.assertEqual(len(df), 25)
            self.assertTrue({"SA", "FY", "FZ", "IA", "P"}.issubset(df.columns))

    def test_unfiltered_mat_struct_and_missing_ia(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mat_path = os.path.join(tmpdir, "struct_no_ia.mat")
            sa = np.linspace(-2.0, 2.0, 15)
            savemat(
                mat_path,
                {
                    "ttc_run": {
                        "SA": sa,
                        "FY": -420.0 * sa,
                        "FZ": np.full_like(sa, 800.0),
                    }
                },
            )
            mgr = TireDataManager()
            df = mgr.load_unfiltered(RunConfig(filepath=mat_path, test_type="Cornering"))
            self.assertEqual(len(df), 15)
            self.assertIn("SA", df.columns)
            self.assertIn("FY", df.columns)
            self.assertIn("FZ", df.columns)
            self.assertIn("IA", df.columns)
            self.assertTrue(df["IA"].isna().all())

    def test_unfiltered_csv_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "cornering.csv")
            sa = np.linspace(-3.0, 3.0, 20)
            pd.DataFrame(
                {
                    "SA": sa,
                    "FY": -600.0 * sa,
                    "FZ": np.full_like(sa, 1000.0),
                    "IA": np.full_like(sa, -2.0),
                    "P": np.full_like(sa, 12.0),
                }
            ).to_csv(csv_path, index=False)
            mgr = TireDataManager()
            df = mgr.load_unfiltered(RunConfig(filepath=csv_path, test_type="Cornering"))
            self.assertEqual(len(df), 20)
            self.assertAlmostEqual(float(df["P"].median()), 12.0, places=3)


class TestStiffnessHelpersAndCalculation(unittest.TestCase):
    """Testuje wykrywanie jednostek P, sortowanie koszykow i regresje C_Fa."""

    def test_detect_pressure_multiplier_auto_and_explicit(self) -> None:
        mult_bar, desc_bar = detect_pressure_multiplier(pd.DataFrame({"P": [1.1, 1.2]}), "Auto")
        self.assertAlmostEqual(mult_bar, 100.0)
        self.assertIn("bar", desc_bar)

        mult_psi, desc_psi = detect_pressure_multiplier(pd.DataFrame({"P": [11.8, 12.2]}), "Auto")
        self.assertAlmostEqual(mult_psi, 6.894757, places=4)
        self.assertIn("psi", desc_psi)

        mult_kpa, desc_kpa = detect_pressure_multiplier(pd.DataFrame({"P": [82.0, 83.0]}), "Auto")
        self.assertAlmostEqual(mult_kpa, 1.0)
        self.assertIn("kPa", desc_kpa)

        mult_none, _ = detect_pressure_multiplier(pd.DataFrame({"SA": [0.0, 1.0]}), "Auto")
        self.assertAlmostEqual(mult_none, 1.0)

        mult_exp, _ = detect_pressure_multiplier(pd.DataFrame({"P": [82.0]}), "psi")
        self.assertAlmostEqual(mult_exp, 6.894757, places=4)

    def test_sort_bin_labels_numeric_order(self) -> None:
        df = pd.DataFrame(
            {
                "Fz_bin_N": ["1200", "600", "1000", "800"],
                "Fz_mean_N": [1202.0, 601.0, 998.0, 799.0],
                "IA_bin_deg": ["0", "-3", "-1", "-2"],
                "IA_mean_deg": [0.0, -3.0, -1.0, -2.0],
            }
        )
        self.assertEqual(
            _sort_bin_labels(df, "Fz_bin_N", "Fz_mean_N"),
            ["600", "800", "1000", "1200"],
        )
        self.assertEqual(
            _sort_bin_labels(df, "IA_bin_deg", "IA_mean_deg"),
            ["-3", "-2", "-1", "0"],
        )

    def test_compute_linear_stiffness_and_dwell_rejection(self) -> None:
        sa_lin = np.linspace(-1.4, 1.4, 10)
        df_lin = pd.DataFrame(
            {
                "SA": sa_lin,
                "FY": -650.0 * sa_lin + 10.0,
                "FZ": np.full_like(sa_lin, 800.0),
                "IA": np.zeros_like(sa_lin),
                "P": np.full_like(sa_lin, 82.0),
                "Fz_bin_N": "800",
                "IA_bin_deg": "0",
                "P_bin_kPa": "75-95",
                "is_out_of_range": False,
            }
        )

        # Koszyk postoju przy SA ~ 0 (rozpietosc 0.04 deg < 0.4 deg) nie powinien przejsc
        sa_dwell = np.linspace(-0.02, 0.02, 12)
        df_dwell = pd.DataFrame(
            {
                "SA": sa_dwell,
                "FY": np.linspace(-200.0, 200.0, 12),
                "FZ": np.full_like(sa_dwell, 1000.0),
                "IA": np.zeros_like(sa_dwell),
                "P": np.full_like(sa_dwell, 82.0),
                "Fz_bin_N": "1000",
                "IA_bin_deg": "0",
                "P_bin_kPa": "75-95",
                "is_out_of_range": False,
            }
        )

        res = compute_linear_stiffness(
            pd.concat([df_lin, df_dwell], ignore_index=True),
            sa_max_deg=1.5,
            min_points=6,
            min_sa_span_deg=0.4,
        )
        self.assertEqual(len(res), 1)
        self.assertEqual(res.iloc[0]["Fz_bin_N"], "800")
        self.assertAlmostEqual(float(res.iloc[0]["cfa_N_deg"]), 650.0, places=5)
        self.assertAlmostEqual(
            float(res.iloc[0]["cfa_N_rad"]),
            650.0 * 180.0 / np.pi,
            places=3,
        )

    def test_speed_dwell_and_combined_slip_rejection_and_plateau_clustering(self):
        sa = np.linspace(-1.4, 1.4, 24)
        df_clean = pd.DataFrame(
            {
                "SA": sa,
                "FY": -520.0 * sa,
                "FZ": np.full_like(sa, 890.0),
                "IA": np.zeros_like(sa),
                "P": np.full_like(sa, 82.7),
                "V": np.full_like(sa, 40.0),
                "SL": np.zeros_like(sa),
                "Fz_bin_N": "890",
                "IA_bin_deg": "0",
                "P_bin_kPa": "76-89",
            }
        )
        # Contaminated stationary (V=0) + combined braking (SL=-0.18) samples inside same bin
        sa_bad = np.linspace(-1.4, 1.4, 16)
        df_bad = pd.DataFrame(
            {
                "SA": sa_bad,
                "FY": np.zeros_like(sa_bad),
                "FZ": np.full_like(sa_bad, 890.0),
                "IA": np.zeros_like(sa_bad),
                "P": np.full_like(sa_bad, 82.7),
                "V": np.concatenate([np.zeros(8), np.full(8, 40.0)]),
                "SL": np.concatenate([np.zeros(8), np.full(8, -0.18)]),
                "Fz_bin_N": "890",
                "IA_bin_deg": "0",
                "P_bin_kPa": "76-89",
            }
        )
        res = compute_linear_stiffness(pd.concat([df_clean, df_bad], ignore_index=True), sa_max_deg=1.5)
        self.assertEqual(len(res), 1)
        self.assertAlmostEqual(float(res.iloc[0]["cfa_N_deg"]), 520.0, delta=1.0)
        self.assertAlmostEqual(float(res.iloc[0]["r2"]), 1.0, places=4)

        # Plateau clustering with transition ramps between 222 N, 667 N, 1112 N
        plateaus = np.concatenate([
            np.full(400, 222.0),
            np.linspace(222.0, 667.0, 25),
            np.full(400, 667.0),
            np.linspace(667.0, 1112.0, 25),
            np.full(400, 1112.0),
        ])
        levels = st_mod._cluster_values(plateaus, threshold=110.0, round_step=50.0)
        self.assertEqual(levels, [200.0, 650.0, 1100.0])

    def test_fsae_demo_stiffness_computation(self) -> None:
        df_demo, _ = generate_fsae_demo_telemetry()
        df_demo = df_demo.copy()
        df_demo["P_kPa"] = df_demo["P"]
        spec = ConditionSpec(
            fz_bins=center_bins([450.0, 900.0, 1350.0], 150.0),
            ia_bins=center_bins([0.0, 2.0, 4.0], 0.5),
            p_bins=range_bins([(75.0, 95.0)]),
            nearest_only=True,
        )
        classified, _counts = classify_frame(df_demo, spec)
        res = compute_linear_stiffness(classified, sa_max_deg=1.5, min_points=6)
        self.assertGreaterEqual(len(res), 6)
        self.assertTrue((res["cfa_N_deg"] > 100.0).all())


class TestStiffnessTabHeadlessUI(unittest.TestCase):
    """Testy integracyjne GUI StiffnessTab (Demo FSAE, porownanie A/B, 5 wykresow, wyczysc)."""

    def test_workflow_auto_bins_all_plot_modes_and_clear(self) -> None:
        try:
            root = tk.Tk()
            root.withdraw()
        except tk.TclError as exc:
            self.skipTest(f"Tkinter display niedostepny: {exc}")
            return

        self.addCleanup(root.destroy)

        with mock.patch.object(st_mod.messagebox, "showwarning"), mock.patch.object(
            st_mod.messagebox, "showerror"
        ):
            tab = StiffnessTab(
                root,
                get_df_cornering=lambda: generate_fsae_demo_telemetry()[0],
                get_runs=lambda: [],
                show_step_header=False,
            )
            tab.pack(fill="both", expand=True)

            # Import z Kroku 1 (Demo FSAE) powinien automatycznie dopasowac koszyki i policzyc C_Fa
            tab.use_app_cornering_as_a()
            self.assertIsNotNone(tab._raw_a)
            self.assertFalse(tab._res_a.empty)

            # Zaladuj Opona B z lekko wyzszym Fy, zeby sprawdzic porownanie A vs B
            df_b = generate_fsae_demo_telemetry()[0].copy()
            df_b["FY"] = df_b["FY"] * 1.12
            tab._raw_b = df_b
            tab.var_name_b.set("Test Tire B")
            tab.compute_and_refresh()
            self.assertFalse(tab._res_b.empty)
            self.assertFalse(tab._gain_df.empty)

            # Kazdy z 5 trybow wykresu musi narysowac sie bez bledu
            for mode in PLOT_MODES:
                tab.var_plot_mode.set(mode)
                tab.draw()
                self.assertGreater(len(tab.fig.axes), 0)

            # Wyczyszczenie Opony B musi przywrocic pusty stan dla B i dalej rysowac A
            tab._clear_tire("B")
            self.assertIsNone(tab._raw_b)
            self.assertTrue(tab._res_b.empty)
            self.assertEqual(tab.var_path_b.get(), "")
            tab.var_plot_mode.set(PLOT_MODES[0])
            tab.draw()
            self.assertGreater(len(tab.fig.axes), 0)


if __name__ == "__main__":
    unittest.main()
