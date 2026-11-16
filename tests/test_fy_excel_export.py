import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import openpyxl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from condition_export import center_bins, range_bins
from fy_excel_export import (
    DEFAULT_FZ_TOL,
    DEFAULT_IA_TOL,
    FyExcelSpec,
    build_fy_excel_from_df,
    format_export_summary,
    generate_excel_workbook,
    make_sa_grid,
    save_fy_excel,
)


class TestFyExcelExport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

        # Generowanie syntetycznych danych Cornering
        rng = np.random.default_rng(42)
        n = 800
        # Dwa poziomy Fz: 800 i 1000 N
        fz = rng.choice([800.0, 1000.0], n)
        ia = np.zeros(n)
        # Dwa poziomy ciśnienia: 55 kPa i 70 kPa
        p = rng.choice([55.0, 70.0], n)
        sa = rng.uniform(-10.0, 10.0, n)
        # Model Pacejka-like: Fy = mu * Fz * sin(1.3 * arctan(B * sa))
        mu = 1.5 + (p - 55.0) * 0.005 # niewielki wpływ ciśnienia
        # TTC: FY jest ujemne przy dodatnim SA
        fy_raw = -(mu * fz * np.sin(1.3 * np.arctan(15.0 * np.deg2rad(sa))) + rng.normal(0, 15, n))

        self.df = pd.DataFrame({
            "FZ": fz,
            "IA": ia,
            "P_kPa": p,
            "SA": sa,
            "FY": fy_raw,
            "Fz_bin_N": fz.astype(int).astype(str),
            "IA_bin_deg": "0",
            "P_bin_kPa": np.where(p < 60, "50-60", "65-75"),
            "__type__": "Cornering",
        })

    def test_make_sa_grid(self):
        grid = make_sa_grid(-10.0, 10.0, 1.0)
        self.assertEqual(len(grid), 21)
        self.assertEqual(grid[0], -10.0)
        self.assertEqual(grid[-1], 10.0)

        with self.assertRaises(ValueError):
            make_sa_grid(10.0, -10.0, 1.0)

        with self.assertRaises(ValueError):
            make_sa_grid(-5.0, 5.0, 0.0)

    def test_build_fy_excel_from_df_single_fz(self):
        spec = FyExcelSpec(
            fz_bins=center_bins((1000.0,), DEFAULT_FZ_TOL),
            ia_bins=center_bins((0.0,), DEFAULT_IA_TOL),
            p_bins=range_bins(((50.0, 60.0), (65.0, 75.0))),
            p_to_kpa=1.0,
            p_unit_label="kPa",
            nearest_only=True,
            sa_min=-10.0,
            sa_max=10.0,
            sa_step=1.0,
        )

        result = build_fy_excel_from_df(self.df, spec)
        self.assertEqual(len(result.conditions), 2) # (1000N, 50-60) i (1000N, 65-75)
        self.assertGreater(result.cells_fitted, 0)
        self.assertIn("Fz [N]", format_export_summary(result))

        # Zapis do pliku Excel
        out_path = os.path.join(self.tmp.name, "test_fy_single.xlsx")
        save_fy_excel(out_path, result)
        self.assertTrue(os.path.exists(out_path))

        # Weryfikacja skoroszytu openpyxl
        wb = openpyxl.load_workbook(out_path)
        sheet_names = wb.sheetnames
        self.assertTrue(any("Fz=1000N" in name for name in sheet_names))
        self.assertIn("Summary_KPI", sheet_names)
        self.assertIn("Metadata_Info", sheet_names)

        ws = wb["Fy(SA) @ Fz=1000N"]
        self.assertEqual(ws["A4"].value, "Slip Angle SA [deg]")
        self.assertIn("Fy [N]", str(ws["B4"].value))
        self.assertIn("Fy [N]", str(ws["C4"].value))
        self.assertGreater(len(ws._charts), 0)

    def test_build_fy_excel_from_df_multi_fz(self):
        spec = FyExcelSpec(
            fz_bins=center_bins((800.0, 1000.0), DEFAULT_FZ_TOL),
            ia_bins=center_bins((0.0,), DEFAULT_IA_TOL),
            p_bins=range_bins(((50.0, 60.0), (65.0, 75.0))),
            p_to_kpa=1.0,
            p_unit_label="kPa",
            nearest_only=True,
            sa_min=-8.0,
            sa_max=8.0,
            sa_step=1.0,
        )

        result = build_fy_excel_from_df(self.df, spec)
        self.assertEqual(len(result.conditions), 4)
        self.assertEqual(result.cells_fitted, 4)

        out_path = os.path.join(self.tmp.name, "test_fy_multi.xlsx")
        save_fy_excel(out_path, result)
        self.assertTrue(os.path.exists(out_path))

        wb = openpyxl.load_workbook(out_path)
        self.assertIn("Matrix_Fy_SA_Fz_P", wb.sheetnames)
        self.assertTrue(any("Fz=800N" in name for name in wb.sheetnames))
        self.assertTrue(any("Fz=1000N" in name for name in wb.sheetnames))

    def test_end_to_end_mat_file(self):
        import scipy.io
        from tire_model import RunConfig
        from fy_excel_export import build_fy_excel_data

        n = 400
        rng = np.random.default_rng(123)
        fz = rng.choice([-1000.0], n)
        ia = np.zeros(n)
        p = rng.choice([55.0, 70.0], n)
        sa = np.linspace(-10, 10, n)
        fy = -1200.0 * np.sin(1.3 * np.arctan(12.0 * np.deg2rad(sa)))

        mat_path = os.path.join(self.tmp.name, "cornering_run.mat")
        scipy.io.savemat(mat_path, {
            "SA": sa, "FY": fy, "FZ": fz, "IA": ia, "P": p, "ET": np.arange(n, dtype=float)
        })

        run = RunConfig(filepath=mat_path, test_type="Cornering", target_fz=1000, tol_fz=200, target_param=0, tol_param=0.5)
        spec = FyExcelSpec(
            fz_bins=center_bins((1000.0,), DEFAULT_FZ_TOL),
            ia_bins=center_bins((0.0,), DEFAULT_IA_TOL),
            p_bins=range_bins(((50.0, 60.0), (65.0, 75.0))),
            p_to_kpa=1.0,
            p_unit_label="kPa",
            nearest_only=True,
            sa_min=-8.0,
            sa_max=8.0,
            sa_step=1.0,
        )

        result = build_fy_excel_data([run], spec)
        self.assertEqual(result.files_used, 1)
        self.assertEqual(result.cells_fitted, 2)

        out_path = os.path.join(self.tmp.name, "mat_export.xlsx")
        save_fy_excel(out_path, result)
        self.assertTrue(os.path.exists(out_path))


if __name__ == "__main__":
    unittest.main()
