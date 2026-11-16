import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from tire_model import RunConfig, TireDataManager


class TireDataManagerTests(unittest.TestCase):
    def test_drive_filter_preserves_operating_conditions_and_source_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.mat"
            savemat(
                path,
                {
                    "FX": np.array([100.0, 200.0, 300.0, 400.0]),
                    "FY": np.array([-500.0, -510.0, -520.0, -530.0]),
                    "FZ": np.array([-995.0, -1005.0, -1000.0, -1300.0]),
                    "SL": np.array([-0.1, 0.0, 0.1, 0.0]),
                    "SA": np.array([-3.0, -3.0, 0.0, -3.0]),
                    "IA": np.zeros(4),
                    "ET": np.arange(4, dtype=float),
                },
            )
            config = RunConfig(
                str(path), "Drive/Brake", 1000.0, 20.0, 0.0, 0.2,
                target_sa=-3.0, tol_sa=0.2, target_ia=0.0, tol_ia=0.1,
            )

            frame = TireDataManager.load_and_filter(config)
            raw = TireDataManager.load_unfiltered(config)

            self.assertEqual(frame["__row__"].tolist(), [0, 1])
            self.assertEqual(len(raw), 4)
            self.assertEqual(raw["FZ"].tolist(), [995.0, 1005.0, 1000.0, 1300.0])
            self.assertEqual(frame["SA"].tolist(), [-3.0, -3.0])
            self.assertEqual(frame["FZ"].tolist(), [995.0, 1005.0])
            self.assertEqual(frame["FZ_RAW"].tolist(), [-995.0, -1005.0])
            self.assertIn("ET", frame)

    def test_unknown_test_type_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.mat"
            savemat(path, {"FZ": np.array([-1000.0])})
            config = RunConfig(str(path), "Unknown", 1000.0, 20.0, 0.0, 0.1)
            with self.assertRaisesRegex(ValueError, "Unsupported test type"):
                TireDataManager.load_and_filter(config)

    def test_partial_condition_filter_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "SA filter"):
            RunConfig("unused.mat", "Drive/Brake", 1000.0, 20.0, 0.0, 0.2, target_sa=0.0)

    def test_cornering_can_be_loaded_without_optional_mz(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corner.mat"
            savemat(
                path,
                {
                    "SA": np.array([-2.0, 0.0, 2.0]),
                    "FY": np.array([500.0, 0.0, -500.0]),
                    "FZ": np.array([-1000.0, -1000.0, -1000.0]),
                    "IA": np.zeros(3),
                },
            )
            frame = TireDataManager.load_and_filter(RunConfig(str(path), "Cornering", 1000.0, 20.0, 0.0, 0.1))
            self.assertEqual(len(frame), 3)
            self.assertTrue(frame["MZ"].isna().all())

    def test_mat_scalar_constant_broadcast_and_nested_struct(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested_scalar.mat"
            sa = np.linspace(-3.0, 3.0, 20)
            savemat(
                path,
                {
                    "run": {
                        "SA": sa,
                        "FY": -500.0 * sa,
                        "FZ": np.full_like(sa, -900.0),
                        "IA": np.array([[0.0]]),      # scalar constant in struct
                        "P": np.array([[12.0]]),      # scalar constant in psi
                        "V": np.array(["40 kmh"]),    # non-numeric metadata sharing alias
                    }
                },
            )
            df = TireDataManager.load_unfiltered(RunConfig(str(path), "Cornering"))
            self.assertEqual(len(df), 20)
            self.assertTrue((df["IA"] == 0.0).all())
            self.assertTrue(np.allclose(df["P"], 12.0))
            self.assertTrue(df["V"].isna().all())

    def test_csv_european_semicolon_units_header_and_bracketed_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ttc_export.txt"
            lines = [
                "# Calspan TTC Export Header",
                "Slip Angle [deg];Lateral Force (N);Normal Load [N];Camber [deg];Pressure [psi];Velocity [km/h]",
                "deg;N;N;deg;psi;km/h",
                "-1,5;620,5;-900,2;-2,0;12,0;40,2",
                "-0,5;210,0;-899,5;-2,0;12,0;40,2",
                "0,5;-205,0;-901,0;-2,0;12,0;40,2",
                "1,5;-615,2;-900,8;-2,0;12,0;40,2",
            ]
            path.write_text("\n".join(lines), encoding="utf-8")
            df = TireDataManager.load_unfiltered(RunConfig(str(path), "Cornering"))
            self.assertEqual(len(df), 4)
            self.assertAlmostEqual(float(df["SA"].iloc[0]), -1.5, places=3)
            self.assertAlmostEqual(float(df["FY"].iloc[0]), 620.5, places=3)
            self.assertAlmostEqual(float(df["FZ"].iloc[0]), 900.2, places=3)
            self.assertAlmostEqual(float(df["P"].iloc[0]), 12.0, places=3)
            self.assertEqual(df.attrs.get("unit_hints", {}).get("Pressure [psi]"), "psi")


if __name__ == "__main__":
    unittest.main()
