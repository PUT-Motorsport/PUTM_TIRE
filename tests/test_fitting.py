import unittest

import numpy as np
import pandas as pd

from fitting import build_combined_slip_targets, fit_combined_slip, fit_load_sensitive, fit_pure_magic
from tire_model import pacejka_comb_Gyk, pacejka_fy, pacejka_fy_load


class PureMagicFitTests(unittest.TestCase):
    def test_recovers_synthetic_lateral_curve(self):
        rng = np.random.default_rng(7)
        alpha = np.deg2rad(np.linspace(-12.0, 12.0, 700))
        expected = [11.5, 1.3, 2200.0, -0.8, np.deg2rad(0.15), 25.0]
        force = pacejka_fy(alpha, *expected) + rng.normal(0.0, 18.0, alpha.size)

        output = fit_pure_magic(alpha, force, model_name="Fy0", slip_unit="rad")

        self.assertLess(output.diagnostics["nrmse"], 0.02)
        self.assertAlmostEqual(output.parameters["D"], expected[2], delta=120.0)
        self.assertAlmostEqual(output.parameters["Sh"], expected[4], delta=np.deg2rad(0.12))
        self.assertEqual(output.parameters["C"], 1.3)

    def test_load_sensitive_fit_handles_multiple_load_levels(self):
        rng = np.random.default_rng(11)
        alpha_line = np.deg2rad(np.linspace(-11.0, 11.0, 180))
        alpha = np.tile(alpha_line, 3)
        fz = np.repeat([600.0, 1000.0, 1400.0], alpha_line.size)
        parameters = [1.3, 1.65, -0.20, -0.6, 16.0, 2.2, np.deg2rad(0.1), 0.01]
        force = pacejka_fy_load((alpha, fz), *parameters, fz0=1000.0)
        force += rng.normal(0.0, 12.0, force.size)

        output = fit_load_sensitive(alpha, fz, force)

        self.assertLess(output.diagnostics["nrmse"], 0.02)
        self.assertEqual(output.diagnostics["n_load_levels_100N"], 3)
        self.assertAlmostEqual(output.parameters["fz0"], 1000.0)


class CombinedSlipFitTests(unittest.TestCase):
    def _synthetic_frame(self):
        rows = []
        for alpha_deg in (-3.0, -6.0):
            for fz in (700.0, 1100.0):
                fy0 = 1.55 * fz * np.tanh(abs(alpha_deg) / 3.0)
                for kappa in np.linspace(-0.22, 0.22, 91):
                    gyk = pacejka_comb_Gyk(kappa, 8.0, 1.15)
                    rows.append({"SA": alpha_deg, "IA": 0.0, "FZ": fz, "SL": kappa, "FY": -fy0 * gyk})
        return pd.DataFrame(rows)

    def test_uses_measured_zero_slip_baselines(self):
        targets = build_combined_slip_targets(self._synthetic_frame())
        at_zero = targets[np.abs(targets["SL"]) < 1e-12]
        self.assertTrue(np.allclose(at_zero["GYK"], 1.0, atol=0.01))
        self.assertEqual(targets[["SA_GROUP", "IA_GROUP", "FZ_GROUP"]].drop_duplicates().shape[0], 4)

    def test_recovers_synthetic_reduction(self):
        output = fit_combined_slip(self._synthetic_frame())
        self.assertLess(output.diagnostics["nrmse"], 0.015)
        self.assertAlmostEqual(output.parameters["rBy1"], 8.0, delta=0.8)
        self.assertAlmostEqual(output.parameters["rCy1"], 1.15, delta=0.12)


if __name__ == "__main__":
    unittest.main()
