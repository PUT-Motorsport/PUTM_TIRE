import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fitting import fit_load_sensitive, fit_pure_magic
from simple_pacejka import (
    FrictionVsFz, LoadSensitiveFy, SimpleMagicFormula, evaluate, from_fit_result, from_physical,
    load_family, parse_fz_levels, project_dict, summarize, summarize_family, summarize_friction,
)
from tire_model import pacejka_fy, pacejka_fy_load


class SimplePacejkaTests(unittest.TestCase):
    def test_matches_tire_model_function(self):
        model = SimpleMagicFormula("Fy", B=15, C=1.3, D=1500, E=-0.5, Sh=0.002, Sv=10)
        alpha = np.linspace(-0.2, 0.2, 41)
        np.testing.assert_allclose(model(alpha), pacejka_fy(alpha, 15, 1.3, 1500, -0.5, 0.002, 10))

    def test_physical_inputs_give_expected_peak_and_stiffness(self):
        model = from_physical("Fy", mu=1.5, fz=1000, stiffness=600, C=1.3, E=-0.5)
        self.assertAlmostEqual(model.D, 1500.0)
        summary = summarize(model, half_range=15)
        self.assertAlmostEqual(summary["initial_stiffness"], 600.0, delta=1.0)
        self.assertAlmostEqual(abs(summary["peak_force"]), 1500.0, delta=15.0)

    def test_fx_stiffness_is_per_unit_slip(self):
        model = from_physical("Fx", mu=1.6, fz=1000, stiffness=60000, C=1.65, E=-0.5)
        self.assertAlmostEqual(summarize(model, 0.3)["initial_stiffness"], 60000.0, delta=100.0)

    def test_mz_physical_mode_rejected(self):
        with self.assertRaises(ValueError):
            from_physical("Mz", mu=1, fz=1000, stiffness=100)

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            from_physical("Fy", mu=-1, fz=1000, stiffness=600)
        with self.assertRaises(ValueError):
            SimpleMagicFormula("Fz", 1, 1, 1, 0)

    def test_warnings(self):
        model = SimpleMagicFormula("Fy", B=15, C=3.0, D=1500, E=1.5)
        text = " ".join(summarize(model, 15)["warnings"])
        self.assertIn("C outside", text)
        self.assertIn("E > 1", text)
        narrow = SimpleMagicFormula("Fy", B=2, C=1.3, D=1500, E=0)
        self.assertIn("boundary", " ".join(summarize(narrow, 2)["warnings"]))

    def test_project_dict_schema(self):
        model = SimpleMagicFormula("Fy", B=15, C=1.3, D=1500, E=-0.5)
        project = project_dict({"Fy": model})
        self.assertEqual(project["schema_version"], 2)
        self.assertEqual(project["fits"]["Fy"]["slip_unit"], "rad")
        self.assertEqual(project["fits"]["Fy"]["model"], "Fy0_manual")

    def test_evaluate_axis_is_symmetric(self):
        shown, force = evaluate(SimpleMagicFormula("Fy", 15, 1.3, 1500, -0.5), 10, n=101)
        self.assertEqual(len(shown), len(force))
        self.assertAlmostEqual(float(shown[0]), -10.0)
        self.assertAlmostEqual(float(shown[-1]), 10.0)


class LoadSensitiveTests(unittest.TestCase):
    def setUp(self):
        self.model = LoadSensitiveFy(1.3, 1.5, -0.15, -0.5, 40.0, 1.5, fz0=1000.0)

    def test_matches_tire_model_function(self):
        alpha = np.linspace(-0.2, 0.2, 41)
        fz = np.full_like(alpha, 800.0)
        expected = pacejka_fy_load((alpha, fz), 1.3, 1.5, -0.15, -0.5, 40.0, 1.5, 0.0, 0.0, fz0=1000.0)
        np.testing.assert_allclose(self.model(alpha, fz), expected)

    def test_family_has_load_sensitive_mu_and_rising_stiffness(self):
        rows = summarize_family(self.model, [600, 1000, 1400], 15)["rows"]
        mus = [row["mu_peak"] for row in rows]
        stiff = [row["stiffness_n_per_deg"] for row in rows]
        self.assertTrue(mus[0] > mus[1] > mus[2])        # pDy2 < 0
        self.assertTrue(stiff[0] < stiff[1] < stiff[2])  # rośnie z Fz
        self.assertAlmostEqual(rows[1]["mu_peak"], 1.5, delta=0.02)

    def test_invalid_levels_and_params(self):
        with self.assertRaises(ValueError):
            parse_fz_levels("abc")
        with self.assertRaises(ValueError):
            parse_fz_levels("0, 500")
        with self.assertRaises(ValueError):
            parse_fz_levels(", ".join(["500"] * 9))
        self.assertEqual(parse_fz_levels("600, 800;1000 1200"), [600, 800, 1000, 1200])
        with self.assertRaises(ValueError):
            LoadSensitiveFy(1.3, 1.5, -0.1, -0.5, 40, 0.0)
        with self.assertRaises(ValueError):
            load_family(self.model, [])

    def test_warns_when_mu_grows_with_load(self):
        bad = LoadSensitiveFy(1.3, 1.5, 0.2, -0.5, 40.0, 1.5)
        self.assertIn("pDy2", " ".join(summarize_family(bad, [800, 1200], 15)["warnings"]))


class FitRoundTripTests(unittest.TestCase):
    """Parametry z prawdziwego fitu optimizera muszą odtwarzać tę samą krzywą w kreatorze."""

    def test_pure_fy_fit_round_trip(self):
        rng = np.random.default_rng(0)
        alpha = np.deg2rad(np.linspace(-12, 12, 400))
        truth = pacejka_fy(alpha, 14.0, 1.3, 1600.0, -0.4, 0.002, 20.0)
        output = fit_pure_magic(alpha, truth + rng.normal(0, 5, alpha.size), model_name="Fy0", slip_unit="rad")
        model = from_fit_result({**output.parameters, **output.diagnostics})
        self.assertEqual(model.channel, "Fy")
        np.testing.assert_allclose(model(alpha), output.prediction, rtol=1e-9, atol=1e-6)

    def test_fx_and_mz_key_names(self):
        fx = {"model": "Fx0", "Bx": 20.0, "Cx": 1.65, "Dx": 1500.0, "Ex": -0.3, "Shx": 0.001, "Svx": 5.0}
        mz = {"model": "Mz0", "Bz": 9.0, "Cz": 1.2, "Dz": 25.0, "Ez": -1.0, "Shz": 0.0, "Svz": 0.5}
        self.assertEqual((from_fit_result(fx).channel, from_fit_result(fx).B), ("Fx", 20.0))
        self.assertEqual((from_fit_result(mz).channel, from_fit_result(mz).D), ("Mz", 25.0))

    def test_load_fit_round_trip(self):
        rng = np.random.default_rng(1)
        truth = LoadSensitiveFy(1.3, 1.5, -0.15, -0.5, 40.0, 1.5, fz0=1000.0)
        alpha = np.tile(np.deg2rad(np.linspace(-12, 12, 120)), 4)
        fz = np.repeat([600.0, 800.0, 1000.0, 1200.0], 120)
        force = truth(alpha, fz) + rng.normal(0, 4, alpha.size)
        output = fit_load_sensitive(alpha, fz, force)
        model = from_fit_result({**output.parameters, **output.diagnostics})
        self.assertIsInstance(model, LoadSensitiveFy)
        np.testing.assert_allclose(model(alpha, fz), output.prediction, rtol=1e-9, atol=1e-6)

    def test_friction_vs_fz_evaluation(self):
        m = FrictionVsFz(pDy1=1.5, pDy2=-0.15, pDx1=1.65, pDx2=-0.18, fz0=1000.0)
        mu_y, mu_x, Dy, Dx = m.evaluate(np.array([1000.0, 2000.0]))
        # At Fz=Fz0 (1000N):
        self.assertAlmostEqual(mu_y[0], 1.5)
        self.assertAlmostEqual(mu_x[0], 1.65)
        self.assertAlmostEqual(Dy[0], 1500.0)
        self.assertAlmostEqual(Dx[0], 1650.0)
        # At Fz=2000N (dfz=1.0):
        self.assertAlmostEqual(mu_y[1], 1.5 - 0.15)
        self.assertAlmostEqual(mu_x[1], 1.65 - 0.18)
        self.assertAlmostEqual(Dy[1], (1.5 - 0.15) * 2000.0)

        s = summarize_friction(m, fz_max=2200.0)
        self.assertEqual(len(s["rows"]), 6)
        self.assertEqual(len(s["warnings"]), 0)

    def test_unknown_model_rejected(self):
        with self.assertRaises(ValueError):
            from_fit_result({"model": "CombinedSlip"})


if __name__ == "__main__":
    unittest.main()
