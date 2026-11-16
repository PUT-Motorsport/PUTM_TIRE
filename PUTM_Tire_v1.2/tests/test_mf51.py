import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mf51_model import MF51Model, DEFAULT_MF51_PARAMS


class MF51ModelTests(unittest.TestCase):
    def setUp(self):
        self.model = MF51Model()

    def test_fy_evaluation_and_peaks(self):
        alpha_deg = np.linspace(-15, 15, 301)
        alpha_rad = np.deg2rad(alpha_deg)
        fy, meta = self.model.evaluate_fy(alpha_rad, fz=1000.0, gamma_rad=0.0)

        self.assertEqual(len(fy), len(alpha_deg))
        self.assertAlmostEqual(meta["mu_y"], self.model.get("PDY1"), places=4)
        self.assertAlmostEqual(meta["Dy"], self.model.get("PDY1") * 1000.0, places=2)
        # Force at alpha=0 should be close to 0 with zero camber and zero shift
        idx_zero = len(alpha_deg) // 2
        self.assertAlmostEqual(fy[idx_zero], 0.0, places=1)
        # Peak force should be reached
        self.assertAlmostEqual(np.max(fy), meta["Dy"], delta=20.0)

    def test_fy_camber_effect(self):
        alpha_zero = np.array([0.0])
        # With positive camber, camber thrust produces a lateral force at zero slip angle
        _, meta_0 = self.model.evaluate_fy(alpha_zero, fz=1000.0, gamma_rad=0.0)
        fy_camber, meta_camber = self.model.evaluate_fy(alpha_zero, fz=1000.0, gamma_rad=np.deg2rad(4.0))

        self.assertNotEqual(fy_camber[0], 0.0)
        self.assertGreater(meta_camber["Svy"], meta_0["Svy"])

    def test_fx_evaluation(self):
        kappa = np.linspace(-0.3, 0.3, 201)
        fx, meta = self.model.evaluate_fx(kappa, fz=1000.0)

        self.assertEqual(len(fx), len(kappa))
        self.assertAlmostEqual(meta["Dx"], self.model.get("PDX1") * 1000.0, places=2)
        self.assertAlmostEqual(fx[100], 0.0, places=1)
        self.assertAlmostEqual(np.max(fx), meta["Dx"], delta=20.0)

    def test_mz_aligning_moment_and_trail(self):
        alpha_deg = np.linspace(-15, 15, 301)
        alpha_rad = np.deg2rad(alpha_deg)
        mz, trail, mzr, meta = self.model.evaluate_mz(alpha_rad, fz=1000.0, gamma_rad=0.0)

        self.assertEqual(len(mz), len(alpha_deg))
        self.assertEqual(len(trail), len(alpha_deg))
        # Trail should be positive at small slip angles (self-aligning effect)
        self.assertGreater(trail[150], 0.0)
        # Trail should drop towards zero at high slip angles (sliding tire)
        self.assertLess(abs(trail[0]), abs(trail[150]))
        # Max Mz should be positive and bounded
        self.assertGreater(meta["max_mz_Nm"], 5.0)

    def test_combined_slip(self):
        alpha_deg = np.array([4.0])
        alpha_rad = np.deg2rad(alpha_deg)

        fy_pure, _ = self.model.evaluate_fy(alpha_rad, fz=1000.0)
        fy_comb = self.model.evaluate_combined_fy(alpha_rad, kappa=0.2, fz=1000.0)

        # Lateral force should be reduced when longitudinal slip is present
        self.assertLess(fy_comb[0], fy_pure[0])

    def test_tir_generation(self):
        content = self.model.to_tir_content("TestTyre")
        self.assertIn("[MDI_HEADER]", content)
        self.assertIn("FILE_TYPE                ='tir'", content)
        self.assertIn("[LATERAL_COEFFICIENTS]", content)
        self.assertIn("PCY1", content)
        self.assertIn("PDY1", content)
        self.assertIn("[ALIGNING_COEFFICIENTS]", content)

    def test_import_from_fit_results(self):
        mock_fit_results = {
            "Fy_load": {
                "parameters": {
                    "pCy1": 1.45,
                    "pDy1": 1.72,
                    "pDy2": -0.22,
                    "pEy1": -0.45,
                    "pKy1": 42.0,
                    "pKy2": 1.9,
                    "pUy1": 0.005,
                    "pVy1": 0.01,
                    "fz0": 1100.0,
                }
            }
        }
        imported = self.model.import_from_fit_results(mock_fit_results)
        self.assertTrue(len(imported) > 0)
        self.assertAlmostEqual(self.model.get("PCY1"), 1.45)
        self.assertAlmostEqual(self.model.get("PDY1"), 1.72)
        self.assertAlmostEqual(self.model.get("FNOMIN"), 1100.0)


if __name__ == "__main__":
    unittest.main()
