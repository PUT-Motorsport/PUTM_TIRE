"""Unit tests for AutoMF51Extractor and MF51 auto-fitting functionality."""

import unittest
import numpy as np
import pandas as pd

from mf51_model import MF51Model
from auto_mf51_extractor import AutoMF51Extractor, _normalize_pressure_series, _find_clusters


class TestAutoMF51Extractor(unittest.TestCase):
    def test_find_clusters(self):
        values = np.array([405, 395, 401, 802, 798, 810, 1195, 1205])
        clusters = _find_clusters(values, threshold=50.0)
        self.assertEqual(len(clusters), 3)
        self.assertAlmostEqual(clusters[0], 400.0, delta=10)
        self.assertAlmostEqual(clusters[1], 803.0, delta=10)
        self.assertAlmostEqual(clusters[2], 1200.0, delta=10)

    def test_normalize_pressure_series(self):
        # psi (np. 12 psi ~ 82.7 kPa)
        p_psi = pd.Series([12.0, 12.1, 11.9, 12.0])
        p_kpa = _normalize_pressure_series(p_psi)
        self.assertAlmostEqual(p_kpa.iloc[0], 82.73, delta=0.5)

        # kPa (np. 80 kPa -> pozostaje 80 kPa)
        p_kpa_in = pd.Series([80.0, 82.0, 79.5])
        p_out = _normalize_pressure_series(p_kpa_in)
        self.assertAlmostEqual(p_out.iloc[0], 80.0, delta=0.1)

    def test_extract_from_synthetic_dataframe(self):
        # Generujemy syntetyczne dane dla 3 poziomów Fz: 500 N, 1000 N, 1500 N
        rows = []
        for fz in [500.0, 1000.0, 1500.0]:
            dfz = (fz - 1000.0) / 1000.0
            mu = 1.6 * (1.0 - 0.15 * dfz)
            d = mu * fz
            c = 1.3
            b = 14.0
            for sa in np.linspace(-12.0, 12.0, 31):
                sa_rad = np.deg2rad(sa)
                fy = d * np.sin(c * np.arctan(b * sa_rad))
                rows.append({"SA": sa, "FY": -fy, "FZ": fz, "P": 12.0, "IA": 0.0})

        df_corn = pd.DataFrame(rows)
        model = MF51Model()

        result = AutoMF51Extractor.extract_from_app_data(
            model=model,
            runs=[],
            df_cornering=df_corn,
            df_drive=None
        )

        self.assertTrue(result.success)
        self.assertIn("PDY1", result.parameters)
        self.assertIn("PDY2", result.parameters)
        self.assertAlmostEqual(result.parameters["FNOMIN"], 1000.0, delta=100.0)
        self.assertGreater(result.parameters["PDY1"], 1.0)
        self.assertLess(result.parameters["PDY2"], 0.0)  # tarcie spada z obciążeniem


if __name__ == "__main__":
    unittest.main()
